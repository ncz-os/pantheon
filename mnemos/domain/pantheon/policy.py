"""Rolling-window adaptive routing policy for PANTHEON v0.2.

This module also owns the PANTHEON-wide "adaptive routing is degraded" status
flag, exposed via :func:`is_adaptive_routing_degraded` and surfaced on
:class:`RouteDecision` as ``adaptive_routing_degraded``. Callers and operators
can read this flag to learn that the rolling-window telemetry backend was
unavailable on the most recent resolution and that routing fell through to the
cheapest-first safe default.

The flag is process-global — PANTHEON has a single routing pipeline per worker,
and "degraded right now" is a property of that worker, not of an individual
alias. The companion warning log is rate-limited (see ``_LOG_THROTTLE``) so a
sustained telemetry outage does not flood the operator's log pipeline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from mnemos.core.config import get_settings
from mnemos.core.numeric import safe_float

logger = logging.getLogger(__name__)


# How long, in seconds, to suppress repeated "telemetry unavailable" warnings.
# A sustained outage with one warning per minute is enough to make the degraded
# state discoverable from logs without flooding the operator's log pipeline.
_LOG_THROTTLE_SECONDS = 60.0


# Module-level state — see module docstring for the rationale.
_adaptive_routing_degraded: bool = False
_last_degradation_warning_at: float = 0.0


def is_adaptive_routing_degraded() -> bool:
    """Return ``True`` if the most recent :func:`resolve_with_policy` call
    fell back to cheapest-first because no rolling-window telemetry was
    available.

    Process-global. The flag is cleared when telemetry becomes available
    again, and re-set whenever a subsequent resolution fails to read it.
    Operator dashboards and the ``/health`` endpoint read this flag.
    """
    return _adaptive_routing_degraded


def _mark_degraded(reason: str) -> None:
    """Flip the global flag and emit a rate-limited WARNING.

    The warning is suppressed if a prior warning for the same condition fired
    within ``_LOG_THROTTLE_SECONDS``. The flag itself is always set so callers
    who poll :func:`is_adaptive_routing_degraded` see the current state.
    """
    global _adaptive_routing_degraded, _last_degradation_warning_at
    _adaptive_routing_degraded = True
    now = time.monotonic()
    if now - _last_degradation_warning_at < _LOG_THROTTLE_SECONDS:
        return
    _last_degradation_warning_at = now
    logger.warning(
        "[PANTHEON] adaptive routing degraded: rolling-window telemetry "
        "unavailable, falling back to cheapest-first policy (%s). "
        "Callers/operators can inspect "
        "RouteDecision.adaptive_routing_degraded or call "
        "mnemos.domain.pantheon.policy.is_adaptive_routing_degraded() "
        "to confirm. Next warning will be suppressed for up to "
        "%.0fs to avoid log spam.",
        reason,
        _LOG_THROTTLE_SECONDS,
    )


def _mark_healthy() -> None:
    """Clear the global flag — telemetry is back."""
    global _adaptive_routing_degraded
    _adaptive_routing_degraded = False


def reset_degradation_flag_for_tests() -> None:
    """Reset module-level state. Test-only helper."""
    global _adaptive_routing_degraded, _last_degradation_warning_at
    _adaptive_routing_degraded = False
    _last_degradation_warning_at = 0.0


@dataclass(frozen=True)
class ResolvedRoute:
    selected: dict[str, Any]
    candidates: list[str]
    rolling_window_minutes: int
    scores: dict[str, dict[str, Any]]
    selection_reason: str
    telemetry_available: bool
    # PANTHEON telemetry-loss-reporting: explicit degradation flag surfaced on
    # the route decision so callers/operators can detect when adaptive routing
    # has silently fallen back to cheapest-first. ``True`` whenever the rolling
    # window produced no usable telemetry rows (table empty, query errored,
    # pool unavailable). Mirrors :func:`is_adaptive_routing_degraded` at the
    # moment of the call.
    adaptive_routing_degraded: bool = False


def _candidate_backend(candidate: dict[str, Any] | str) -> str:
    if isinstance(candidate, str):
        return candidate
    return str(candidate.get("id") or candidate.get("model_id") or "")


def _cost_sort_value(candidate: dict[str, Any] | str) -> tuple[bool, float, str]:
    if isinstance(candidate, str):
        return (True, float("inf"), candidate)
    cost = candidate.get("cost_per_mtok")
    if cost is None:
        in_cost = candidate.get("input_cost_per_mtok", candidate.get("price_in"))
        out_cost = candidate.get("output_cost_per_mtok", candidate.get("price_out"))
        if in_cost is not None and out_cost is not None:
            cost = (safe_float(in_cost) + safe_float(out_cost)) / 2.0
    quality = -safe_float(candidate.get("quality_score") or candidate.get("graeae_weight") or 0.0)
    latency = safe_float(candidate.get("p50_latency_ms") or candidate.get("latency_ms") or 0.0)
    return (cost is None, safe_float(cost) if cost is not None else float("inf"), quality, latency, _candidate_backend(candidate))


def _fallback_cheapest(candidates: list[dict[str, Any] | str]) -> dict[str, Any] | str:
    return sorted(candidates, key=_cost_sort_value)[0]


def _normalize_lower(value: float | None, values: list[float]) -> float:
    if value is None or not values:
        return 1.0
    low = min(values)
    high = max(values)
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _weight_settings() -> tuple[float, float, float]:
    settings = get_settings().pantheon
    latency = float(settings.policy_latency_weight)
    error = float(settings.policy_error_weight)
    cost = float(settings.policy_cost_weight)
    total = latency + error + cost
    if total <= 0:
        return 0.40, 0.40, 0.20
    return latency / total, error / total, cost / total


async def _fetch_window(pool: Any, candidate_list: list[str], window_minutes: int) -> list[Any]:
    if pool is None or not candidate_list:
        return []
    async with pool.acquire() as conn:
        return list(
            await conn.fetch(
                """
                SELECT resolved_to AS backend,
                       AVG(latency_ms::FLOAT) AS avg_latency_ms,
                       SUM(CASE WHEN outcome = 'error' THEN 1 ELSE 0 END)::FLOAT
                         / COUNT(*)::FLOAT AS error_rate,
                       AVG(cost_usd::FLOAT) AS avg_cost
                FROM pantheon_routing_audit
                WHERE created > NOW() - ($1::int * INTERVAL '1 minute')
                  AND resolved_to = ANY($2::text[])
                GROUP BY backend
                """,
                int(window_minutes),
                candidate_list,
            )
        )


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if hasattr(row, "get"):
        return row.get(key, default)
    try:
        return row[key]
    except (KeyError, TypeError):
        return default


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_candidates(
    candidates: list[dict[str, Any] | str],
    rows: list[Any],
) -> dict[str, dict[str, Any]]:
    telemetry: dict[str, dict[str, float | None]] = {}
    for row in rows:
        backend = str(_row_get(row, "backend") or "")
        if not backend:
            continue
        telemetry[backend] = {
            "avg_latency_ms": _maybe_float(_row_get(row, "avg_latency_ms")),
            "error_rate": _maybe_float(_row_get(row, "error_rate")),
            "avg_cost": _maybe_float(_row_get(row, "avg_cost")),
        }

    latency_values = [
        float(item["avg_latency_ms"])
        for item in telemetry.values()
        if item.get("avg_latency_ms") is not None
    ]
    cost_values = [
        float(item["avg_cost"])
        for item in telemetry.values()
        if item.get("avg_cost") is not None
    ]
    latency_weight, error_weight, cost_weight = _weight_settings()
    scores: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        backend = _candidate_backend(candidate)
        metrics = telemetry.get(backend, {})
        avg_latency_ms = metrics.get("avg_latency_ms")
        error_rate = metrics.get("error_rate")
        avg_cost = metrics.get("avg_cost")
        latency_component = _normalize_lower(
            float(avg_latency_ms) if avg_latency_ms is not None else None,
            latency_values,
        )
        error_component = max(0.0, min(1.0, float(error_rate))) if error_rate is not None else 1.0
        cost_component = _normalize_lower(float(avg_cost) if avg_cost is not None else None, cost_values)
        weighted_score = (
            latency_weight * latency_component
            + error_weight * error_component
            + cost_weight * cost_component
        )
        scores[backend] = {
            "avg_latency_ms": avg_latency_ms,
            "error_rate": error_rate,
            "avg_cost": avg_cost,
            "weighted_score": weighted_score,
            "telemetry_counted": backend in telemetry,
        }
    return scores


async def resolve_with_policy(
    pool: Any,
    alias: str,
    candidates: list[dict[str, Any] | str],
    *,
    window_minutes: int = 15,
) -> ResolvedRoute:
    """Pick the best candidate for an ``auto:*`` alias using recent routing telemetry."""
    if not candidates:
        raise ValueError("candidates must not be empty")

    candidate_list = [_candidate_backend(candidate) for candidate in candidates]
    try:
        rows = await _fetch_window(pool, candidate_list, window_minutes)
        fetch_error: str | None = None
    except Exception as exc:
        # Telemetry backend itself failed (Postgres down, pool closed,
        # permissions, …). PANTHEON telemetry-loss-reporting: previously
        # this path was *silent* — the caller got a normal-looking
        # cheapest-first decision with no signal that the smarter policy
        # never ran. Surface the degradation explicitly so callers and
        # operators can see it.
        rows = []
        fetch_error = f"{type(exc).__name__}: {exc}"
        logger.debug("[PANTHEON] rolling-window policy unavailable for %s: %s", alias, exc)

    scores = _score_candidates(candidates, rows)
    telemetry_available = any(score.get("telemetry_counted") for score in scores.values())
    if not telemetry_available:
        # Telemetry is either unavailable (fetch errored) or empty (table
        # has no rows for the requested candidates). Either way adaptive
        # routing has degraded to cheapest-first; mark the global flag
        # and emit a rate-limited warning so an operator scanning the log
        # sees WHY the request ended up on the cheapest candidate.
        if fetch_error is not None:
            _mark_degraded(
                f"alias={alias!r} telemetry fetch raised {fetch_error}"
            )
        else:
            _mark_degraded(
                f"alias={alias!r} telemetry table returned no rows for "
                f"{len(candidate_list)} candidates within "
                f"{window_minutes}m window"
            )
        selected = _fallback_cheapest(candidates)
        return ResolvedRoute(
            selected=dict(selected) if isinstance(selected, dict) else {"id": selected},
            candidates=candidate_list,
            rolling_window_minutes=window_minutes,
            scores=scores,
            selection_reason="fallback cheapest-first policy (no rolling telemetry)",
            telemetry_available=False,
            adaptive_routing_degraded=True,
        )

    # Telemetry is healthy — clear the global flag so callers who poll
    # ``is_adaptive_routing_degraded`` see the recovery.
    _mark_healthy()

    selected_backend = min(
        scores,
        key=lambda backend: (
            float(scores[backend]["weighted_score"]),
            _cost_sort_value(next(candidate for candidate in candidates if _candidate_backend(candidate) == backend)),
        ),
    )
    selected = next(candidate for candidate in candidates if _candidate_backend(candidate) == selected_backend)
    return ResolvedRoute(
        selected=dict(selected) if isinstance(selected, dict) else {"id": selected},
        candidates=candidate_list,
        rolling_window_minutes=window_minutes,
        scores=scores,
        selection_reason="best weighted score (latency+error+cost)",
        telemetry_available=True,
        adaptive_routing_degraded=False,
    )


__all__ = [
    "ResolvedRoute",
    "is_adaptive_routing_degraded",
    "reset_degradation_flag_for_tests",
    "resolve_with_policy",
]
