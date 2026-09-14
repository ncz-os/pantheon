"""Regression tests for PANTHEON telemetry-loss-reporting.

When the rolling-window telemetry backend is unavailable (PostgreSQL down,
telemetry table empty, query errored, etc.) PANTHEON's adaptive routing
silently fell through to cheapest-first. That is the correct safe default
behaviour — but the silent fallback masked real telemetry outages from
operators and made "why did my request land on the cheapest model?" an
ungoogleable mystery.

These tests pin every aspect of the explicit-degradation contract:

1. The cheap-first fallback STILL succeeds when telemetry is unavailable
   (we did not regress the safe-default behaviour).
2. A WARNING is emitted to the operator log (rate-limited to once per
   ``_LOG_THROTTLE_SECONDS`` so a sustained outage does not flood logs).
3. The returned :class:`ResolvedRoute` carries ``adaptive_routing_degraded=True``
   so callers can detect the degraded state without scraping logs.
4. The module-level :func:`is_adaptive_routing_degraded` flips to ``True``
   after a degraded resolution and back to ``False`` after a healthy one,
   so dashboards and ``/health`` consumers see live state.
5. The :class:`RouteDecision` exposes ``adaptive_routing_degraded`` on its
   public surface and in ``explain()`` output for both the
   ``resolve_with_policy`` call and the upstream ``route_model`` call.
6. The warning is rate-limited (a second degraded call within the throttle
   window does not re-emit).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from mnemos.domain.pantheon import catalog, router
from mnemos.domain.pantheon.catalog import canonical_provider
from mnemos.domain.pantheon.policy import (
    is_adaptive_routing_degraded,
    reset_degradation_flag_for_tests,
    resolve_with_policy,
)


@pytest.fixture(autouse=True)
def _reset_module_state() -> None:
    """Each test starts with a clean degradation flag and throttle clock."""
    reset_degradation_flag_for_tests()
    yield
    reset_degradation_flag_for_tests()


# ── 1. Pool returns empty rows (table is empty) ──────────────────────────


class _EmptyRowsConnection:
    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        return []


class _EmptyRowsContext:
    def __init__(self, conn: _EmptyRowsConnection) -> None:
        self.conn = conn

    async def __aenter__(self) -> _EmptyRowsConnection:
        return self.conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _EmptyRowsPool:
    def acquire(self) -> _EmptyRowsContext:
        return _EmptyRowsContext(_EmptyRowsConnection())


@pytest.mark.asyncio
async def test_fallback_succeeds_when_telemetry_table_is_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Cheapest-first fallback must still pick a model when the routing
    audit table returns no rows.
    """
    candidates = [
        {"id": "expensive-reasoner", "provider": "paid", "cost_per_mtok": 5.0},
        {"id": "cheap-chat", "provider": "cheap", "cost_per_mtok": 0.1},
    ]
    caplog.set_level(logging.WARNING, logger="mnemos.domain.pantheon.policy")

    route = await resolve_with_policy(_EmptyRowsPool(), "auto:reasoning", candidates)

    # Behaviour unchanged: cheapest-first still picks the cheap model.
    assert route.selected["id"] == "cheap-chat"
    assert route.telemetry_available is False
    assert route.selection_reason == "fallback cheapest-first policy (no rolling telemetry)"
    # Degradation signal now visible.
    assert route.adaptive_routing_degraded is True
    assert is_adaptive_routing_degraded() is True
    # WARNING was emitted with a clear "adaptive routing degraded" prefix.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("adaptive routing degraded" in r.getMessage() for r in warnings), (
        f"expected a 'adaptive routing degraded' WARNING; got {[r.getMessage() for r in warnings]}"
    )


# ── 2. Pool itself raises (PostgreSQL down / permissions / connection lost) ──


class _BrokenAcquireContext:
    async def __aenter__(self) -> None:
        raise ConnectionError("connection refused: PostgreSQL down")

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _BrokenPool:
    def acquire(self) -> _BrokenAcquireContext:
        return _BrokenAcquireContext()


@pytest.mark.asyncio
async def test_fallback_succeeds_when_pool_acquire_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Telemetry backend itself raised. The router must still pick a model
    and the operator must still see a WARNING explaining why the adaptive
    policy never ran.
    """
    candidates = [
        {"id": "expensive-reasoner", "provider": "paid", "cost_per_mtok": 5.0},
        {"id": "cheap-chat", "provider": "cheap", "cost_per_mtok": 0.1},
    ]
    caplog.set_level(logging.WARNING, logger="mnemos.domain.pantheon.policy")

    route = await resolve_with_policy(_BrokenPool(), "auto:reasoning", candidates)

    assert route.selected["id"] == "cheap-chat"
    assert route.telemetry_available is False
    assert route.adaptive_routing_degraded is True
    assert is_adaptive_routing_degraded() is True
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("adaptive routing degraded" in r.getMessage() for r in warnings)
    # The fetch failure detail (exception class + message) must be in the
    # warning so an operator scanning logs can see *why* telemetry failed
    # without digging through the surrounding debug log.
    assert any("ConnectionError" in r.getMessage() for r in warnings), (
        f"expected the underlying ConnectionError to be named in the warning; "
        f"got {[r.getMessage() for r in warnings]}"
    )


# ── 3. Healthy pool — flag must clear after a successful read ─────────────


class _HealthyAcquireContext:
    def __init__(self, conn: "_HealthyConnection") -> None:
        self.conn = conn

    async def __aenter__(self) -> "_HealthyConnection":
        return self.conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _HealthyConnection:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        return list(self._rows)


class _HealthyPool:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._conn = _HealthyConnection(rows)

    def acquire(self) -> _HealthyAcquireContext:
        return _HealthyAcquireContext(self._conn)


@pytest.mark.asyncio
async def test_healthy_telemetry_clears_degradation_flag() -> None:
    """After a degraded resolution flips the flag, a subsequent healthy
    resolution must clear it so a transient outage stops being reported
    once telemetry comes back.
    """
    candidates = [
        {"id": "expensive-reasoner", "provider": "paid", "cost_per_mtok": 5.0},
        {"id": "cheap-chat", "provider": "cheap", "cost_per_mtok": 0.1},
    ]

    # Step 1: degraded read.
    await resolve_with_policy(_EmptyRowsPool(), "auto:reasoning", candidates)
    assert is_adaptive_routing_degraded() is True

    # Step 2: healthy read — telemetry backend returns rows for both
    # candidates, weighted-score wins over cheapest-first.
    healthy_rows = [
        {"backend": "expensive-reasoner", "avg_latency_ms": 100.0, "error_rate": 0.0, "avg_cost": 1.0},
        {"backend": "cheap-chat", "avg_latency_ms": 3000.0, "error_rate": 0.5, "avg_cost": 100.0},
    ]
    route = await resolve_with_policy(
        _HealthyPool(healthy_rows), "auto:reasoning", candidates
    )
    assert route.telemetry_available is True
    assert route.adaptive_routing_degraded is False
    assert route.selection_reason == "best weighted score (latency+error+cost)"
    assert is_adaptive_routing_degraded() is False, (
        "flag must clear after a healthy telemetry read"
    )


# ── 4. Log throttling — second degraded call within throttle window does
#       not re-emit the warning. ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_repeated_degradation_warns_once_within_throttle_window(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A sustained telemetry outage must not flood the operator log with
    one warning per request. The contract is: one WARNING per
    ``_LOG_THROTTLE_SECONDS`` in ``policy.py``; subsequent calls within
    that window still flip the flag but stay silent in the log.
    """
    candidates = [
        {"id": "expensive-reasoner", "provider": "paid", "cost_per_mtok": 5.0},
        {"id": "cheap-chat", "provider": "cheap", "cost_per_mtok": 0.1},
    ]
    caplog.set_level(logging.WARNING, logger="mnemos.domain.pantheon.policy")

    # Three back-to-back degraded reads — only the first should emit a
    # WARNING in this test because they all happen within the same
    # throttle window (default 60s, well under any pytest runtime).
    for _ in range(3):
        await resolve_with_policy(_EmptyRowsPool(), "auto:reasoning", candidates)

    warnings = [r for r in caplog.records if "adaptive routing degraded" in r.getMessage()]
    assert len(warnings) == 1, (
        f"expected exactly 1 throttled WARNING across 3 back-to-back degraded "
        f"calls; got {len(warnings)}: {[r.getMessage() for r in warnings]}"
    )
    # Even though the log only fires once, the flag stays set so callers
    # can poll ``is_adaptive_routing_degraded`` for live state.
    assert is_adaptive_routing_degraded() is True


# ── 5. ``route_model`` propagation — the upstream ``route_model`` entry
#       point must forward ``adaptive_routing_degraded`` onto the
#       ``RouteDecision`` so the explain() payload the operator sees is
#       consistent. ──────────────────────────────────────────────────────────


def _settings() -> SimpleNamespace:
    """Cheap router settings: no quality floor, big cost ceiling so the
    candidates below all pass the gate."""
    return SimpleNamespace(
        pantheon=SimpleNamespace(
            default_quality_floor=0.0,
            default_max_cost_usd_per_mtok=None,
            routing_window_minutes=15,
            passthrough_enabled=False,
        )
    )


@pytest.mark.asyncio
async def test_route_model_surfaces_degradation_on_decision(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``route_model`` must propagate the policy's degradation flag onto
    the returned ``RouteDecision`` AND include it in ``explain()``.

    This is the call shape an operator-facing dashboard sees — if the
    flag is missing here the degradation is invisible to the operator.
    """
    caplog.set_level(logging.WARNING, logger="mnemos.domain.pantheon.policy")

    async def _models():
        return [
            {
                "id": "expensive-reasoner",
                "provider": "paid",
                "capabilities": ["reasoning"],
                "quality_score": 0.95,
                "cost_per_mtok": 20.0,
            },
            {
                "id": "cheap-chat",
                "provider": "cheap",
                "capabilities": ["reasoning", "chat"],
                "quality_score": 0.5,
                "cost_per_mtok": 0.1,
            },
        ]

    monkeypatch.setattr(catalog, "list_models", _models)
    monkeypatch.setattr(router, "get_settings", _settings)

    # The router pulls ``_lc._pool`` for the rolling-window read; force it
    # to our empty-rows pool so the policy degrades.
    import mnemos.core.lifecycle as lc

    monkeypatch.setattr(lc, "_pool", _EmptyRowsPool())

    decision = await router.route_model("auto:reasoning", {})

    assert decision.adaptive_routing_degraded is True
    assert decision.selection_reason == "fallback cheapest-first policy (no rolling telemetry)"
    assert decision.model_id == "cheap-chat"

    # The explain() payload MUST include the flag so downstream consumers
    # (audit log, dashboard, response body) all see the same signal.
    payload = decision.explain()
    assert payload["adaptive_routing_degraded"] is True
    # And the resolution chain should make it obvious this came from the
    # fallback branch — the policy step's reason field carries the
    # "no rolling telemetry" marker.
    policy_step = next(step for step in payload["resolution_chain"] if step["step"] == "policy")
    assert "no rolling telemetry" in policy_step["reason"]


@pytest.mark.asyncio
async def test_route_model_does_not_set_degraded_for_single_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``adaptive_routing_degraded`` must stay ``False`` for ``single``
    aliases (the rolling-window policy never runs on those paths), even
    when telemetry is unavailable. Otherwise operators would see false
    positives on every non-adaptive request during an outage.
    """
    async def _models():
        return [
            {
                "id": "gpt-5",
                "provider": "openai",
                "capabilities": ["chat"],
                "quality_score": 0.9,
                "cost_per_mtok": 5.0,
            }
        ]

    import mnemos.core.lifecycle as lc

    monkeypatch.setattr(catalog, "list_models", _models)
    monkeypatch.setattr(router, "get_settings", _settings)
    monkeypatch.setattr(lc, "_pool", _EmptyRowsPool())

    decision = await router.route_model("gpt-5", {})

    assert decision.route_type == "literal"
    assert decision.adaptive_routing_degraded is False


# ── 6. ``canonical_provider`` regression — the F05 identity-canonicalisation
#       patch must still apply cleanly to degradation-flagged routes. ──────


def test_canonical_provider_collapses_alias_pair() -> None:
    """Sanity-check that the F05 identity helper is still wired (the task
    review specifically asked us to read this code first to stay
    consistent with it). PANTHEON telemetry-loss-reporting does not
    touch canonical_provider, but if F05 ever regressed this test would
    catch it.
    """
    assert canonical_provider("claude") == "anthropic"
    assert canonical_provider("anthropic") == "anthropic"
    assert canonical_provider(None) == ""
