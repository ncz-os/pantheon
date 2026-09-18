"""PANTHEON model catalog derived from the GRAEAE muses registry."""

from __future__ import annotations

import json
import logging
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import mnemos.core.lifecycle as _lc
from mnemos.core.config import get_settings
from mnemos.core.numeric import safe_float
from mnemos.core.provider_registry import GRAEAE_REGISTRY_MAP
from mnemos.domain.graeae.engine import get_graeae_engine

logger = logging.getLogger(__name__)

# Set once by _warn_registry_overlay_unavailable() so the missing overlay is
# reported a single time per process rather than on every catalog read.
_registry_overlay_warned = False
CATALOG_CACHE_SCHEMA = "mnemos.pantheon.catalog.v1"


def canonical_provider(provider: str | None) -> str:
    """Return the single canonical provider identity for a PANTHEON ``provider`` string.

    PANTHEON's routing layer speaks the GRAEAE provider name (e.g. ``claude`` for
    Anthropic models); KNEMON's budget ledger is keyed on the registry provider
    (e.g. ``anthropic``). Mixing the two on either side of an admission/ledger
    pair lets a single physical provider evade its cap under either spelling.
    Funneling every caller through this one helper collapses the alias pair into
    one identity, so admission (``provider=...``) and the recorded ledger row
    (``UsageLedgerRecord(provider=...)``) use the same key.

    Precedence:
      * known GRAEAE mapping: registry_provider wins (e.g. ``claude`` -> ``anthropic``).
      * fleet / passthrough aliases: the configured ``passthrough_provider`` wins.
      * otherwise: the lowercased, stripped input verbatim.

    The function is total and side-effect-free; it never raises so it can be
    called from hot paths and from logging.
    """
    raw = str(provider or "").strip()
    if not raw:
        return ""
    lower = raw.lower()
    mapping = GRAEAE_REGISTRY_MAP.get(lower)
    if mapping is not None:
        canonical = str(mapping.get("registry_provider") or "").strip().lower()
        if canonical:
            return canonical
    # Fleet / passthrough aliases (codex->openai, configured passthrough_provider).
    # A GRAEAE provider that IS the registry provider (e.g. ``openai``) maps to
    # itself; the explicit guards below avoid the round-trip through the registry
    # mapping (which would still produce the same answer but at extra cost).
    if lower == "codex":
        return "openai"
    try:
        configured = str(get_settings().pantheon.passthrough_provider or "").strip().lower()
    except Exception:  # noqa: BLE001
        configured = ""
    if configured and lower == configured:
        return configured
    return lower


def canonical_model_id(model_id: str | None) -> str:
    """Return the canonical (lowercased, whitespace-stripped) form of a model id.

    The wire-id is what is sent to the upstream provider; aliases are
    PANTHEON-side and must not leak into the ledger (which would break
    per-model accounting). The function is total and side-effect-free.
    """
    return str(model_id or "").strip().lower()
DEFAULT_SNAPSHOT_TTL_SECONDS = 15.0
_DEFAULT_CACHE_PATHS = (
    "data/pantheon_catalog.json",
    "data/pantheon/catalog.json",
    "data/llm_provider_registry.json",
    "llm_provider_registry.json",
)
_CACHE_OVERLAY_KEYS = (
    "display_name",
    "capabilities",
    "usage_tier",
    "cost_per_mtok",
    "price_in",
    "price_out",
    "input_cost_per_mtok",
    "output_cost_per_mtok",
    "quality_score",
    "arena_rank",
    "graeae_weight",
    "release_date",
    "last_synced",
    "context_window",
    "model_max_ctx",
    "max_output_tokens",
    "p50_latency_ms",
    "available",
    "deprecated",
    "pricing_source",
    "pricing_fetched_at",
    "pricing_raw_model_id",
)
_PANTHEON_FLEET_MODELS: tuple[dict[str, Any], ...] = (
    {
        "provider": "nvidia",
        "id": "gpt-5.3-codex",
        "model_id": "openai/openai/gpt-5.3-codex",
        "display_name": "GPT-5.3 Codex",
        "capabilities": ["chat", "code", "reasoning", "tools"],
        "usage_tier": "frontier",
        "input_cost_per_mtok": 0.0,
        "output_cost_per_mtok": 0.0,
        "context_window": 200000,
        "max_output_tokens": 100000,
        "quality_score": 0.92,
        "pricing_raw_model_id": "gpt-5.3-codex",
        "available": True,
        "deprecated": False,
    },
    {
        "provider": "nvidia",
        "id": "gpt-5.5",
        "model_id": "openai/openai/gpt-5.5",
        "display_name": "GPT-5.5",
        "capabilities": ["chat", "reasoning", "tools", "vision"],
        "usage_tier": "frontier",
        "input_cost_per_mtok": 5.0,
        "output_cost_per_mtok": 30.0,
        "quality_score": 0.94,
        "pricing_raw_model_id": "gpt-5.5",
        "available": True,
        "deprecated": False,
    },
)
_FLEET_MODELS_BY_ID = {
    str(value).strip().lower(): model
    for model in _PANTHEON_FLEET_MODELS
    for value in (model.get("id"), model.get("model_id"), model.get("pricing_raw_model_id"))
    if str(value or "").strip()
}
_catalog_snapshots: dict[tuple[bool, object], tuple[float, tuple[dict[str, Any], ...]]] = {}


def _snapshot_ttl_seconds() -> float:
    raw = os.environ.get("MNEMOS_PANTHEON_CATALOG_SNAPSHOT_TTL_SECONDS")
    if raw is None:
        return DEFAULT_SNAPSHOT_TTL_SECONDS
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return DEFAULT_SNAPSHOT_TTL_SECONDS


def invalidate_catalog_snapshot() -> None:
    """Invalidate process-local routing snapshots after an explicit refresh."""
    _catalog_snapshots.clear()


async def catalog_snapshot(*, use_synced_cache: bool = True) -> list[dict[str, Any]]:
    """Return a short-lived, mutation-isolated catalog snapshot for hot paths."""
    ttl = _snapshot_ttl_seconds()
    if ttl <= 0:
        return await list_models() if use_synced_cache else await list_models(use_synced_cache=False)
    now = time.monotonic()
    # Including the loader itself keeps replacements isolated without risking
    # reuse of a discarded callable's numeric identity.
    key = (use_synced_cache, list_models)
    cached = _catalog_snapshots.get(key)
    if cached is not None and cached[0] > now:
        return deepcopy(list(cached[1]))
    models = await list_models() if use_synced_cache else await list_models(use_synced_cache=False)
    snapshot = tuple(deepcopy(models))
    _catalog_snapshots.clear()
    _catalog_snapshots[key] = (now + ttl, snapshot)
    return deepcopy(list(snapshot))


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if hasattr(row, "get"):
        return row.get(key, default)
    try:
        return row[key]
    except (KeyError, TypeError):
        return default


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _registry_provider(graeae_provider: str) -> str:
    mapping = GRAEAE_REGISTRY_MAP.get(graeae_provider)
    return mapping["registry_provider"] if mapping else graeae_provider


def _fleet_model(model_id: Any) -> dict[str, Any] | None:
    return _FLEET_MODELS_BY_ID.get(str(model_id or "").strip().lower())


def _fleet_provider(fleet: dict[str, Any]) -> str:
    try:
        configured = get_settings().pantheon.passthrough_provider
    except Exception:  # noqa: BLE001
        configured = None
    provider = str(configured or fleet.get("provider") or "").strip()
    return provider or "nvidia"


def _fleet_provider_registered(model_id: Any, provider_cfgs: dict[str, dict[str, Any]]) -> bool:
    fleet = _fleet_model(model_id)
    if not fleet:
        return True
    provider = _fleet_provider(fleet)
    return bool(provider and provider in provider_cfgs)


def _catalog_key(model: dict[str, Any]) -> tuple[str, str] | None:
    provider = str(model.get("provider") or "").strip()
    model_id = str(model.get("id") or model.get("model_id") or "").strip()
    if not provider or not model_id:
        return None
    return (provider, model_id.lower())


def _provider_aliases(model: dict[str, Any]) -> set[str]:
    aliases = {
        str(model.get("provider") or "").strip(),
        str(model.get("registry_provider") or "").strip(),
        str(model.get("owned_by") or "").strip(),
    }
    fleet = _fleet_model(model.get("id") or model.get("model_id"))
    if fleet:
        aliases.add(_fleet_provider(fleet))
    if "codex" in aliases:
        aliases.add("openai")
    return {alias for alias in aliases if alias}


def _cost_per_mtok(source: dict[str, Any]) -> float | None:
    explicit = source.get("cost_per_mtok")
    if explicit is not None:
        return safe_float(explicit)
    in_cost = source.get("price_in", source.get("input_cost_per_mtok"))
    out_cost = source.get("price_out", source.get("output_cost_per_mtok"))
    if in_cost is None or out_cost is None:
        cost = source.get("cost")
        if isinstance(cost, dict):
            in_cost = cost.get("input")
            out_cost = cost.get("output")
    if in_cost is None or out_cost is None:
        return None
    return (safe_float(in_cost) + safe_float(out_cost)) / 2.0


def _quality_score(source: dict[str, Any], provider_cfg: dict[str, Any]) -> float:
    for key in ("quality_score", "graeae_weight", "weight"):
        if source.get(key) is not None:
            return safe_float(source[key])
    if provider_cfg.get("weight") is not None:
        return safe_float(provider_cfg["weight"])
    return 0.0


def _infer_capabilities(model_id: str, provider_cfg: dict[str, Any]) -> list[str]:
    configured = provider_cfg.get("capabilities")
    if isinstance(configured, (list, tuple, set)):
        return sorted({str(cap).strip() for cap in configured if str(cap).strip()})

    caps = {"chat"}
    mid = model_id.lower()
    api = str(provider_cfg.get("api") or "").lower()

    if "embed" in mid:
        caps.add("embeddings")
    if any(token in mid for token in ("code", "coder", "codestral")):
        caps.add("code")
    if any(token in mid for token in ("reason", "thinking", "r1", "qwq", "o3", "o4")):
        caps.add("reasoning")
    if any(token in mid for token in ("claude", "gpt-5", "grok-4", "gemini-3")):
        caps.add("reasoning")
    if any(token in mid for token in ("vision", "vl", "4o", "gemini", "claude", "grok")):
        caps.add("vision")
    if any(token in mid for token in ("sonar", "search", "online", "perplexity")):
        caps.add("web_search")
    if api == "gemini":
        caps.add("vision")

    return sorted(caps)


def _usage_tier(source: dict[str, Any], quality_score: float, cost: float | None) -> str:
    explicit = source.get("usage_tier") or source.get("tier")
    if explicit:
        return str(explicit)
    if quality_score >= 0.95:
        return "frontier"
    if quality_score >= 0.85:
        return "premium"
    if cost is not None and cost <= 1.0:
        return "budget"
    return "standard"


def _provider_health(provider: str, status: dict[str, Any]) -> dict[str, Any]:
    def _as_dict(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    circuit = _as_dict((status.get("circuit_breakers") or {}).get(provider))
    quality = _as_dict((status.get("quality") or {}).get(provider))
    rate_limiter_raw = (status.get("rate_limiters") or {}).get(provider)
    # Some resilience backends return per-provider counters as scalars
    # (e.g. an int for "current limited count"). Coerce shape so the
    # health response stays consistent regardless of backend variant.
    if isinstance(rate_limiter_raw, dict):
        rate_limited_value: Any = rate_limiter_raw.get("limited")
    else:
        rate_limited_value = rate_limiter_raw
    concurrency = (status.get("concurrency") or {}).get(provider, {})
    return {
        "state": circuit.get("state") or "unknown",
        "success_rate": quality.get("success_rate"),
        "p50_latency_ms": quality.get("p50_latency_ms"),
        "rate_limited": rate_limited_value,
        "concurrency": concurrency,
    }


def _normalize_model(
    *,
    provider: str,
    provider_cfg: dict[str, Any],
    model_source: dict[str, Any],
    health: dict[str, Any],
) -> dict[str, Any]:
    catalog_id = str(model_source.get("id") or model_source.get("model_id") or provider_cfg.get("model") or "")
    wire_model_id = str(model_source.get("model_id") or provider_cfg.get("model") or catalog_id)
    display_name = str(model_source.get("display_name") or model_source.get("name") or catalog_id)
    capabilities = model_source.get("capabilities")
    if not isinstance(capabilities, (list, tuple, set)):
        capabilities = _infer_capabilities(wire_model_id, provider_cfg)
    capabilities = sorted({str(cap).strip() for cap in capabilities if str(cap).strip()})
    cost = _cost_per_mtok({**provider_cfg, **model_source})
    quality_score = _quality_score(model_source, provider_cfg)
    p50_latency_ms = model_source.get("p50_latency_ms") or provider_cfg.get("p50_latency_ms")
    if p50_latency_ms is None:
        p50_latency_ms = health.get("p50_latency_ms")

    return {
        "id": catalog_id,
        "model_id": wire_model_id,
        "object": "model",
        "created": int(model_source.get("created") or provider_cfg.get("created") or time.time()),
        "owned_by": str(model_source.get("owned_by") or provider),
        "provider": provider,
        "registry_provider": _registry_provider(provider),
        "display_name": display_name,
        "capabilities": capabilities,
        "usage_tier": _usage_tier({**provider_cfg, **model_source}, quality_score, cost),
        "cost_per_mtok": cost,
        "price_in": _first_not_none(model_source.get("price_in"), provider_cfg.get("price_in")),
        "price_out": _first_not_none(model_source.get("price_out"), provider_cfg.get("price_out")),
        "input_cost_per_mtok": _first_not_none(
            model_source.get("input_cost_per_mtok"),
            provider_cfg.get("input_cost_per_mtok"),
        ),
        "output_cost_per_mtok": _first_not_none(
            model_source.get("output_cost_per_mtok"),
            provider_cfg.get("output_cost_per_mtok"),
        ),
        "quality_score": quality_score,
        "arena_rank": model_source.get("arena_rank") or provider_cfg.get("arena_rank"),
        "graeae_weight": model_source.get("graeae_weight") or provider_cfg.get("graeae_weight"),
        "release_date": model_source.get("release_date") or provider_cfg.get("release_date"),
        "last_synced": model_source.get("last_synced") or provider_cfg.get("last_synced"),
        "context_window": model_source.get("context_window") or provider_cfg.get("context_window"),
        "model_max_ctx": model_source.get("model_max_ctx")
        or model_source.get("context_window")
        or provider_cfg.get("model_max_ctx")
        or provider_cfg.get("context_window"),
        "max_output_tokens": model_source.get("max_output_tokens") or provider_cfg.get("max_output_tokens"),
        "p50_latency_ms": p50_latency_ms,
        "available": bool(model_source.get("available", provider_cfg.get("available", True))),
        "deprecated": bool(model_source.get("deprecated", provider_cfg.get("deprecated", False))),
        "health": health,
    }


def _coerce_cached_model(item: dict[str, Any]) -> dict[str, Any] | None:
    raw_model_id = str(item.get("id") or item.get("model_id") or "").strip()
    if not raw_model_id:
        return None
    fleet = _fleet_model(raw_model_id)
    model_id = str((fleet or {}).get("id") or raw_model_id).strip()
    wire_model_id = str((fleet or {}).get("model_id") or item.get("model_id") or model_id).strip()
    provider = str(
        (_fleet_provider(fleet) if fleet else None)
        or item.get("provider")
        or item.get("registry_provider")
        or item.get("owned_by")
        or ""
    ).strip()
    if not provider:
        return None
    cfg = {
        "model": wire_model_id,
        "api": item.get("api") or "openai",
        "weight": item.get("quality_score") or item.get("graeae_weight") or item.get("weight") or 0.0,
        "price_in": item.get("price_in"),
        "price_out": item.get("price_out"),
        "input_cost_per_mtok": item.get("input_cost_per_mtok"),
        "output_cost_per_mtok": item.get("output_cost_per_mtok"),
        "context_window": item.get("context_window"),
        "model_max_ctx": item.get("model_max_ctx"),
        "max_output_tokens": item.get("max_output_tokens"),
    }
    health = item.get("health") if isinstance(item.get("health"), dict) else {"state": "cached"}
    normalized = _normalize_model(
        provider=provider,
        provider_cfg=cfg,
        model_source={**item, "id": model_id, "model_id": wire_model_id},
        health=health,
    )
    for key in _CACHE_OVERLAY_KEYS:
        if key in item:
            normalized[key] = item[key]
    normalized["provider"] = provider
    normalized["registry_provider"] = _registry_provider(provider)
    normalized["owned_by"] = str(item.get("owned_by") or provider)
    normalized["health"] = health
    return normalized


def _overlay_cached_model(base: dict[str, Any], cached: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key in _CACHE_OVERLAY_KEYS:
        if key in cached and cached[key] is not None:
            out[key] = cached[key]
    cached_caps = cached.get("capabilities")
    if isinstance(cached_caps, (list, tuple, set)):
        out["capabilities"] = sorted(
            {
                str(cap).strip()
                for cap in [*(base.get("capabilities") or []), *cached_caps]
                if str(cap).strip()
            }
        )
    out["id"] = base["id"]
    out["provider"] = base["provider"]
    out["registry_provider"] = base["registry_provider"]
    out["object"] = "model"
    out["owned_by"] = str(base.get("owned_by") or base["provider"])
    out["health"] = base.get("health") or cached.get("health") or {}
    return out


def _fill_missing_model_defaults(base: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key in _CACHE_OVERLAY_KEYS:
        if out.get(key) is None and defaults.get(key) is not None:
            out[key] = defaults[key]
    default_caps = defaults.get("capabilities")
    if isinstance(default_caps, (list, tuple, set)):
        out["capabilities"] = sorted(
            {
                str(cap).strip()
                for cap in [*(base.get("capabilities") or []), *default_caps]
                if str(cap).strip()
            }
        )
    return out


def _warn_registry_overlay_unavailable() -> None:
    """Log once when the model_registry overlay cannot be reached.

    Every shipped persistence backend -- Postgres, SQLite, MySQL, MariaDB,
    Oracle and Db2 -- implements ``ConsultationAuditRepository`` and
    guarantees ``provider``/``model_id``/``display_name`` from
    ``fetch_available_models()``; Oracle's implementation currently returns
    richer fields on top of that (family, pricing, capabilities, arena
    rank/score, context window). Whatever the active backend doesn't supply
    directly (pricing, capabilities, context window, arena rank on the other
    five) can still be filled in by the synced pricing cache
    (``mnemos.domain.pantheon.pricing``) when it has run.

    This warning fires only when even that guaranteed baseline could not be
    read -- no persistence backend AND no legacy pool configured, or the
    configured backend's ``consultations_audit`` call itself failed/is
    missing. ``_registry_rows()`` still returns an empty list in that case
    (never raises), so ``/v1/models`` keeps answering from the GRAEAE engine
    and JSON/synced-cache paths; models that exist *only* in
    ``model_registry`` are what's actually lost. Emitted once per process so
    the degradation is diagnosable without logging on every catalog read.
    """
    global _registry_overlay_warned
    if _registry_overlay_warned:
        return
    _registry_overlay_warned = True

    backend = getattr(_lc, "_persistence_backend", None)
    logger.warning(
        "[PANTHEON] model_registry overlay unavailable on %s: catalog is "
        "served without model_registry-sourced data for this read. Every "
        "persistence backend guarantees provider/model_id/display_name via "
        "ConsultationAuditRepository.fetch_available_models(); Oracle "
        "currently returns richer fields (pricing, capabilities, arena "
        "rank/score, context window). Missing fields on other backends may "
        "still be backfilled by the synced pricing cache.",
        type(backend).__name__ if backend is not None else "no configured backend",
    )


async def _registry_rows_via_backend(backend: Any) -> list[Any] | None:
    """Fetch model_registry rows through the backend-neutral audit repository.

    Returns ``None`` -- not ``[]`` -- when the backend cannot serve the
    overlay at all (no ``consultations_audit`` repository, no
    ``transactional()``, or the call raised), so the caller can distinguish
    "backend incapable / failing" from "backend queried successfully and the
    registry is genuinely empty".
    """
    consultations_audit = getattr(backend, "consultations_audit", None)
    transactional = getattr(backend, "transactional", None)
    if consultations_audit is None or not callable(transactional):
        logger.debug(
            "[PANTHEON] persistence backend %s exposes no consultations_audit capability",
            type(backend).__name__,
        )
        return None
    try:
        async with transactional() as tx:
            rows = await consultations_audit.fetch_available_models(tx)
        return list(rows) if rows is not None else []
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "[PANTHEON] model_registry overlay unavailable via %s.consultations_audit: %s",
            type(backend).__name__,
            exc,
        )
        return None


async def _registry_rows_via_pool(pool: Any) -> list[Any] | None:
    """Fetch the richer legacy projection from an asyncpg-compatible pool."""
    queries = [
        """
        SELECT provider, model_id, display_name, capabilities,
               price_in, price_out, input_cost_per_mtok, output_cost_per_mtok,
               context_window, model_max_ctx, max_output_tokens,
               arena_rank, COALESCE(graeae_weight, 0) AS graeae_weight,
               release_date, last_synced, available, deprecated
        FROM model_registry
        WHERE available = true
        ORDER BY provider, model_id
        """,
        """
        SELECT provider, model_id, display_name, capabilities,
               input_cost_per_mtok, output_cost_per_mtok,
               context_window, max_output_tokens, arena_rank,
               COALESCE(graeae_weight, 0) AS graeae_weight,
               last_synced, available, deprecated
        FROM model_registry
        WHERE available = true
        ORDER BY provider, model_id
        """,
    ]
    try:
        async with pool.acquire() as conn:
            for query in queries:
                try:
                    return list(await conn.fetch(query))
                except Exception as exc:  # noqa: BLE001
                    logger.debug("[PANTHEON] model_registry catalog overlay unavailable: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[PANTHEON] model_registry catalog overlay unavailable: %s", exc)
    return None


async def _registry_rows() -> list[Any]:
    """Return model_registry rows for the catalog overlay.

    Backend-neutral path first: every backend (Postgres, SQLite, MySQL,
    MariaDB, Oracle, Db2) implements
    ``ConsultationAuditRepository.fetch_available_models`` inside a
    ``backend.transactional()`` scope, so this no longer depends on
    ``mnemos.core.lifecycle._pool`` (Postgres-only) to see registry data.

    When the configured backend also exposes ``_lc._pool`` (the Postgres
    profile), the legacy query enriches the neutral three-column result so
    existing Postgres metadata is preserved. A pool failure falls back to the
    neutral rows. Non-Postgres profiles set ``_lc._pool`` to ``None``, so they
    never execute Postgres SQL. The legacy queries also remain compatible with
    callers that set a pool directly without configuring a backend.
    """
    backend = getattr(_lc, "_persistence_backend", None)
    if backend is not None:
        rows = await _registry_rows_via_backend(backend)
        if rows is None:
            _warn_registry_overlay_unavailable()
            return []
        pool = _lc._pool
        if pool is None:
            return rows
        rich_rows = await _registry_rows_via_pool(pool)
        return rich_rows or rows

    pool = _lc._pool
    if pool is None:
        _warn_registry_overlay_unavailable()
        return []
    rows = await _registry_rows_via_pool(pool)
    if rows is None:
        _warn_registry_overlay_unavailable()
        return []
    return rows


def _sort_models(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        models,
        key=lambda item: (
            not item.get("available", True),
            bool(item.get("deprecated", False)),
            item.get("cost_per_mtok") is None,
            item.get("cost_per_mtok") if item.get("cost_per_mtok") is not None else float("inf"),
            -float(item.get("quality_score") or 0.0),
            str(item.get("id") or ""),
        ),
    )


def _synced_cache_models() -> list[dict[str, Any]] | None:
    try:
        from mnemos.domain.pantheon.pricing import read_json_cache

        payload = read_json_cache()
    except Exception as exc:  # noqa: BLE001
        logger.debug("[PANTHEON] synced catalog cache unavailable: %s", exc)
        return None
    if not isinstance(payload, dict) or payload.get("schema") != CATALOG_CACHE_SCHEMA:
        return None
    models = payload.get("models")
    if not isinstance(models, list) or not models:
        return None
    cached = [
        normalized
        for model in models
        if isinstance(model, dict)
        for normalized in [_coerce_cached_model(dict(model))]
        if normalized is not None
    ]
    return cached or None


def _catalog_cache_paths() -> list[Path]:
    paths: list[str] = []
    pricing_cache_path = os.environ.get("PANTHEON_CATALOG_CACHE")
    if pricing_cache_path:
        paths.append(pricing_cache_path)
    try:
        configured = get_settings().pantheon.catalog_cache_path
        if configured:
            paths.append(str(configured))
    except Exception as exc:  # noqa: BLE001
        logger.debug("[PANTHEON] configured catalog cache path unavailable: %s", exc)
    env_path = os.environ.get("MNEMOS_PANTHEON_CATALOG_CACHE_PATH")
    if env_path:
        paths.append(env_path)
    paths.extend(_DEFAULT_CACHE_PATHS)
    cwd = Path.cwd()
    out: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw).expanduser()
        path = path if path.is_absolute() else cwd / path
        if path not in seen:
            out.append(path)
            seen.add(path)
    return out


def _cache_payload_models(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        raw = payload.get("data") or payload.get("models") or payload.get("catalog") or []
    else:
        raw = payload
    if not isinstance(raw, list):
        return []
    models: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        normalized = _coerce_cached_model(dict(item))
        if normalized is not None:
            models.append(normalized)
    return models


def _cached_catalog_models() -> list[dict[str, Any]]:
    for path in _catalog_cache_paths():
        try:
            if not path.exists():
                continue
            models = _cache_payload_models(json.loads(path.read_text()))
            if models:
                logger.info("[PANTHEON] loaded cached phase-A catalog from %s", path)
                return models
        except Exception as exc:  # noqa: BLE001
            logger.debug("[PANTHEON] cached catalog %s unavailable: %s", path, exc)
    return []


def _model_sources(provider_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    raw_models = provider_cfg.get("models")
    if isinstance(raw_models, list) and raw_models:
        out: list[dict[str, Any]] = []
        for item in raw_models:
            if isinstance(item, dict):
                out.append(dict(item))
            elif item:
                out.append({"model_id": str(item)})
        return out
    return [{"model_id": provider_cfg.get("model")}]


def _pantheon_provider_defaults() -> dict[str, dict[str, Any]]:
    try:
        from mnemos.domain.pantheon.gateway import _PANTHEON_PROVIDER_DEFAULTS

        return {provider: dict(cfg) for provider, cfg in _PANTHEON_PROVIDER_DEFAULTS.items()}
    except Exception as exc:  # noqa: BLE001
        logger.debug("[PANTHEON] gateway provider defaults unavailable for catalog: %s", exc)
        return {}


def _catalog_provider_cfgs(engine_providers: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    cfgs = _pantheon_provider_defaults()
    for provider, cfg in engine_providers.items():
        cfgs[provider] = {**cfgs.get(provider, {}), **dict(cfg)}
    return cfgs


def _cache_indexes(
    cached_models: list[dict[str, Any]] | None,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    by_id: dict[str, list[dict[str, Any]]] = {}
    for model in cached_models or []:
        model_id = str(model.get("id") or model.get("model_id") or "").strip()
        if not model_id:
            continue
        lower_id = model_id.lower()
        by_id.setdefault(lower_id, []).append(model)
        for provider in _provider_aliases(model):
            by_key[(provider, lower_id)] = model
    return by_key, by_id


def _cached_overlay_for(
    model: dict[str, Any],
    by_key: dict[tuple[str, str], dict[str, Any]],
    by_id: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    model_id = str(model.get("id") or model.get("model_id") or "").strip()
    if not model_id:
        return None
    lower_id = model_id.lower()
    for provider in _provider_aliases(model):
        cached = by_key.get((provider, lower_id))
        if cached is not None:
            return cached
    id_matches = by_id.get(lower_id) or []
    if len(id_matches) == 1 or _fleet_model(model_id):
        return id_matches[0] if id_matches else None
    return None


def _fleet_model_sources(provider_cfgs: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for source in _PANTHEON_FLEET_MODELS:
        provider = _fleet_provider(source)
        if not provider or provider not in provider_cfgs:
            continue
        provider_cfg = dict(provider_cfgs[provider])
        provider_cfg["model"] = source["model_id"]
        provider_cfg.setdefault("api", "openai")
        provider_cfg.setdefault("key_name", provider)
        provider_cfg.setdefault("weight", source.get("quality_score") or 0.0)
        out.append((provider, provider_cfg, dict(source)))
    return out


async def list_models(*, use_synced_cache: bool = True) -> list[dict[str, Any]]:
    """Return the extended PANTHEON model catalog."""
    synced_cached_models = _synced_cache_models() if use_synced_cache else None
    legacy_cached_models = _cached_catalog_models() if use_synced_cache and synced_cached_models is None else []
    cached_by_key, cached_by_id = _cache_indexes(synced_cached_models or legacy_cached_models)

    engine = get_graeae_engine()
    try:
        provider_status = engine.provider_status()
    except Exception:  # noqa: BLE001
        provider_status = {}

    models: dict[tuple[str, str], dict[str, Any]] = {}

    def add_model(normalized: dict[str, Any], *, overwrite: bool = True) -> None:
        cached = _cached_overlay_for(normalized, cached_by_key, cached_by_id)
        if cached is not None:
            normalized = _overlay_cached_model(normalized, cached)
        key = _catalog_key(normalized)
        if key is not None and (overwrite or key not in models):
            models[key] = normalized

    engine_provider_cfgs = {name: dict(cfg) for name, cfg in engine.providers.items()}
    provider_cfgs = _catalog_provider_cfgs(engine_provider_cfgs)
    for provider, cfg in provider_cfgs.items():
        provider_cfg = dict(cfg)
        if provider_cfg.get("enabled") is False:
            continue
        health = _provider_health(provider, provider_status)
        for model_source in _model_sources(provider_cfg):
            if not model_source.get("model_id") and not model_source.get("id"):
                continue
            normalized = _normalize_model(
                provider=provider,
                provider_cfg=provider_cfg,
                model_source=model_source,
                health=health,
            )
            add_model(normalized)

    registry_to_graeae = {cfg["registry_provider"]: name for name, cfg in GRAEAE_REGISTRY_MAP.items()}
    for row in await _registry_rows():
        registry_provider = str(_row_get(row, "provider") or "")
        raw_model_id = _row_get(row, "model_id")
        provider = registry_to_graeae.get(registry_provider, registry_provider)
        fleet = _fleet_model(raw_model_id)
        if fleet:
            provider = _fleet_provider(fleet)
            if not _fleet_provider_registered(raw_model_id, provider_cfgs):
                continue
        model_id = str((fleet or {}).get("id") or raw_model_id)
        wire_model_id = str((fleet or {}).get("model_id") or raw_model_id)
        provider_cfg = provider_cfgs.get(provider, {"model": wire_model_id})
        model_source = {
            "id": model_id,
            "model_id": wire_model_id,
            "display_name": _row_get(row, "display_name"),
            "capabilities": _row_get(row, "capabilities") or [],
            "price_in": _row_get(row, "price_in"),
            "price_out": _row_get(row, "price_out"),
            "input_cost_per_mtok": _row_get(row, "input_cost_per_mtok"),
            "output_cost_per_mtok": _row_get(row, "output_cost_per_mtok"),
            "context_window": _row_get(row, "context_window"),
            "model_max_ctx": _row_get(row, "model_max_ctx"),
            "max_output_tokens": _row_get(row, "max_output_tokens"),
            "arena_rank": _row_get(row, "arena_rank"),
            "graeae_weight": _row_get(row, "graeae_weight"),
            "release_date": _row_get(row, "release_date"),
            "last_synced": _row_get(row, "last_synced"),
            "available": _row_get(row, "available", True),
            "deprecated": _row_get(row, "deprecated", False),
        }
        health = _provider_health(provider, provider_status)
        normalized = _normalize_model(
            provider=provider,
            provider_cfg=provider_cfg,
            model_source=model_source,
            health=health,
        )
        add_model(normalized)

    for provider, provider_cfg, model_source in _fleet_model_sources(provider_cfgs):
        health = _provider_health(provider, provider_status)
        normalized = _normalize_model(
            provider=provider,
            provider_cfg=provider_cfg,
            model_source=model_source,
            health=health,
        )
        key = _catalog_key(normalized)
        if key is not None and key in models:
            models[key] = _fill_missing_model_defaults(models[key], normalized)
        else:
            add_model(normalized, overwrite=False)

    for cached_model in synced_cached_models or legacy_cached_models:
        if not _fleet_provider_registered(cached_model.get("id") or cached_model.get("model_id"), provider_cfgs):
            continue
        key = _catalog_key(cached_model)
        if key is not None and key not in models:
            models[key] = cached_model

    return _sort_models(list(models.values()))


async def models_response(
    *,
    filter_capabilities: list[str] | None = None,
    filter_tier: str | None = None,
    max_cost: float | None = None,
) -> dict[str, Any]:
    models = filter_models(
        await list_models(),
        filter_capabilities=filter_capabilities,
        filter_tier=filter_tier,
        max_cost=max_cost,
    )
    return {"object": "list", "data": models}


def filter_models(
    models: list[dict[str, Any]],
    *,
    filter_capabilities: list[str] | None = None,
    filter_tier: str | None = None,
    max_cost: float | None = None,
) -> list[dict[str, Any]]:
    required = {cap.strip() for cap in (filter_capabilities or []) if cap and cap.strip()}
    tier = filter_tier.strip() if isinstance(filter_tier, str) and filter_tier.strip() else None
    out: list[dict[str, Any]] = []
    for model in models:
        if required and not required.issubset(set(model.get("capabilities") or [])):
            continue
        if tier and model.get("usage_tier") != tier:
            continue
        cost = model.get("cost_per_mtok")
        if max_cost is not None and (cost is None or float(cost) > max_cost):
            continue
        out.append(model)
    return out


def find_model(models: list[dict[str, Any]], model_id: str) -> dict[str, Any] | None:
    for model in models:
        if model["id"] == model_id:
            return model
        namespaced = f"{model['provider']}/{model['id']}"
        registry_namespaced = f"{model['registry_provider']}/{model['id']}"
        if model_id in {namespaced, registry_namespaced}:
            return model
    return None
