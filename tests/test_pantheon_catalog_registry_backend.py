"""Regression coverage for catalog._registry_rows() going through the
backend-neutral ConsultationAuditRepository (`backend.transactional()` +
`backend.consultations_audit.fetch_available_models(tx)`) instead of raw
asyncpg SQL against `mnemos.core.lifecycle._pool`.

Covers:
  * a registry-only model (not present via GRAEAE engine/fleet defaults)
    surfacing in `list_models()` and being locatable via `find_model()`
    when the configured backend has NO asyncpg pool underneath at all;
  * graceful, non-raising behaviour with a single warning when the
    backend has no `consultations_audit` capability;
  * graceful, non-raising behaviour with a single warning when the
    backend's `consultations_audit.fetch_available_models()` call raises;
  * in both failure cases, the legacy Postgres-dialect `_lc._pool` path is
    never touched -- a non-Postgres backend must not fall back to
    Postgres SQL.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import mnemos.core.lifecycle as lc
import pytest

from mnemos.domain.pantheon import catalog


class _FakeEngine:
    """Empty GRAEAE engine -- the registry overlay is what's under test."""

    def __init__(self) -> None:
        self.providers: dict[str, Any] = {}

    def provider_status(self) -> dict[str, Any]:
        return {}


class _FakeTx:
    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None


class _FakeConsultationAudit:
    """Minimal ConsultationAuditRepository double with no pool underneath."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.calls = 0

    async def fetch_available_models(self, tx: Any) -> list[dict[str, Any]]:
        self.calls += 1
        assert isinstance(tx, _FakeTx)
        return list(self._rows)


class _FailingConsultationAudit:
    """A consultations_audit whose registry call always raises."""

    def __init__(self) -> None:
        self.calls = 0

    async def fetch_available_models(self, tx: Any) -> list[dict[str, Any]]:
        self.calls += 1
        raise RuntimeError("model_registry table unavailable on this backend")


class _FakeNonPoolBackend:
    """A persistence backend double with no asyncpg pool anywhere.

    This is the shape every non-Postgres Mnemos core backend has: no
    `_lc._pool`, only `transactional()` + repository facades. If catalog
    code ever reached for `_lc._pool` while this backend is configured, the
    pool-touch counters in these tests would catch it.
    """

    def __init__(self, consultations_audit: Any) -> None:
        self.consultations_audit = consultations_audit
        self.transactional_calls = 0

    @asynccontextmanager
    async def transactional(self, *, isolation: Any = None, readonly: bool = False):
        self.transactional_calls += 1
        yield _FakeTx()


class _NoConsultationsAuditBackend:
    """A backend that exposes transactional() but no consultations_audit."""

    consultations_audit = None

    def __init__(self) -> None:
        self.transactional_calls = 0

    @asynccontextmanager
    async def transactional(self, *, isolation: Any = None, readonly: bool = False):
        self.transactional_calls += 1
        yield _FakeTx()


class _PoolTouchSentinel:
    """Stands in for `_lc._pool`; records whether anything ever tried to use it."""

    def __init__(self) -> None:
        self.acquire_calls = 0

    def acquire(self):
        self.acquire_calls += 1
        raise AssertionError(
            "legacy Postgres pool fallback must not be used once a "
            "persistence backend is configured"
        )


class _FakePool:
    def __init__(self, rows: list[dict[str, Any]] | None = None, *, fail: bool = False) -> None:
        self.rows = rows or []
        self.fail = fail
        self.acquire_calls = 0

    def acquire(self):
        self.acquire_calls += 1
        pool = self

        class _Acquire:
            async def __aenter__(self):
                if pool.fail:
                    raise RuntimeError("asyncpg pool unavailable")
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def fetch(self, query: str):
                return list(pool.rows)

        return _Acquire()


@pytest.fixture(autouse=True)
def _reset_catalog_state(monkeypatch):
    """Isolate the catalog module's process-global caches/flags per test."""
    monkeypatch.setattr(catalog, "_registry_overlay_warned", False)
    monkeypatch.setattr(catalog, "_PANTHEON_FLEET_MODELS", ())
    monkeypatch.setattr(catalog, "_FLEET_MODELS_BY_ID", {})
    monkeypatch.setattr(catalog, "get_graeae_engine", lambda: _FakeEngine())
    monkeypatch.setattr(catalog, "_synced_cache_models", lambda: None)
    monkeypatch.setattr(catalog, "_cached_catalog_models", list)
    catalog.invalidate_catalog_snapshot()
    monkeypatch.setattr(lc, "_pool", None)
    monkeypatch.setattr(lc, "_persistence_backend", None)
    yield


@pytest.mark.asyncio
async def test_registry_only_model_surfaces_via_non_pool_backend(monkeypatch):
    """A model that exists ONLY in model_registry must appear via the ABC path."""
    rows = [
        {
            "provider": "groq",
            "model_id": "registry-only-model",
            "display_name": "Registry Only Model",
        }
    ]
    audit = _FakeConsultationAudit(rows)
    backend = _FakeNonPoolBackend(audit)
    monkeypatch.setattr(lc, "_persistence_backend", backend)
    monkeypatch.setattr(lc, "_pool", None)

    models = await catalog.list_models(use_synced_cache=False)

    assert backend.transactional_calls == 1
    assert audit.calls == 1

    found = catalog.find_model(models, "registry-only-model")
    assert found is not None
    assert found["provider"] == "groq"
    assert found["display_name"] == "Registry Only Model"
    assert found["model_id"] == "registry-only-model"


@pytest.mark.asyncio
async def test_registry_rows_returns_guaranteed_fields_directly():
    """Unit-level: `_registry_rows()` returns exactly what the ABC hands back."""
    rows = [
        {"provider": "anthropic", "model_id": "claude-x", "display_name": "Claude X"},
    ]
    audit = _FakeConsultationAudit(rows)
    backend = _FakeNonPoolBackend(audit)
    lc._persistence_backend = backend
    try:
        result = await catalog._registry_rows()
    finally:
        lc._persistence_backend = None

    assert result == rows
    assert backend.transactional_calls == 1


@pytest.mark.asyncio
async def test_postgres_pool_preserves_rich_registry_projection(monkeypatch):
    baseline = [{"provider": "openai", "model_id": "gpt-x", "display_name": "GPT X"}]
    rich = [{**baseline[0], "capabilities": ["reasoning"], "context_window": 200_000}]
    backend = _FakeNonPoolBackend(_FakeConsultationAudit(baseline))
    pool = _FakePool(rich)
    monkeypatch.setattr(lc, "_persistence_backend", backend)
    monkeypatch.setattr(lc, "_pool", pool)

    result = await catalog._registry_rows()

    assert result == rich
    assert pool.acquire_calls == 1


@pytest.mark.asyncio
async def test_postgres_pool_failure_keeps_backend_neutral_rows(monkeypatch):
    baseline = [{"provider": "openai", "model_id": "gpt-x", "display_name": "GPT X"}]
    backend = _FakeNonPoolBackend(_FakeConsultationAudit(baseline))
    pool = _FakePool(fail=True)
    monkeypatch.setattr(lc, "_persistence_backend", backend)
    monkeypatch.setattr(lc, "_pool", pool)

    result = await catalog._registry_rows()

    assert result == baseline
    assert pool.acquire_calls == 1
    assert catalog._registry_overlay_warned is False


@pytest.mark.asyncio
async def test_missing_consultations_audit_capability_warns_once_and_returns_empty(
    monkeypatch, caplog
):
    backend = _NoConsultationsAuditBackend()
    pool_sentinel = _PoolTouchSentinel()
    monkeypatch.setattr(lc, "_persistence_backend", backend)
    monkeypatch.setattr(lc, "_pool", pool_sentinel)

    with caplog.at_level(logging.DEBUG, logger="mnemos.domain.pantheon.catalog"):
        first = await catalog._registry_rows()
        second = await catalog._registry_rows()

    assert first == []
    assert second == []
    assert pool_sentinel.acquire_calls == 0
    # transactional() is never opened -- there's nothing to run a query in.
    assert backend.transactional_calls == 0

    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("consultations_audit" in r.getMessage() for r in debug_records)
    # Warned exactly once across both calls -- not once per catalog read.
    assert len(warning_records) == 1
    assert "model_registry overlay unavailable" in warning_records[0].getMessage()


@pytest.mark.asyncio
async def test_failing_consultations_audit_call_warns_once_and_returns_empty(
    monkeypatch, caplog
):
    audit = _FailingConsultationAudit()
    backend = _FakeNonPoolBackend(audit)
    pool_sentinel = _PoolTouchSentinel()
    monkeypatch.setattr(lc, "_persistence_backend", backend)
    monkeypatch.setattr(lc, "_pool", pool_sentinel)

    with caplog.at_level(logging.DEBUG, logger="mnemos.domain.pantheon.catalog"):
        first = await catalog._registry_rows()
        second = await catalog._registry_rows()

    assert first == []
    assert second == []
    assert audit.calls == 2
    assert backend.transactional_calls == 2
    assert pool_sentinel.acquire_calls == 0

    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("model_registry table unavailable" in r.getMessage() for r in debug_records)
    assert len(warning_records) == 1


@pytest.mark.asyncio
async def test_no_backend_configured_falls_back_to_legacy_pool(monkeypatch):
    """Compatibility fallback: only used when NO backend is configured at all."""

    class _FakeConn:
        def __init__(self, rows: list[dict[str, Any]]):
            self.rows = rows
            self.queries: list[str] = []

        async def fetch(self, query: str, *args: Any):
            self.queries.append(" ".join(query.split()))
            return list(self.rows)

    class _FakeAcquire:
        def __init__(self, conn: _FakeConn):
            self.conn = conn

        async def __aenter__(self):
            return self.conn

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _FakePool:
        def __init__(self, rows: list[dict[str, Any]]):
            self.conn = _FakeConn(rows)

        def acquire(self):
            return _FakeAcquire(self.conn)

    rows = [{"provider": "openai", "model_id": "legacy-pool-model", "display_name": "Legacy"}]
    monkeypatch.setattr(lc, "_persistence_backend", None)
    monkeypatch.setattr(lc, "_pool", _FakePool(rows))

    result = await catalog._registry_rows()

    assert result == rows
