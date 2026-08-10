from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def pytest_sessionstart(session) -> None:
    """Wire optional routers once with PANTHEON enabled before test-local toggles."""
    previous = os.environ.get("MNEMOS_PANTHEON_ENABLED")
    os.environ["MNEMOS_PANTHEON_ENABLED"] = "true"
    try:
        from mnemos.api.main import app  # noqa: F401
    finally:
        if previous is None:
            os.environ.pop("MNEMOS_PANTHEON_ENABLED", None)
        else:
            os.environ["MNEMOS_PANTHEON_ENABLED"] = previous
        from mnemos.core.config import _reset_settings_for_tests

        _reset_settings_for_tests()


class _FakeConnection:
    def __init__(self, state: dict[str, Any]):
        self.state = state

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        compact = " ".join(query.split())
        if "FROM usage_ledger" in compact and "SUM(est_cost_usd)" in compact:
            caller = str(args[0]) if args else "pantheon"
            rows = [row for row in self.state["usage_ledger"] if row.get("caller_subsystem") == caller]
            return [{"spent_usd": sum(float(row.get("est_cost_usd") or 0) for row in rows)}]
        if "FROM pantheon_routing_audit" in compact:
            return list(self.state["pantheon_routing_audit"])
        if "FROM model_registry" in compact:
            return list(self.state["model_registry"])
        return []


class _AcquireContext:
    def __init__(self, conn: _FakeConnection):
        self.conn = conn

    async def __aenter__(self) -> _FakeConnection:
        return self.conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class FakePool:
    """Small asyncpg-shaped store used by the extracted subsystem tests."""

    def __init__(self):
        self.state: dict[str, Any] = {
            "memories": {},
            "pantheon_routing_audit": [],
            "usage_ledger": [],
            "model_registry": [],
        }
        self._conn = _FakeConnection(self.state)

    def acquire(self) -> _AcquireContext:
        return _AcquireContext(self._conn)


class FakePoolBackedBackend:
    """Backend facade over ``FakePool`` for audit and budget integration tests."""

    def __init__(self, pool: FakePool):
        self._pool = pool

    @asynccontextmanager
    async def transactional(self):
        async with self._pool.acquire() as conn:
            yield SimpleNamespace(conn=conn)

    async def insert_pantheon_routing_audit(self, _tx, record: dict[str, Any]) -> None:
        payload = record["payload_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        self._pool.state["pantheon_routing_audit"].append({**record, "payload": payload})

    async def record_usage_ledger(self, _tx, record):
        from mnemos.persistence.base import UsageLedgerResult

        cost = record.est_cost_usd
        if cost is None:
            cost = Decimal("0")
        row = {
            "caller_subsystem": record.caller_subsystem,
            "provider": record.provider,
            "est_cost_usd": cost,
        }
        self._pool.state["usage_ledger"].append(row)
        return UsageLedgerResult(id=len(self._pool.state["usage_ledger"]), est_cost_usd=cost)


@pytest.fixture
def db_pool() -> FakePool:
    return FakePool()


@pytest.fixture(autouse=True)
def _single_worker_shared_state_mode(monkeypatch):
    """The repository test process is intentionally a one-worker dev runtime."""
    monkeypatch.setenv("MNEMOS_PANTHEON_ALLOW_LOCAL_SHARED_STATE", "true")


@pytest.fixture(autouse=True)
def _isolated_catalog_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("PANTHEON_CATALOG_CACHE", str(tmp_path / "absent-catalog.json"))
    monkeypatch.delenv("MNEMOS_PANTHEON_CATALOG_CACHE_PATH", raising=False)


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    from mnemos.core.rate_limit import limiter

    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture(autouse=True)
def _editable_worker_namespace():
    """Make wheel-merged worker modules visible in this multi-repo editable checkout."""
    import mnemos.workers

    worker_path = str(Path(__file__).parents[1] / "mnemos" / "workers")
    if worker_path not in mnemos.workers.__path__:
        mnemos.workers.__path__.append(worker_path)
