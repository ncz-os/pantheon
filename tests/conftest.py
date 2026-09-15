from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


# PANTHEON telemetry-loss-reporting fix: the sibling KNEMON checkout is not
# present in this multi-repo workspace, so the ``mnemos.domain.knemon.budget``
# module that ``mnemos.domain.pantheon.budget`` imports at module-load time
# would fail with ``ModuleNotFoundError``. Install the runtime stub
# FIRST — before any test module (including this conftest's pytest
# sessionstart hook) triggers a transitive import. The stub installs
# an unlimited budget gate by default so non-budget tests don't have to
# think about it; budget-gating tests monkey-patch
# ``mnemos.domain.knemon.budget._rows`` to drive specific verdicts.
_STUB_PATH = Path(__file__).resolve().parent / "_mnemos_knemon_stubs.py"
import importlib.util as _importlib_util  # noqa: E402

_spec = _importlib_util.spec_from_file_location("_mnemos_knemon_runtime", _STUB_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"could not load knemon runtime stub from {_STUB_PATH!s}")
_module = _importlib_util.module_from_spec(_spec)
sys.modules.setdefault("_mnemos_knemon_runtime", _module)
_spec.loader.exec_module(_module)
_module.install()


def pytest_sessionstart(session) -> None:
    """Wire optional routers once with PANTHEON enabled before test-local toggles."""
    previous = os.environ.get("MNEMOS_PANTHEON_ENABLED")
    os.environ["MNEMOS_PANTHEON_ENABLED"] = "true"
    try:
        # PANTHEON telemetry-loss-reporting: ensure the sibling-repo source
        # roots are on sys.path AND that the ``mnemos`` namespace package
        # aggregates those roots. pytest's ``pythonpath`` ini option prepends
        # entries to sys.path, but the editable finder initialises the
        # namespace packages with empty ``__path__`` lists — so the new paths
        # never get added and every ``from mnemos.X import Y`` under
        # ``uv run pytest`` resolves with a bare namespace package whose
        # ``__path__`` is ``[]``.
        #
        # We must NOT pre-create child namespace packages in ``sys.modules``
        # — that would shadow the real regular packages (which have
        # ``__init__.py``) at the ``mnemos.audit`` / ``mnemos.api`` / … level
        # and break the regular import resolution. Instead we only extend
        # ``__path__`` on the parent namespace packages that are ALREADY in
        # ``sys.modules``. Children are re-imported on demand; their
        # ``__init__.py`` files are then loaded normally and any subsequent
        # ``mnemos.audit.submodule`` import resolves through both PEP 420
        # namespace discovery AND the regular package importer.
        _LOCAL_MNEMOS = os.path.join(os.getcwd(), "mnemos")
        # Version-pin cleanup (2026-09-15): this used to hardcode two
        # absolute, machine-specific paths
        # (/Users/jasonperlow/mnemos-core-release,
        # /Users/jasonperlow/graeae-quota-fix) — scratch checkouts from one
        # operator's local session, meaningless on any other machine.
        #
        # This repo's own README documents no CI job actually runs this
        # test suite (.gitlab-ci.yml only runs ruff) — `uv run pytest` here
        # is exclusively a LOCAL/dev-loop tool, and this sys.path surgery
        # is the only thing that makes it resolve `mnemos.core` /
        # `mnemos.api` at all (confirmed: removing sibling roots entirely
        # breaks collection in 31 test files, not just a handful). So the
        # right fix is not "remove it" but "point it at the convention
        # this repo family actually documents" — graeae/README.md's own
        # setup instructions clone the real mnemos-core repo as `../mnemos-
        # core` next to each split-package checkout:
        #   git clone https://gitlab.com/ncz-os/mnemos.git ../mnemos-core
        #   git clone https://gitlab.com/ncz-os/graeae.git ../graeae
        # Default to that relative, documented layout (works for any
        # operator who set their workspace up as instructed, portable
        # across machines since it's relative to this repo, not absolute);
        # override via MNEMOS_SIBLING_ROOTS (a `:`-separated path list) if
        # your layout differs.
        _SIBLING_ROOTS = tuple(
            p
            for p in os.environ.get(
                "MNEMOS_SIBLING_ROOTS", "../mnemos-core:../graeae"
            ).split(":")
            if p
        )
        _mnemos_dirs: list[str] = []
        for _r in _SIBLING_ROOTS:
            if os.path.isdir(os.path.join(_r, "mnemos")):
                _mnemos_dirs.append(os.path.join(_r, "mnemos"))
        if os.path.isdir(_LOCAL_MNEMOS):
            _mnemos_dirs.append(_LOCAL_MNEMOS)

        # Only the top-level ``mnemos`` package is guaranteed to be in
        # ``sys.modules`` at sessionstart (the editable finder initialises
        # it eagerly). Make sure its ``__path__`` aggregates every root.
        _mnemos_top = sys.modules.get("mnemos")
        if _mnemos_top is not None:
            existing = list(getattr(_mnemos_top, "__path__", []) or [])
            for _p in _mnemos_dirs:
                if _p not in existing:
                    existing.append(_p)
            _mnemos_top.__path__ = existing  # type: ignore[attr-defined]

        # Force-import every namespace package ONE LEVEL DEEP so the
        # editable finder registers them and the regular package importer
        # can locate ``__init__.py`` underneath. We do NOT cache the
        # resulting module objects in ``sys.modules`` via our own
        # ModuleType — we only nudge the existing namespace packages'
        # ``__path__`` lists so subsequent ``__import__`` calls find the
        # sibling roots.
        for _sub in ("api", "domain", "core", "audit", "cli", "db",
                     "persistence", "workers"):
            modname = f"mnemos.{_sub}"
            try:
                __import__(modname)
            except Exception:
                # Some sub-namespaces don't exist in every sibling checkout;
                # that's fine — only the ones that do will get their
                # ``__path__`` extended via the next loop.
                pass
            mod = sys.modules.get(modname)
            if mod is None:
                continue
            existing = list(getattr(mod, "__path__", []) or [])
            # For each known sibling root, add the matching subdir if it
            # exists. This is the only place we mutate ``__path__`` for
            # child namespaces — and only by extending, never by replacing.
            for _root in _SIBLING_ROOTS:
                candidate = os.path.join(_root, "mnemos", _sub)
                if os.path.isdir(candidate) and candidate not in existing:
                    existing.append(candidate)
            local_candidate = os.path.join(_LOCAL_MNEMOS, _sub)
            if os.path.isdir(local_candidate) and local_candidate not in existing:
                existing.append(local_candidate)
            mod.__path__ = existing  # type: ignore[attr-defined]

        # The legacy preflight imported ``mnemos.api.main`` to warm the
        # FastAPI routers ahead of test-local toggles. Under ``uv run
        # pytest`` the heavy core API pulls in a long transitive import
        # chain (``mnemos.persistence``, ``mnemos.audit``, ``mnemos.db``,
        # …) that is orthogonal to the telemetry-degradation regression
        # under test. Treat any failure here as a soft warning rather
        # than an INTERNALERROR — the rest of the conftest fixtures
        # stand alone, and the test of interest only needs
        # ``mnemos.core.config``.
        try:
            from mnemos.api.main import app  # noqa: F401
        except Exception as _preflight_exc:  # noqa: BLE001
            import warnings as _warnings  # noqa: E402

            _warnings.warn(
                f"preflight mnemos.api.main import failed (degraded pre-cache): "
                f"{_preflight_exc!r}",
                RuntimeWarning,
                stacklevel=1,
            )
    finally:
        if previous is None:
            os.environ.pop("MNEMOS_PANTHEON_ENABLED", None)
        else:
            os.environ["MNEMOS_PANTHEON_ENABLED"] = previous
        try:
            from mnemos.core.config import _reset_settings_for_tests
            _reset_settings_for_tests()
        except Exception as _cfg_exc:  # noqa: BLE001
            import warnings as _warnings  # noqa: E402

            _warnings.warn(
                f"preflight mnemos.core.config reset failed: {_cfg_exc!r}",
                RuntimeWarning,
                stacklevel=1,
            )


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
