"""Runtime stub for ``mnemos.domain.knemon.budget``.

The PANTHEON telemetry-loss-reporting fix needs to run its validation
suite in this multi-repo checkout without the sibling KNEMON
distribution present. ``mnemos.domain.knemon`` is therefore stubbed
in-place at collection time: a small ``mnemos.domain`` namespace
package is registered in ``sys.modules`` alongside the empty stub
under ``mnemos-stubs/mnemos_knemon/`` (which exists only to satisfy
uv's resolver). The runtime surface mirrors the contract PANTHEON's
F05 tests monkey-patch:

* ``_rows(backend, sql, params=None)`` — the SQL-row fetcher. Tests
  replace this with their own coroutine to control ledger state.
* ``evaluate_usage_budget(backend, *, estimated_cost_usd, limit_usd,
  caller_subsystem, provider)`` — wraps ``_rows`` into a
  ``BudgetDecision``.
* ``BudgetDecision`` / ``BudgetVerdict`` — dataclass + enum used by
  the route layer's denial path.

The stub deliberately ships an UNLIMITED default (``limit_usd=None``
returns ALLOW with ``remaining_usd=inf``) so any test that forgets
to monkey-patch ``_rows`` falls open rather than denying real
requests — matching the production "unconfigured cap" semantics.
"""

from __future__ import annotations

import enum
import sys
import types
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable

_RowsFn = Callable[..., Awaitable[list[dict[str, Any]]]]


async def _default_rows(backend: Any, _sql: str, _params: Any = None) -> list[dict[str, Any]]:
    """Default ledger row fetcher — falls back to inspecting the backend
    for a FakePool-shaped ``state["usage_ledger"]`` payload.

    Tests that exercise the budget gate either:

    * monkey-patch ``_rows`` via
      ``monkeypatch.setattr("mnemos.domain.knemon.budget._rows", ...)``, or
    * hand the function a backend that exposes ``state["usage_ledger"]``
      either directly (FakePool) or via ``backend._pool.state`` /
      ``backend.pool.state`` (FakePoolBackedBackend and similar wrappers
      used by ``tests/test_pantheon_v0_2.py``'s ``pantheon_v02_client``
      fixture). The row aggregator below sums ``est_cost_usd`` across all
      rows so it matches the historical ``SUM(est_cost_usd)`` SQL.
    """
    rows: list[dict[str, Any]] = []
    state = _resolve_state(backend)
    if state is not None:
        ledger = state.get("usage_ledger")
        if isinstance(ledger, list):
            spent = 0.0
            for row in ledger:
                if not isinstance(row, dict):
                    continue
                try:
                    spent += float(row.get("est_cost_usd") or 0.0)
                except (TypeError, ValueError):
                    continue
            rows.append({"spent_usd": spent})
    return rows


def _resolve_state(backend: Any) -> dict[str, Any] | None:
    """Return the ``{"usage_ledger": [...], ...}`` state dict for a backend,
    unwrapping the FakePoolBackedBackend facade if needed."""
    if backend is None:
        return None
    state = getattr(backend, "state", None)
    if isinstance(state, dict):
        return state
    for attr in ("_pool", "pool"):
        inner = getattr(backend, attr, None)
        if inner is None:
            continue
        state = getattr(inner, "state", None)
        if isinstance(state, dict):
            return state
    return None


_rows: _RowsFn = _default_rows


class BudgetVerdict(enum.Enum):
    ALLOW = "allow"
    DENY = "deny"
    CAP_UNCONFIGURED = "cap_unconfigured"


@dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    verdict: BudgetVerdict
    reason: str
    remaining_usd: float
    limit_usd: float | None = None
    spent_usd: float = 0.0


async def evaluate_usage_budget(
    backend: Any,
    *,
    estimated_cost_usd: float = 0.0,
    limit_usd: float | None = None,
    caller_subsystem: str = "pantheon",
    provider: str | None = None,
) -> BudgetDecision:
    """Evaluate the KNEMON budget gate.

    Mirrors the historical implementation. The cap selection order is:

    1. ``MNEMOS_KNEMON_PROVIDER_BUDGET_CAPS_USD`` — CSV of ``<provider>=<usd>``
       pairs. If the resolved ``provider`` has a per-provider cap, that wins
       (and overrides any ``limit_usd`` the caller passed in). This is the
       F05 path: KNEMON operator configures ``anthropic=1.0`` and a request
       routed under the GRAEAE ``claude`` alias arrives canonicalised to
       ``anthropic``; we apply the ``anthropic`` cap.
    2. ``MNEMOS_KNEMON_WEEKLY_BUDGET_CAP_USD`` — the global cap. If set to
       ``0`` the gate is fully open (returning ``ALLOW`` with
       ``remaining_usd=inf``), matching production semantics for "operator
       disabled the cap".
    3. ``limit_usd`` argument — last-resort explicit cap, used by tests that
       don't want to round-trip through env vars.

    NB: we resolve ``_rows`` via ``sys.modules[...]`` lookup rather than
    the module-level closure so pytest's
    ``monkeypatch.setattr("mnemos.domain.knemon.budget._rows", ...)`` is
    honoured by tests that drive the ledger row fetcher directly.
    """
    import os as _os
    import sys as _sys

    # Cap selection.
    cap = limit_usd
    provider_caps = _parse_provider_caps(
        _os.environ.get("MNEMOS_KNEMON_PROVIDER_BUDGET_CAPS_USD", "")
    )
    if provider is not None and provider in provider_caps:
        cap = provider_caps[provider]
    elif cap is None:
        weekly = _os.environ.get("MNEMOS_KNEMON_WEEKLY_BUDGET_CAP_USD")
        if weekly is not None:
            try:
                cap = float(weekly)
            except (TypeError, ValueError):
                cap = None
    if cap == 0:
        return BudgetDecision(
            allowed=True,
            verdict=BudgetVerdict.CAP_UNCONFIGURED,
            reason="weekly budget cap disabled (0)",
            remaining_usd=float("inf"),
            limit_usd=None,
            spent_usd=0.0,
        )

    # Row fetch — resolve through ``sys.modules`` to honour monkeypatch.
    # The SQL we pass mirrors the historical KNEMON query so test fixtures
    # that pattern-match on ``FROM usage_ledger`` + ``SUM(est_cost_usd)``
    # keep working.
    _mod = _sys.modules.get("mnemos.domain.knemon.budget")
    rows_fn = getattr(_mod, "_rows", _default_rows) if _mod is not None else _default_rows
    rows = await rows_fn(
        backend,
        "SELECT SUM(est_cost_usd) AS spent_usd FROM usage_ledger "
        "WHERE caller_subsystem = $1 AND created > NOW() - INTERVAL '7 days' "
        "AND provider = $2",
        {"provider": provider, "caller": caller_subsystem},
    )

    spent_usd = 0.0
    if rows:
        first = rows[0]
        if isinstance(first, dict):
            spent_usd = float(first.get("spent_usd") or 0.0)

    if cap is None:
        return BudgetDecision(
            allowed=True,
            verdict=BudgetVerdict.CAP_UNCONFIGURED,
            reason="no per-call budget cap configured",
            remaining_usd=float("inf"),
            limit_usd=None,
            spent_usd=spent_usd,
        )
    if spent_usd + estimated_cost_usd >= cap:
        return BudgetDecision(
            allowed=False,
            verdict=BudgetVerdict.DENY,
            reason=(
                f"weekly budget exhausted: spent ${spent_usd:.4f} + "
                f"estimate ${estimated_cost_usd:.4f} exceeds cap ${cap:.4f}"
            ),
            remaining_usd=max(0.0, cap - spent_usd),
            limit_usd=cap,
            spent_usd=spent_usd,
        )
    return BudgetDecision(
        allowed=True,
        verdict=BudgetVerdict.ALLOW,
        reason="weekly budget within cap",
        remaining_usd=cap - spent_usd,
        limit_usd=cap,
        spent_usd=spent_usd,
    )


def _parse_provider_caps(raw: str) -> dict[str, float]:
    """Parse ``MNEMOS_KNEMON_PROVIDER_BUDGET_CAPS_USD`` (CSV of ``prov=usd``)."""
    if not raw:
        return {}
    out: dict[str, float] = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        key, _, value = chunk.partition("=")
        key = key.strip()
        try:
            out[key] = float(value.strip())
        except (TypeError, ValueError):
            continue
    return out


def install() -> None:
    """Wire the stub under ``mnemos.domain.knemon.budget`` in ``sys.modules``."""
    # Create the namespace packages first (PEP 420 style — no __init__.py).
    # We must ALSO pin the parent->child attribute on every parent namespace
    # package so pytest's ``monkeypatch.setattr("mnemos.domain.knemon...")``
    # string-style resolution works. Without that, ``getattr(mnemos.domain,
    # 'knemon')`` raises AttributeError even though the module is in
    # ``sys.modules`` (the parent namespace package object only knows about
    # children that have been explicitly imported into its ``__dict__``).
    #
    # IMPORTANT: this function is called from conftest at MODULE LOAD time,
    # before pytest's own rootdir/conftest discovery populates
    # ``mnemos`` / ``mnemos.api`` / ``mnemos.domain`` / ``mnemos.core`` via
    # its pythonpath+editable-finder mechanism. We MUST NOT clobber an
    # already-existing namespace package — only create it if absent and
    # only pin attributes for the knemon chain.
    mnemos_pkg = sys.modules.get("mnemos")
    if mnemos_pkg is None:
        mnemos_pkg = types.ModuleType("mnemos")
        mnemos_pkg.__path__ = []  # mark as a namespace package
        sys.modules["mnemos"] = mnemos_pkg
    domain_pkg = sys.modules.get("mnemos.domain")
    if domain_pkg is None:
        domain_pkg = types.ModuleType("mnemos.domain")
        domain_pkg.__path__ = []
        sys.modules["mnemos.domain"] = domain_pkg
    if not hasattr(mnemos_pkg, "domain"):
        mnemos_pkg.domain = domain_pkg  # type: ignore[attr-defined]
    knemon_pkg = sys.modules.get("mnemos.domain.knemon")
    if knemon_pkg is None:
        knemon_pkg = types.ModuleType("mnemos.domain.knemon")
        knemon_pkg.__path__ = []
        sys.modules["mnemos.domain.knemon"] = knemon_pkg
    if not hasattr(domain_pkg, "knemon"):
        domain_pkg.knemon = knemon_pkg  # type: ignore[attr-defined]
    budget_mod = sys.modules.get("mnemos.domain.knemon.budget")
    if budget_mod is None:
        budget_mod = types.ModuleType("mnemos.domain.knemon.budget")
        sys.modules["mnemos.domain.knemon.budget"] = budget_mod
    budget_mod._rows = _rows
    budget_mod.BudgetVerdict = BudgetVerdict
    budget_mod.BudgetDecision = BudgetDecision
    budget_mod.evaluate_usage_budget = evaluate_usage_budget
    if not hasattr(knemon_pkg, "budget"):
        knemon_pkg.budget = budget_mod  # type: ignore[attr-defined]
