"""PANTHEON budget pre-gate delegated to KNEMON.

PANTHEON does not compute spend. It supplies only a pre-dispatch cost estimate
and asks KNEMON to evaluate the ledger-backed weekly cap. Budget verdict types
are re-exported for the historical import path.
"""

from __future__ import annotations

from typing import Any

from mnemos.domain.knemon.budget import BudgetDecision, BudgetVerdict, evaluate_usage_budget
from mnemos.domain.pantheon.catalog import canonical_provider


async def evaluate_budget(
    *,
    backend: Any,
    estimated_cost_usd: float = 0.0,
    limit_usd: float | None = None,
    caller_subsystem: str = "pantheon",
    provider: str | None = None,
) -> BudgetDecision:
    """Delegate PANTHEON affordability checks to KNEMON's usage ledger.

    The ``provider`` argument is run through :func:`catalog.canonical_provider`
    so both sides of the admission pair — KNEMON's per-provider cap query and
    the ledger row PANTHEON records afterward — speak the same identity. F05
    review found that PANTHEON's ``claude`` alias and the registry key
    ``anthropic`` were treated as separate providers: a cap configured under
    one was silently bypassed when the request came in under the other. Funnel
    every provider string through this one helper to collapse the alias pair.
    """
    return await evaluate_usage_budget(
        backend,
        estimated_cost_usd=estimated_cost_usd,
        limit_usd=limit_usd,
        caller_subsystem=caller_subsystem,
        provider=canonical_provider(provider) if provider else None,
    )


__all__ = ["BudgetDecision", "BudgetVerdict", "evaluate_budget"]
