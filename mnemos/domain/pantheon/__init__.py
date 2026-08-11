"""PANTHEON unified LLM facade domain package."""

# Import order is intentional: validate the supported fleet seam before any
# module below imports GRAEAE.
# ruff: noqa: E402

from mnemos.core.extras import require_extra

require_extra("pantheon")

# Retained for callers that import the fleet hook directly. It deliberately
# does not install deleted private core symbols.
from mnemos.domain.pantheon.fleet_compat import install_core_graeae_compat

install_core_graeae_compat()

from mnemos.domain.pantheon.catalog import list_models, models_response
from mnemos.domain.pantheon.router import RouteDecision, explain_route, route_model
from mnemos.domain.pantheon.triage import recommend

__all__ = [
    "RouteDecision",
    "explain_route",
    "list_models",
    "models_response",
    "recommend",
    "route_model",
]
