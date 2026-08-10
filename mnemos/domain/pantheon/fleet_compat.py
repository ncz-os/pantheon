"""Compatibility bridge for the supported core 6.0/GRAEAE 0.1 release lines.

Core's NATS resilience rewrite removed two private Redis discovery helpers
without a coordinated GRAEAE release.  GRAEAE 0.1 still imports those helpers
for its response cache and quality tracker.  Install their original, small
semantics only when the running core does not provide them, before importing
GRAEAE.  A future coordinated release can remove this bridge after PANTHEON's
dependency floor is raised.
"""

from __future__ import annotations

from typing import Any


def install_core_graeae_compat() -> None:
    """Supply the two core 6.0 accessors required by GRAEAE 0.1."""
    import mnemos.core.resilience as resilience

    if not hasattr(resilience, "_redis_requested"):

        def _redis_requested(settings: Any) -> bool:
            rate_limit = getattr(settings, "rate_limit", None)
            storage_uri = getattr(rate_limit, "storage_uri", getattr(rate_limit, "storage", "memory://"))
            return str(storage_uri).startswith(("redis://", "rediss://"))

        resilience._redis_requested = _redis_requested

    if not hasattr(resilience, "_get_lifecycle_redis_client"):

        def _get_lifecycle_redis_client() -> Any | None:
            try:
                from mnemos.core.lifecycle import get_redis_client
            except (ImportError, AttributeError):
                return None
            return get_redis_client()

        resilience._get_lifecycle_redis_client = _get_lifecycle_redis_client


__all__ = ["install_core_graeae_compat"]
