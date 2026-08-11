"""Shared-state policy for multi-worker PANTHEON deployments."""

from __future__ import annotations

import os

LOCAL_SHARED_STATE_ENV = "MNEMOS_PANTHEON_ALLOW_LOCAL_SHARED_STATE"


class SharedStateUnavailable(RuntimeError):
    """Raised when an authoritative shared-state operation cannot complete."""


def allow_local_shared_state() -> bool:
    """Return whether process-local state is safe for this deployment.

    An explicit opt-out remains available for development.  Otherwise the
    documented no-NATS, single-worker configuration uses local state, while a
    configured NATS service or an explicitly multi-worker process fails closed.
    ``WEB_CONCURRENCY`` is included because the standalone PANTHEON launcher
    uses it rather than ``MNEMOS_WORKERS``.
    """
    if os.getenv(LOCAL_SHARED_STATE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}:
        return True

    from mnemos.core.config import get_settings

    settings = get_settings()
    if str(settings.nats.url or "").strip():
        return False
    worker_count = int(settings.server.workers)
    web_concurrency = os.getenv("WEB_CONCURRENCY", "").strip()
    if web_concurrency:
        try:
            worker_count = max(worker_count, int(web_concurrency))
        except ValueError:
            # An invalid launcher value must not silently weaken shared-state
            # enforcement; gunicorn will reject it independently at startup.
            return False
    return worker_count == 1


def unavailable_message(operation: str) -> str:
    return (
        f"PANTHEON shared state is unavailable during {operation}; configure MNEMOS_NATS_URL "
        f"and JetStream, or set {LOCAL_SHARED_STATE_ENV}=true only for development/single-worker use"
    )


__all__ = [
    "LOCAL_SHARED_STATE_ENV",
    "SharedStateUnavailable",
    "allow_local_shared_state",
    "unavailable_message",
]
