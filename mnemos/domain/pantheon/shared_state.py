"""Shared-state policy for multi-worker PANTHEON deployments."""

from __future__ import annotations

import os

LOCAL_SHARED_STATE_ENV = "MNEMOS_PANTHEON_ALLOW_LOCAL_SHARED_STATE"


class SharedStateUnavailable(RuntimeError):
    """Raised when an authoritative shared-state operation cannot complete."""


def allow_local_shared_state() -> bool:
    """Return whether the explicit development/single-worker opt-out is set."""
    return os.getenv(LOCAL_SHARED_STATE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


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
