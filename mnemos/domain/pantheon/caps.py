"""Per-session usage caps for PANTHEON consultation-only models."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from mnemos.domain.pantheon.shared_state import SharedStateUnavailable, allow_local_shared_state


@dataclass(frozen=True)
class ConsultationCapResult:
    allowed: bool
    cap: int
    used: int
    remaining: int


class InMemoryConsultationCapBucket:
    """Explicit development/single-worker cap keyed by user and session."""

    def __init__(self) -> None:
        self._counts: dict[tuple[str, str], int] = {}
        self._lock = Lock()

    def check_and_increment(self, *, user_id: str, session_id: str, cap: int) -> ConsultationCapResult:
        key = (user_id, session_id)
        with self._lock:
            used = self._counts.get(key, 0)
            if used >= cap:
                return ConsultationCapResult(
                    allowed=False,
                    cap=cap,
                    used=used,
                    remaining=0,
                )
            used += 1
            self._counts[key] = used
            return ConsultationCapResult(
                allowed=True,
                cap=cap,
                used=used,
                remaining=max(cap - used, 0),
            )

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()


class ConsultationCapBucket:
    """Use the runtime's authoritative JetStream store for atomic cap grants."""

    def __init__(self) -> None:
        self._local = InMemoryConsultationCapBucket()

    async def acheck_and_increment(
        self, *, user_id: str, session_id: str, cap: int
    ) -> ConsultationCapResult:
        # Import lazily to avoid a caps -> gateway -> package import cycle.
        from mnemos.domain.pantheon.gateway import get_runtime

        store = get_runtime().cooldown._store
        shared_increment = getattr(store, "acheck_and_increment_consultation_cap", None)
        if shared_increment is not None:
            try:
                allowed, used = await shared_increment(
                    user_id=user_id,
                    session_id=session_id,
                    cap=cap,
                )
                return ConsultationCapResult(
                    allowed=allowed,
                    cap=cap,
                    used=used,
                    remaining=max(cap - used, 0),
                )
            except SharedStateUnavailable:
                if not allow_local_shared_state():
                    raise

        if allow_local_shared_state():
            return self._local.check_and_increment(user_id=user_id, session_id=session_id, cap=cap)
        raise SharedStateUnavailable(
            "PANTHEON consultation cap shared store is unavailable; configure MNEMOS_NATS_URL "
            "and JetStream, or set MNEMOS_PANTHEON_ALLOW_LOCAL_SHARED_STATE=true only for "
            "development/single-worker use"
        )

    def reset(self) -> None:
        """Reset only the explicit local bucket (primarily for test/dev use)."""
        self._local.reset()


consultation_cap_bucket = ConsultationCapBucket()


__all__ = [
    "ConsultationCapResult",
    "ConsultationCapBucket",
    "InMemoryConsultationCapBucket",
    "consultation_cap_bucket",
]
