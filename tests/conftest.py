from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _single_worker_shared_state_mode(monkeypatch):
    """The repository test process is intentionally a one-worker dev runtime."""
    monkeypatch.setenv("MNEMOS_PANTHEON_ALLOW_LOCAL_SHARED_STATE", "true")
