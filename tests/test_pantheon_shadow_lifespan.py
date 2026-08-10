from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException


def test_shadow_lifespan_owns_configured_persistence_backend(monkeypatch):
    import mnemos.core.lifecycle as lifecycle
    from mnemos.api import pantheon_shadow
    from mnemos.domain.pantheon import codex_oauth

    class _Backend:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    backend = _Backend()

    async def _build(_settings):
        return "sqlite", backend

    async def _stop_refresh():
        return None

    settings = SimpleNamespace(pantheon=SimpleNamespace(shadow_no_auth=False))
    monkeypatch.setattr(lifecycle, "_persistence_backend", None)
    monkeypatch.setattr(lifecycle, "build_configured_persistence_backend", _build)
    monkeypatch.setattr(pantheon_shadow, "get_settings", lambda: settings)
    monkeypatch.setattr(pantheon_shadow, "configure_auth", lambda _pool: None)
    monkeypatch.setattr(codex_oauth, "start_background_refresh", lambda: None)
    monkeypatch.setattr(codex_oauth, "stop_background_refresh", _stop_refresh)

    async def _run():
        shadow_app = FastAPI()
        async with pantheon_shadow._shadow_lifespan(shadow_app):
            assert lifecycle._persistence_backend is backend
            assert shadow_app.state.persistence_backend is backend
        assert lifecycle._persistence_backend is None

    asyncio.run(_run())
    assert backend.closed is True


def test_shadow_health_fails_when_persistence_is_unavailable(monkeypatch):
    import mnemos.core.lifecycle as lifecycle
    from mnemos.api import pantheon_shadow

    monkeypatch.setattr(lifecycle, "_persistence_backend", None)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(pantheon_shadow.health())

    assert exc_info.value.status_code == 503
