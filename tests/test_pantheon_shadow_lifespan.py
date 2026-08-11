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

    settings = SimpleNamespace(
        pantheon=SimpleNamespace(shadow_no_auth=False),
        auth=SimpleNamespace(enabled=True),
    )
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


def test_shadow_lifespan_refuses_authenticated_mode_when_auth_is_disabled(monkeypatch):
    import mnemos.core.lifecycle as lifecycle
    from mnemos.api import pantheon_shadow

    settings = SimpleNamespace(
        pantheon=SimpleNamespace(shadow_no_auth=False),
        auth=SimpleNamespace(enabled=False),
    )

    async def _must_not_build(_settings):
        raise AssertionError("unsafe configuration must fail before opening persistence")

    monkeypatch.setattr(lifecycle, "_persistence_backend", None)
    monkeypatch.setattr(lifecycle, "build_configured_persistence_backend", _must_not_build)
    monkeypatch.setattr(pantheon_shadow, "get_settings", lambda: settings)

    async def _run():
        with pytest.raises(RuntimeError, match="MNEMOS_AUTH_ENABLED=true"):
            async with pantheon_shadow._shadow_lifespan(FastAPI()):
                pass

    asyncio.run(_run())


def test_shadow_health_fails_when_persistence_is_unavailable(monkeypatch):
    import mnemos.core.lifecycle as lifecycle
    from mnemos.api import pantheon_shadow

    monkeypatch.setattr(lifecycle, "_persistence_backend", None)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(pantheon_shadow.health())

    assert exc_info.value.status_code == 503


def test_shadow_health_fails_when_shared_state_is_unavailable(monkeypatch):
    import mnemos.core.lifecycle as lifecycle
    from mnemos.api import pantheon_shadow
    from mnemos.domain.pantheon import gateway
    from mnemos.domain.pantheon.shared_state import SharedStateUnavailable

    async def _unavailable():
        raise SharedStateUnavailable("shared state unavailable; configure MNEMOS_NATS_URL")

    monkeypatch.setattr(lifecycle, "_persistence_backend", object())
    monkeypatch.setattr(gateway, "require_shared_state", _unavailable)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(pantheon_shadow.health())

    assert exc_info.value.status_code == 503
    assert "MNEMOS_NATS_URL" in str(exc_info.value.detail)
