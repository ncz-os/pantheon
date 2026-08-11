from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException


@pytest.mark.parametrize("backend_type", ["sqlite", "oracle", "db2", "mysql", "mariadb"])
def test_shadow_lifespan_owns_configured_non_postgres_backend(monkeypatch, backend_type):
    import mnemos.core.lifecycle as lifecycle
    from mnemos.api import pantheon_shadow
    from mnemos.domain.pantheon import codex_oauth

    class _Backend:
        def __init__(self):
            self.closed = False
            self._pool = object()

        async def close(self):
            self.closed = True

    backend = _Backend()

    async def _build(_settings):
        return backend_type, backend

    async def _stop_refresh():
        return None

    settings = SimpleNamespace(
        pantheon=SimpleNamespace(shadow_no_auth=False),
        auth=SimpleNamespace(enabled=True),
    )
    monkeypatch.setattr(lifecycle, "_persistence_backend", None)
    monkeypatch.setattr(lifecycle, "_pool", None)
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
            assert lifecycle._pool is None
            assert shadow_app.state.pool is None
        assert lifecycle._persistence_backend is None

    asyncio.run(_run())
    assert backend.closed is True


def test_shadow_lifespan_exposes_only_postgres_pool_and_bootstraps_nats(monkeypatch):
    import mnemos.core.lifecycle as lifecycle
    import mnemos.nats
    from mnemos.api import pantheon_shadow
    from mnemos.domain.pantheon import codex_oauth
    from mnemos.workers import pantheon_routing_audit_consumer

    pool = object()

    class _Backend:
        def __init__(self):
            self._pool = pool
            self.closed = False

        async def close(self):
            self.closed = True

    class _NatsClient:
        def __init__(self):
            self.drained = False

        async def drain(self):
            self.drained = True

    backend = _Backend()
    nats_client = _NatsClient()
    jetstream = SimpleNamespace(_nc=nats_client)
    consumer_started = asyncio.Event()

    async def _build(_settings):
        return "postgres", backend

    async def _connect(url, token):
        assert (url, token) == ("nats://test", "secret")
        return jetstream

    async def _consumer(active_backend, *, settings, ready_event):
        assert active_backend is backend
        ready_event.set()
        consumer_started.set()
        await asyncio.Event().wait()

    async def _stop_refresh():
        return None

    settings = SimpleNamespace(
        pantheon=SimpleNamespace(shadow_no_auth=False),
        auth=SimpleNamespace(enabled=True),
        nats=SimpleNamespace(
            url="nats://test",
            token="secret",
            audit_consumer_enabled=True,
        ),
    )
    monkeypatch.setattr(lifecycle, "_persistence_backend", None)
    monkeypatch.setattr(lifecycle, "_pool", None)
    monkeypatch.setattr(lifecycle, "build_configured_persistence_backend", _build)
    monkeypatch.setattr(pantheon_shadow, "get_settings", lambda: settings)
    monkeypatch.setattr(pantheon_shadow, "configure_auth", lambda active_pool: active_pool is pool)
    monkeypatch.setattr(mnemos.nats, "connect_nats", _connect)
    monkeypatch.setattr(pantheon_routing_audit_consumer, "consumer_loop", _consumer)
    monkeypatch.setattr(codex_oauth, "start_background_refresh", lambda: None)
    monkeypatch.setattr(codex_oauth, "stop_background_refresh", _stop_refresh)

    async def _run():
        shadow_app = FastAPI()
        async with pantheon_shadow._shadow_lifespan(shadow_app):
            await consumer_started.wait()
            assert lifecycle._pool is pool
            assert shadow_app.state.pool is pool
            assert shadow_app.state.audit_consumer_task is not None
            assert shadow_app.state.audit_consumer_ready.is_set()
        assert lifecycle._pool is None

    asyncio.run(_run())
    assert backend.closed is True
    assert nats_client.drained is True


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

    class _Backend:
        async def ping(self):
            return True

    monkeypatch.setattr(lifecycle, "_persistence_backend", _Backend())
    monkeypatch.setattr(gateway, "require_shared_state", _unavailable)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(pantheon_shadow.health())

    assert exc_info.value.status_code == 503
    assert "MNEMOS_NATS_URL" in str(exc_info.value.detail)


@pytest.mark.parametrize("ping_result", [False, RuntimeError("database down")])
def test_shadow_health_fails_when_persistence_ping_fails(monkeypatch, ping_result):
    import mnemos.core.lifecycle as lifecycle
    from mnemos.api import pantheon_shadow

    class _Backend:
        async def ping(self):
            if isinstance(ping_result, Exception):
                raise ping_result
            return ping_result

    monkeypatch.setattr(lifecycle, "_persistence_backend", _Backend())

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(pantheon_shadow.health())

    assert exc_info.value.status_code == 503
    assert "persistence backend unhealthy" in str(exc_info.value.detail)


def test_shadow_health_reports_component_status_for_default_flags(monkeypatch):
    import mnemos.core.lifecycle as lifecycle
    from mnemos.api import pantheon_shadow
    from mnemos.domain.pantheon import gateway

    class _Backend:
        async def ping(self):
            return True

    async def _shared_state_ready():
        return None

    settings = SimpleNamespace(
        pantheon=SimpleNamespace(shadow_port=4101, shadow_no_auth=True),
        nats=SimpleNamespace(
            publish_pantheon_routing=False,
            audit_consumer_enabled=False,
        ),
    )
    monkeypatch.setattr(lifecycle, "_persistence_backend", _Backend())
    monkeypatch.setattr(gateway, "require_shared_state", _shared_state_ready)
    monkeypatch.setattr(pantheon_shadow, "get_settings", lambda: settings)

    result = asyncio.run(pantheon_shadow.health())

    assert result["status"] == "ok"
    assert result["components"] == {"database": "ok", "shared_state": "ok"}


def test_shadow_health_fails_when_enabled_audit_consumer_is_not_ready(monkeypatch):
    import mnemos.core.lifecycle as lifecycle
    from mnemos.api import pantheon_shadow
    from mnemos.domain.pantheon import gateway

    class _Backend:
        async def ping(self):
            return True

    class _Task:
        def done(self):
            return False

    async def _shared_state_ready():
        return None

    settings = SimpleNamespace(
        pantheon=SimpleNamespace(shadow_port=4101, shadow_no_auth=False),
        nats=SimpleNamespace(
            publish_pantheon_routing=False,
            audit_consumer_enabled=True,
        ),
    )
    pantheon_shadow.app.state.audit_consumer_task = _Task()
    pantheon_shadow.app.state.audit_consumer_ready = asyncio.Event()
    monkeypatch.setattr(lifecycle, "_persistence_backend", _Backend())
    monkeypatch.setattr(gateway, "require_shared_state", _shared_state_ready)
    monkeypatch.setattr(pantheon_shadow, "get_settings", lambda: settings)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(pantheon_shadow.health())

    assert exc_info.value.status_code == 503
    assert "audit consumer unavailable" in str(exc_info.value.detail)


def test_shadow_health_fails_when_enabled_audit_publisher_is_not_ready(monkeypatch):
    import mnemos.core.lifecycle as lifecycle
    import mnemos.nats.client as nats_client
    from mnemos.api import pantheon_shadow
    from mnemos.domain.pantheon import gateway

    class _Backend:
        async def ping(self):
            return True

    async def _shared_state_ready():
        return None

    settings = SimpleNamespace(
        pantheon=SimpleNamespace(shadow_port=4101, shadow_no_auth=False),
        nats=SimpleNamespace(
            publish_pantheon_routing=True,
            audit_consumer_enabled=False,
        ),
    )
    monkeypatch.setattr(lifecycle, "_persistence_backend", _Backend())
    monkeypatch.setattr(gateway, "require_shared_state", _shared_state_ready)
    monkeypatch.setattr(pantheon_shadow, "get_settings", lambda: settings)
    monkeypatch.setattr(nats_client, "publishing_enabled", lambda: False)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(pantheon_shadow.health())

    assert exc_info.value.status_code == 503
    assert "audit publisher unavailable" in str(exc_info.value.detail)
