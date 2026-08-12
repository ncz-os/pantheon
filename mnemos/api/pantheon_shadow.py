"""Shadow OpenAI-compatible PANTHEON app.

Run on :4101 during PANTHEON phase B. This module intentionally does not touch
VIP :4100 or Caddy; operators can launch it with:

    MNEMOS_PANTHEON_ENABLED=true uvicorn mnemos.api.pantheon_shadow:app --host 127.0.0.1 --port 4101

By default the shadow app installs a local no-auth dependency override for
loopback validation only. Set MNEMOS_PANTHEON_SHADOW_NO_AUTH=false to exercise
the configured auth dependency instead.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from mnemos.api.dependencies import UserContext, configure_auth, get_current_user
from mnemos.api.routes.pantheon import openai_router, router as pantheon_router
from mnemos.core.config import get_settings
from mnemos.core.rate_limit import (
    RateLimitExceeded,
    SlowAPIMiddleware,
    _rate_limit_exceeded_handler,
    limiter,
)

_MISSING = object()

# No-auth mode grants role=root, so it MUST be a direct loopback connection only.
# These are the TCP-peer addresses we accept; a proxied request (caddy/nginx in
# front) surfaces a loopback peer too, so we additionally reject any request that
# carries forwarding headers.
_LOOPBACK_PEERS = frozenset({"127.0.0.1", "::1", "localhost"})
_FORWARD_HEADERS = ("x-forwarded-for", "x-real-ip", "forwarded", "x-forwarded-host")


async def _shadow_current_user(request: Request) -> UserContext:
    # SECURITY (review #5): shadow_no_auth returns an unconditional role=root
    # context. That is only acceptable for direct loopback validation. FAIL
    # CLOSED for anything else — a non-loopback TCP peer, or any request that
    # traversed a proxy — rather than silently handing root to a remote caller.
    client_host = (request.client.host if request.client else "") or ""
    if client_host not in _LOOPBACK_PEERS:
        raise HTTPException(
            status_code=401,
            detail=(
                "pantheon shadow no-auth mode is loopback-only; remote requests "
                "require configured auth (set MNEMOS_PANTHEON_SHADOW_NO_AUTH=false)"
            ),
        )
    if any(request.headers.get(h) for h in _FORWARD_HEADERS):
        raise HTTPException(
            status_code=401,
            detail=(
                "pantheon shadow no-auth mode rejects proxied requests; it must be "
                "reached directly over loopback, not through a reverse proxy"
            ),
        )
    settings = get_settings().auth
    return UserContext(
        user_id=settings.personal_user_id,
        group_ids=[],
        role="root",
        namespace=settings.default_namespace,
        authenticated=False,
    )


@asynccontextmanager
async def _shadow_lifespan(shadow_app: FastAPI) -> AsyncIterator[None]:
    """Initialize persistence and auth needed by the standalone gateway."""
    import mnemos.core.lifecycle as lifecycle

    settings = get_settings()
    if not settings.pantheon.shadow_no_auth and not settings.auth.enabled:
        raise RuntimeError(
            "PANTHEON authenticated mode requires MNEMOS_AUTH_ENABLED=true; "
            "use MNEMOS_PANTHEON_SHADOW_NO_AUTH=true only for direct loopback validation"
        )
    owned_backend = None
    owned_pool = None
    nats_jetstream = None
    audit_consumer_task: asyncio.Task[None] | None = None
    audit_consumer_ready: asyncio.Event | None = None
    if lifecycle._persistence_backend is None:
        backend_type, owned_backend = await lifecycle.build_configured_persistence_backend(settings)
        lifecycle._persistence_backend = owned_backend
        if backend_type == "postgres":
            owned_pool = getattr(owned_backend, "_pool", None)
            lifecycle._pool = owned_pool
    shadow_app.state.persistence_backend = lifecycle._persistence_backend
    shadow_app.state.pool = lifecycle._pool
    previous_override: Any = _MISSING
    codex_oauth = None
    try:
        if settings.pantheon.shadow_no_auth:
            previous_override = shadow_app.dependency_overrides.get(get_current_user, _MISSING)
            shadow_app.dependency_overrides[get_current_user] = _shadow_current_user
        else:
            # Core calls this with None (mnemos/core/lifecycle.py:877); the
            # signature is configure_auth(config: dict | None). Passing an
            # asyncpg Pool made it call config.get(...) on a Pool, which has no
            # .get, so an auth-enabled shadow gateway died at startup.
            configure_auth(None)
        nats_settings = getattr(settings, "nats", None)
        if nats_settings is not None:
            from mnemos.nats import connect_nats

            nats_jetstream = await connect_nats(nats_settings.url, nats_settings.token)
            if nats_settings.audit_consumer_enabled:
                from mnemos.workers.pantheon_routing_audit_consumer import consumer_loop

                audit_consumer_ready = asyncio.Event()
                audit_consumer_task = asyncio.create_task(
                    consumer_loop(
                        lifecycle._persistence_backend,
                        settings=settings,
                        ready_event=audit_consumer_ready,
                    ),
                    name="pantheon-routing-audit-consumer",
                )
        shadow_app.state.audit_consumer_task = audit_consumer_task
        shadow_app.state.audit_consumer_ready = audit_consumer_ready
        try:
            from mnemos.domain.pantheon import codex_oauth as _codex_oauth

            codex_oauth = _codex_oauth
            codex_oauth.start_background_refresh()
        except Exception:
            codex_oauth = None
        yield
    finally:
        if audit_consumer_task is not None:
            audit_consumer_task.cancel()
            await asyncio.gather(audit_consumer_task, return_exceptions=True)
        if codex_oauth is not None:
            await codex_oauth.stop_background_refresh()
        if settings.pantheon.shadow_no_auth:
            if previous_override is _MISSING:
                shadow_app.dependency_overrides.pop(get_current_user, None)
            else:
                shadow_app.dependency_overrides[get_current_user] = previous_override
        if owned_backend is not None:
            try:
                await owned_backend.close()
            finally:
                if lifecycle._pool is owned_pool:
                    lifecycle._pool = None
                if lifecycle._persistence_backend is owned_backend:
                    lifecycle._persistence_backend = None
        if nats_jetstream is not None:
            # connect_nats returns the nats-py JetStream context. Core does not
            # expose a shutdown helper, so close the verified owning connection
            # carried by that context rather than leaving it behind on lifespan
            # restarts.
            nats_client = getattr(nats_jetstream, "_nc", None)
            drain = getattr(nats_client, "drain", None)
            if drain is not None:
                await drain()


app = FastAPI(
    title="PANTHEON OpenAI-compatible shadow gateway",
    version="0.2-shadow",
    lifespan=_shadow_lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.include_router(openai_router)
app.include_router(pantheon_router)


@app.get("/health")
async def health() -> dict[str, object]:
    import mnemos.core.lifecycle as lifecycle
    from mnemos.domain.pantheon import gateway
    from mnemos.domain.pantheon.shared_state import SharedStateUnavailable

    backend = lifecycle._persistence_backend
    if backend is None:
        raise HTTPException(status_code=503, detail="persistence backend unavailable")
    try:
        database_ok = bool(await asyncio.wait_for(backend.ping(), timeout=5.0))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"persistence backend unhealthy: {exc}") from exc
    if not database_ok:
        raise HTTPException(status_code=503, detail="persistence backend unhealthy")
    try:
        await gateway.require_shared_state()
    except SharedStateUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    all_settings = get_settings()
    settings = all_settings.pantheon
    nats_settings = getattr(all_settings, "nats", None)
    if nats_settings is not None and nats_settings.publish_pantheon_routing:
        from mnemos.nats.client import publishing_enabled

        if not publishing_enabled():
            raise HTTPException(status_code=503, detail="routing audit publisher unavailable")
    audit_task = getattr(app.state, "audit_consumer_task", None)
    audit_ready = getattr(app.state, "audit_consumer_ready", None)
    if nats_settings is not None and nats_settings.audit_consumer_enabled:
        if audit_task is None or audit_task.done() or audit_ready is None or not audit_ready.is_set():
            raise HTTPException(status_code=503, detail="routing audit consumer unavailable")
    return {
        "status": "ok",
        "components": {"database": "ok", "shared_state": "ok"},
        "service": "pantheon-shadow",
        "shadow_port": settings.shadow_port,
        "shadow_no_auth": settings.shadow_no_auth,
        "vip_4100_untouched": True,
    }
