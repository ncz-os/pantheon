"""Shadow OpenAI-compatible PANTHEON app.

Run on :4101 during PANTHEON phase B. This module intentionally does not touch
VIP :4100 or Caddy; operators can launch it with:

    MNEMOS_PANTHEON_ENABLED=true uvicorn mnemos.api.pantheon_shadow:app --host 127.0.0.1 --port 4101

By default the shadow app installs a local no-auth dependency override for
loopback validation only. Set MNEMOS_PANTHEON_SHADOW_NO_AUTH=false to exercise
the configured auth dependency instead.
"""

from __future__ import annotations

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
    """Initialize only the auth surface needed by the shadow routers."""
    settings = get_settings()
    previous_override: Any = _MISSING
    if settings.pantheon.shadow_no_auth:
        previous_override = shadow_app.dependency_overrides.get(get_current_user, _MISSING)
        shadow_app.dependency_overrides[get_current_user] = _shadow_current_user
    else:
        configure_auth(None)
    codex_oauth = None
    try:
        from mnemos.domain.pantheon import codex_oauth as _codex_oauth

        codex_oauth = _codex_oauth
        codex_oauth.start_background_refresh()
    except Exception:
        codex_oauth = None
    try:
        yield
    finally:
        if codex_oauth is not None:
            await codex_oauth.stop_background_refresh()
        if settings.pantheon.shadow_no_auth:
            if previous_override is _MISSING:
                shadow_app.dependency_overrides.pop(get_current_user, None)
            else:
                shadow_app.dependency_overrides[get_current_user] = previous_override


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
    settings = get_settings().pantheon
    return {
        "status": "ok",
        "service": "pantheon-shadow",
        "shadow_port": settings.shadow_port,
        "shadow_no_auth": settings.shadow_no_auth,
        "vip_4100_untouched": True,
    }
