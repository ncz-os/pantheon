"""Codex ChatGPT-OAuth fallback provider for PANTHEON.

This module owns the pieces formerly kept alive by the standalone doctor:
ChatGPT-OAuth token refresh, live Codex model slug reconciliation, and one
retry on stale/unsupported model slugs. It is fallback-only and disabled unless
explicitly enabled by environment/config.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from mnemos.domain.pantheon.router import RouteDecision

logger = logging.getLogger(__name__)

DEFAULT_GATEWAY_BASE_URL = "http://127.0.0.1:42617/v1"
DEFAULT_REFRESH_URL = "https://chatgpt.com/backend-api/oauth/token"
DEFAULT_CATALOG_URL = "https://chatgpt.com/backend-api/codex/models"
DEFAULT_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
DEFAULT_MODEL = "gpt-5.3-codex-spark"
DEFAULT_CLIENT_VERSION = "0.131.0"
DEFAULT_TOKEN_REFRESH_SECONDS = 50 * 60
DEFAULT_CATALOG_REFRESH_SECONDS = 10 * 60
DEFAULT_LOCK_WAIT_SECONDS = 10.0
DEFAULT_LOCK_TTL_SECONDS = 60.0
DEFAULT_TIMEOUT_SECONDS = 60.0
_AUTH_REFRESH_LOCK = "pantheon:codex-oauth:auth-refresh"
_TRUTHY = {"1", "true", "yes", "y", "on", "enabled"}
_BACKGROUND_TASK: asyncio.Task | None = None


class CodexOAuthUnavailable(Exception):
    """Codex OAuth fallback is disabled or cannot produce a usable token."""


class CodexOAuthHTTPError(Exception):
    def __init__(self, status_code: int, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.retry_after = retry_after


@dataclass(frozen=True)
class CodexOAuthConfig:
    enabled: bool
    auth_json_path: Path
    gateway_base_url: str
    refresh_url: str
    catalog_url: str
    models_cache_path: Path
    slug_cache_path: Path
    client_id: str
    client_version: str
    default_model: str
    token_refresh_seconds: float
    catalog_refresh_seconds: float
    lock_wait_seconds: float
    lock_ttl_seconds: float
    require_distributed_refresh_lock: bool
    timeout_seconds: float


@dataclass(frozen=True)
class AuthState:
    access_token: str | None
    refresh_token: str | None
    account_id: str | None
    last_refresh: datetime | None
    payload: dict[str, Any]


@dataclass(frozen=True)
class CatalogSnapshot:
    slugs: tuple[str, ...]
    models: tuple[dict[str, Any], ...]
    client_version: str | None = None
    fetched_at: str | None = None


@dataclass
class CodexOpenStream:
    decision: RouteDecision
    manager: Any
    response: httpx.Response

    async def aclose(self) -> None:
        await self.manager.__aexit__(None, None, None)


def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _setting(name: str, default: Any = None) -> Any:
    try:
        from mnemos.core.config import get_settings

        return getattr(getattr(get_settings(), "pantheon", None), name, default)
    except Exception:
        return default


def _as_bool(raw: Any, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in _TRUTHY


def _as_float(raw: Any, default: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0.0 else default


def _default_slug_cache_path() -> Path:
    root = os.environ.get("XDG_CACHE_HOME")
    base = Path(root).expanduser() if root else Path.home() / ".cache"
    return base / "mnemos" / "pantheon-codex-models.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.expanduser().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.debug("[PANTHEON] Codex OAuth JSON read failed for %s: %s", path, exc)
        return None


def _client_version_from_cache(path: Path) -> str | None:
    payload = _read_json(path)
    version = payload.get("client_version") if isinstance(payload, dict) else None
    return str(version).strip() if version else None


def get_config() -> CodexOAuthConfig:
    auth_path = Path(
        _env("PANTHEON_CODEX_OAUTH_AUTH_JSON", "MNEMOS_PANTHEON_CODEX_OAUTH_AUTH_JSON", "CODEX_AUTH_JSON")
        or _setting("codex_oauth_auth_json", "~/.codex/auth.json")
    ).expanduser()
    models_cache = Path(
        _env("PANTHEON_CODEX_OAUTH_MODELS_CACHE", "MNEMOS_PANTHEON_CODEX_OAUTH_MODELS_CACHE")
        or _setting("codex_oauth_models_cache_path", "~/.codex/models_cache.json")
    ).expanduser()
    slug_cache = Path(
        _env("PANTHEON_CODEX_OAUTH_SLUG_CACHE", "MNEMOS_PANTHEON_CODEX_OAUTH_SLUG_CACHE")
        or _setting("codex_oauth_slug_cache_path", str(_default_slug_cache_path()))
    ).expanduser()
    enabled = _as_bool(
        _env("PANTHEON_CODEX_OAUTH_FALLBACK_ENABLED", "MNEMOS_PANTHEON_CODEX_OAUTH_FALLBACK_ENABLED"),
        bool(_setting("codex_oauth_fallback_enabled", False)),
    )
    client_version = (
        _env("PANTHEON_CODEX_OAUTH_CLIENT_VERSION", "MNEMOS_PANTHEON_CODEX_OAUTH_CLIENT_VERSION")
        or str(_setting("codex_oauth_client_version", "") or "").strip()
        or _client_version_from_cache(slug_cache)
        or _client_version_from_cache(models_cache)
        or DEFAULT_CLIENT_VERSION
    )
    return CodexOAuthConfig(
        enabled=enabled,
        auth_json_path=auth_path,
        gateway_base_url=(
            _env("PANTHEON_CODEX_OAUTH_BASE_URL", "MNEMOS_PANTHEON_CODEX_OAUTH_BASE_URL")
            or str(_setting("codex_oauth_base_url", DEFAULT_GATEWAY_BASE_URL))
        ).rstrip("/"),
        refresh_url=(
            _env("PANTHEON_CODEX_OAUTH_REFRESH_URL", "MNEMOS_PANTHEON_CODEX_OAUTH_REFRESH_URL")
            or str(_setting("codex_oauth_refresh_url", DEFAULT_REFRESH_URL))
        ),
        catalog_url=(
            _env("PANTHEON_CODEX_OAUTH_CATALOG_URL", "MNEMOS_PANTHEON_CODEX_OAUTH_CATALOG_URL")
            or str(_setting("codex_oauth_catalog_url", DEFAULT_CATALOG_URL))
        ),
        models_cache_path=models_cache,
        slug_cache_path=slug_cache,
        client_id=(
            _env("PANTHEON_CODEX_OAUTH_CLIENT_ID", "MNEMOS_PANTHEON_CODEX_OAUTH_CLIENT_ID")
            or str(_setting("codex_oauth_client_id", DEFAULT_CLIENT_ID))
        ),
        client_version=client_version,
        default_model=(
            _env("PANTHEON_CODEX_OAUTH_DEFAULT_MODEL", "MNEMOS_PANTHEON_CODEX_OAUTH_DEFAULT_MODEL")
            or str(_setting("codex_oauth_default_model", DEFAULT_MODEL))
        ),
        token_refresh_seconds=_as_float(
            _env("PANTHEON_CODEX_OAUTH_TOKEN_REFRESH_SECONDS", "MNEMOS_PANTHEON_CODEX_OAUTH_TOKEN_REFRESH_SECONDS")
            or _setting("codex_oauth_token_refresh_seconds", DEFAULT_TOKEN_REFRESH_SECONDS),
            DEFAULT_TOKEN_REFRESH_SECONDS,
        ),
        catalog_refresh_seconds=_as_float(
            _env("PANTHEON_CODEX_OAUTH_CATALOG_REFRESH_SECONDS", "MNEMOS_PANTHEON_CODEX_OAUTH_CATALOG_REFRESH_SECONDS")
            or _setting("codex_oauth_catalog_refresh_seconds", DEFAULT_CATALOG_REFRESH_SECONDS),
            DEFAULT_CATALOG_REFRESH_SECONDS,
        ),
        lock_wait_seconds=_as_float(
            _env("PANTHEON_CODEX_OAUTH_LOCK_WAIT_SECONDS", "MNEMOS_PANTHEON_CODEX_OAUTH_LOCK_WAIT_SECONDS")
            or _setting("codex_oauth_lock_wait_seconds", DEFAULT_LOCK_WAIT_SECONDS),
            DEFAULT_LOCK_WAIT_SECONDS,
        ),
        lock_ttl_seconds=_as_float(
            _env("PANTHEON_CODEX_OAUTH_LOCK_TTL_SECONDS", "MNEMOS_PANTHEON_CODEX_OAUTH_LOCK_TTL_SECONDS")
            or _setting("codex_oauth_lock_ttl_seconds", DEFAULT_LOCK_TTL_SECONDS),
            DEFAULT_LOCK_TTL_SECONDS,
        ),
        require_distributed_refresh_lock=_as_bool(
            _env(
                "PANTHEON_CODEX_OAUTH_REQUIRE_DISTRIBUTED_LOCK",
                "MNEMOS_PANTHEON_CODEX_OAUTH_REQUIRE_DISTRIBUTED_LOCK",
            ),
            bool(_setting("codex_oauth_require_distributed_lock", True)),
        ),
        timeout_seconds=_as_float(
            _env("PANTHEON_CODEX_OAUTH_TIMEOUT_SECONDS", "MNEMOS_PANTHEON_CODEX_OAUTH_TIMEOUT_SECONDS")
            or _setting("codex_oauth_timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
            DEFAULT_TIMEOUT_SECONDS,
        ),
    )


def _parse_time(raw: Any) -> datetime | None:
    if not raw:
        return None
    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(float(raw), UTC)
        except (OSError, ValueError):
            return None
    text = str(raw).strip()
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except ValueError:
        return None


def read_auth_state(config: CodexOAuthConfig | None = None) -> AuthState | None:
    config = config or get_config()
    payload = _read_json(config.auth_json_path)
    if not isinstance(payload, dict):
        return None
    tokens = payload.get("tokens")
    if not isinstance(tokens, dict):
        tokens = {}
    access = tokens.get("access_token")
    refresh = tokens.get("refresh_token")
    account = tokens.get("account_id")
    return AuthState(
        access_token=str(access).strip() if access else None,
        refresh_token=str(refresh).strip() if refresh else None,
        account_id=str(account).strip() if account else None,
        last_refresh=_parse_time(payload.get("last_refresh") or tokens.get("last_refresh") or tokens.get("expires_at")),
        payload=payload,
    )


def _is_stale(state: AuthState, config: CodexOAuthConfig) -> bool:
    if not state.access_token:
        return True
    if state.last_refresh is None:
        return False
    age = (datetime.now(UTC) - state.last_refresh.astimezone(UTC)).total_seconds()
    return age >= config.token_refresh_seconds


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = 0o600
    try:
        mode = path.stat().st_mode & 0o777
    except FileNotFoundError:
        pass
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def get_http_client() -> httpx.AsyncClient:
    from mnemos.domain.pantheon.gateway import get_http_client as gateway_client

    return gateway_client()


def _runtime_lock_store() -> Any | None:
    try:
        from mnemos.domain.pantheon.gateway import get_runtime

        return getattr(getattr(get_runtime(), "cooldown", None), "_store", None)
    except Exception:
        return None


async def _try_acquire_distributed_lock(store: Any, config: CodexOAuthConfig, owner: str) -> bool:
    acquire = getattr(store, "atry_acquire_lock", None)
    if acquire is None:
        return False
    try:
        return bool(await acquire(_AUTH_REFRESH_LOCK, owner, ttl_seconds=config.lock_ttl_seconds))
    except Exception as exc:
        logger.debug("[PANTHEON] Codex OAuth distributed refresh lock unavailable: %s", exc)
        return False


def _distributed_lock_store() -> Any | None:
    store = _runtime_lock_store()
    return store if getattr(store, "atry_acquire_lock", None) is not None else None


async def _release_distributed_lock(store: Any | None, owner: str) -> None:
    release = getattr(store, "arelease_lock", None)
    if release is None:
        return
    try:
        await release(_AUTH_REFRESH_LOCK, owner)
    except Exception as exc:
        logger.debug("[PANTHEON] Codex OAuth distributed refresh lock release failed: %s", exc)


class TokenManager:
    def __init__(self) -> None:
        self._local_lock = asyncio.Lock()

    async def access_state(self, *, force_refresh: bool = False) -> AuthState:
        config = get_config()
        if not config.enabled:
            raise CodexOAuthUnavailable("Codex OAuth fallback disabled")
        state = read_auth_state(config)
        if state is None or (not state.access_token and not state.refresh_token):
            raise CodexOAuthUnavailable("Codex OAuth auth.json absent or empty")
        if not force_refresh and not _is_stale(state, config):
            return state

        async with self._local_lock:
            state = read_auth_state(config)
            if state is None or (not state.access_token and not state.refresh_token):
                raise CodexOAuthUnavailable("Codex OAuth auth.json absent or empty")
                if not force_refresh and not _is_stale(state, config):
                    return state
                owner = f"{os.getpid()}:{uuid.uuid4().hex}"
                lock_store = _distributed_lock_store()
                if lock_store is None and config.require_distributed_refresh_lock:
                    raise CodexOAuthUnavailable("Codex OAuth distributed refresh lock unavailable")
                if lock_store is not None and not await _try_acquire_distributed_lock(lock_store, config, owner):
                    return await self._wait_for_peer_refresh(config, state)
            try:
                state = read_auth_state(config)
                if state is None or (not state.access_token and not state.refresh_token):
                    raise CodexOAuthUnavailable("Codex OAuth auth.json absent or empty")
                if not force_refresh and not _is_stale(state, config):
                    return state
                return await self._refresh_locked(config, state)
            finally:
                await _release_distributed_lock(lock_store, owner)

    async def _wait_for_peer_refresh(self, config: CodexOAuthConfig, before: AuthState) -> AuthState:
        deadline = asyncio.get_running_loop().time() + config.lock_wait_seconds
        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.1)
            current = read_auth_state(config)
            if current is None:
                continue
            changed = (
                current.refresh_token != before.refresh_token
                or current.access_token != before.access_token
                or current.last_refresh != before.last_refresh
            )
            if current.access_token and changed:
                return current
        current = read_auth_state(config)
        if current is not None and current.access_token:
            return current
        raise CodexOAuthUnavailable("Codex OAuth refresh did not complete")

    async def _refresh_locked(self, config: CodexOAuthConfig, state: AuthState) -> AuthState:
        if not state.refresh_token:
            if state.access_token:
                return state
            raise CodexOAuthUnavailable("Codex OAuth refresh token missing")
        client = get_http_client()
        response = await client.post(
            config.refresh_url,
            data={
                "grant_type": "refresh_token",
                "refresh_token": state.refresh_token,
                "client_id": config.client_id,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
            timeout=config.timeout_seconds,
        )
        if response.status_code >= 400:
            raise CodexOAuthUnavailable(f"Codex OAuth refresh failed with HTTP {response.status_code}")
        data = response.json()
        token_data = data.get("tokens") if isinstance(data.get("tokens"), dict) else data
        access = token_data.get("access_token")
        refresh = token_data.get("refresh_token") or state.refresh_token
        if not access:
            raise CodexOAuthUnavailable("Codex OAuth refresh response did not include access_token")
        payload = dict(state.payload)
        tokens = dict(payload.get("tokens") if isinstance(payload.get("tokens"), dict) else {})
        tokens["access_token"] = access
        tokens["refresh_token"] = refresh
        if token_data.get("id_token"):
            tokens["id_token"] = token_data["id_token"]
        if token_data.get("account_id") or state.account_id:
            tokens["account_id"] = token_data.get("account_id") or state.account_id
        payload["tokens"] = tokens
        payload["auth_mode"] = payload.get("auth_mode") or "chatgpt"
        refreshed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        payload["last_refresh"] = refreshed_at
        tokens["last_refresh"] = refreshed_at
        _atomic_write_json(config.auth_json_path, payload)
        refreshed = read_auth_state(config)
        if refreshed is None or not refreshed.access_token:
            raise CodexOAuthUnavailable("Codex OAuth refreshed auth.json is unreadable")
        return refreshed


token_manager = TokenManager()


def _models_from_payload(payload: Any) -> tuple[dict[str, Any], ...]:
    raw = []
    if isinstance(payload, dict):
        raw = payload.get("models") or payload.get("data") or []
    elif isinstance(payload, list):
        raw = payload
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        slug = str(item.get("slug") or item.get("id") or item.get("model") or "").strip()
        if not slug:
            continue
        if str(item.get("visibility") or "list").lower() == "hide":
            continue
        out.append(
            {
                "slug": slug,
                "display_name": item.get("display_name") or item.get("name") or slug,
                "visibility": item.get("visibility"),
                "supported_in_api": item.get("supported_in_api"),
                "priority": item.get("priority"),
            }
        )
    return tuple(out)


def _snapshot_from_payload(payload: Any) -> CatalogSnapshot | None:
    models = _models_from_payload(payload)
    if not models:
        return None
    slugs: list[str] = []
    seen: set[str] = set()
    for model in models:
        slug = str(model["slug"])
        if slug not in seen:
            slugs.append(slug)
            seen.add(slug)
    client_version = payload.get("client_version") if isinstance(payload, dict) else None
    fetched_at = payload.get("fetched_at") if isinstance(payload, dict) else None
    return CatalogSnapshot(tuple(slugs), models, str(client_version) if client_version else None, str(fetched_at) if fetched_at else None)


def _slug_cache_payload(snapshot: CatalogSnapshot, config: CodexOAuthConfig) -> dict[str, Any]:
    return {
        "schema": "mnemos.pantheon.codex_oauth.models.v1",
        "fetched_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "client_version": snapshot.client_version or config.client_version,
        "models": list(snapshot.models),
    }


class ModelCatalog:
    def __init__(self) -> None:
        self._snapshot: CatalogSnapshot | None = None
        self._last_live_refresh = 0.0
        self._lock = asyncio.Lock()

    def load_cached(self, config: CodexOAuthConfig | None = None) -> CatalogSnapshot | None:
        config = config or get_config()
        for path in (config.slug_cache_path, config.models_cache_path):
            snapshot = _snapshot_from_payload(_read_json(path))
            if snapshot is not None:
                self._snapshot = snapshot
                return snapshot
        return self._snapshot

    async def refresh(self, *, force: bool = False) -> CatalogSnapshot | None:
        config = get_config()
        cached = self.load_cached(config)
        now = asyncio.get_running_loop().time()
        if not force and cached is not None and now - self._last_live_refresh < config.catalog_refresh_seconds:
            return cached
        if not config.enabled:
            return cached
        async with self._lock:
            cached = self.load_cached(config)
            now = asyncio.get_running_loop().time()
            if not force and cached is not None and now - self._last_live_refresh < config.catalog_refresh_seconds:
                return cached
            try:
                state = await token_manager.access_state()
                snapshot = await self._fetch_live(config, state)
                self._snapshot = snapshot
                self._last_live_refresh = now
                try:
                    _atomic_write_json(config.slug_cache_path, _slug_cache_payload(snapshot, config))
                except Exception as exc:
                    logger.debug("[PANTHEON] Codex model slug cache write failed: %s", exc)
                return snapshot
            except Exception as exc:
                logger.debug("[PANTHEON] Codex model catalog refresh skipped: %s", exc)
                return cached

    async def _fetch_live(self, config: CodexOAuthConfig, state: AuthState) -> CatalogSnapshot:
        client = get_http_client()
        response = await client.get(
            config.catalog_url,
            params={"client_version": config.client_version},
            headers=_oauth_headers(state),
            timeout=config.timeout_seconds,
        )
        if response.status_code == 401:
            state = await token_manager.access_state(force_refresh=True)
            response = await client.get(
                config.catalog_url,
                params={"client_version": config.client_version},
                headers=_oauth_headers(state),
                timeout=config.timeout_seconds,
            )
        if response.status_code >= 400:
            raise CodexOAuthHTTPError(response.status_code, response.text[:500], _retry_after(response))
        payload = response.json()
        snapshot = _snapshot_from_payload(payload)
        if snapshot is None:
            raise CodexOAuthUnavailable("Codex model catalog returned no served slugs")
        return snapshot

    async def reconcile(self, requested: str | None, *, force_refresh: bool = False) -> str:
        snapshot = await self.refresh(force=force_refresh)
        if snapshot is None:
            snapshot = self.load_cached()
        config = get_config()
        slugs = list(snapshot.slugs if snapshot else ())
        if not slugs:
            return config.default_model
        requested_variants = _model_variants(requested)
        for candidate in requested_variants:
            if candidate in slugs:
                return candidate
        for candidate in requested_variants:
            for slug in slugs:
                if slug.startswith(candidate + "-") or slug.startswith(candidate):
                    return slug
        if any("codex" in candidate for candidate in requested_variants):
            if config.default_model in slugs:
                return config.default_model
            for slug in slugs:
                if "codex" in slug:
                    return slug
        if config.default_model in slugs:
            return config.default_model
        return slugs[0]


model_catalog = ModelCatalog()


def _model_variants(requested: str | None) -> list[str]:
    raw = str(requested or "").strip()
    if not raw:
        return []
    out: list[str] = []
    for value in (raw, raw.lower()):
        if value and value not in out:
            out.append(value)
        parts = value.split("/")
        if len(parts) >= 2:
            last = parts[-1]
            if last and last not in out:
                out.append(last)
            tail = "/".join(parts[-2:])
            if tail and tail not in out:
                out.append(tail)
    return out


def _oauth_headers(state: AuthState, *, stream: bool = False) -> dict[str, str]:
    if not state.access_token:
        raise CodexOAuthUnavailable("Codex OAuth access token missing")
    headers = {
        "Authorization": f"Bearer {state.access_token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
    }
    if state.account_id:
        headers["ChatGPT-Account-Id"] = state.account_id
    return headers


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _model_not_supported(status_code: int, text: str) -> bool:
    if status_code != 400:
        return False
    lowered = text.lower()
    return "model not supported" in lowered or "unsupported model" in lowered or "not supported for model" in lowered


def _codex_decision(source: RouteDecision, slug: str) -> RouteDecision:
    return RouteDecision(
        alias=source.alias,
        provider="codex-oauth",
        model_id=slug,
        route_type="fallback",
        reason="codex-oauth-fallback",
        model={
            "id": slug,
            "model_id": slug,
            "provider": "codex-oauth",
            "registry_provider": "codex-oauth",
            "usage_tier": "fallback",
        },
        task_type=source.task_type,
        candidates=[slug],
    )


async def _request_json(decision: RouteDecision, body: dict[str, Any], slug: str, *, refreshed: bool = False, reconciled: bool = False) -> dict[str, Any]:
    from mnemos.domain.pantheon import gateway

    config = get_config()
    state = await token_manager.access_state(force_refresh=refreshed)
    codex_decision = _codex_decision(decision, slug)
    uses_responses = gateway.model_uses_responses_api(slug)
    payload = gateway._provider_payload(codex_decision, body, stream=False)  # noqa: SLF001
    url = f"{config.gateway_base_url}/responses" if uses_responses else f"{config.gateway_base_url}/chat/completions"
    response = await get_http_client().post(
        url,
        json=payload,
        headers=_oauth_headers(state),
        timeout=config.timeout_seconds,
    )
    text = response.text[:500]
    if response.status_code == 401 and not refreshed:
        return await _request_json(decision, body, slug, refreshed=True, reconciled=reconciled)
    if _model_not_supported(response.status_code, text) and not reconciled:
        new_slug = await model_catalog.reconcile(slug, force_refresh=True)
        if new_slug != slug:
            return await _request_json(decision, body, new_slug, refreshed=refreshed, reconciled=True)
    if response.status_code >= 400:
        raise CodexOAuthHTTPError(response.status_code, text, _retry_after(response))
    data = response.json()
    if uses_responses:
        data = gateway._responses_to_chat_completion(data, codex_decision)  # noqa: SLF001
    data.setdefault("model", slug)
    return data


async def forward_chat_completion(decision: RouteDecision, body: dict[str, Any]) -> dict[str, Any]:
    config = get_config()
    if not config.enabled:
        raise CodexOAuthUnavailable("Codex OAuth fallback disabled")
    if read_auth_state(config) is None:
        raise CodexOAuthUnavailable("Codex OAuth auth.json absent")
    requested = decision.model_id or body.get("model") or decision.alias
    slug = await model_catalog.reconcile(str(requested) if requested else None)
    return await _request_json(decision, body, slug)


async def _open_stream(decision: RouteDecision, body: dict[str, Any], slug: str, *, refreshed: bool = False, reconciled: bool = False) -> CodexOpenStream:
    from mnemos.domain.pantheon import gateway

    config = get_config()
    state = await token_manager.access_state(force_refresh=refreshed)
    codex_decision = _codex_decision(decision, slug)
    uses_responses = gateway.model_uses_responses_api(slug)
    payload = gateway._provider_payload(codex_decision, body, stream=True)  # noqa: SLF001
    url = f"{config.gateway_base_url}/responses" if uses_responses else f"{config.gateway_base_url}/chat/completions"
    manager = get_http_client().stream(
        "POST",
        url,
        json=payload,
        headers=_oauth_headers(state, stream=True),
        timeout=None,
    )
    response = await manager.__aenter__()
    if response.status_code >= 400:
        body_bytes = await response.aread()
        await manager.__aexit__(None, None, None)
        text = body_bytes[:500].decode("utf-8", "replace")
        if response.status_code == 401 and not refreshed:
            return await _open_stream(decision, body, slug, refreshed=True, reconciled=reconciled)
        if _model_not_supported(response.status_code, text) and not reconciled:
            new_slug = await model_catalog.reconcile(slug, force_refresh=True)
            if new_slug != slug:
                return await _open_stream(decision, body, new_slug, refreshed=refreshed, reconciled=True)
        raise CodexOAuthHTTPError(response.status_code, text, _retry_after(response))
    return CodexOpenStream(decision=codex_decision, manager=manager, response=response)


async def stream_chat_completion(decision: RouteDecision, body: dict[str, Any]) -> AsyncIterator[bytes]:
    from mnemos.domain.pantheon.gateway import _yield_openai_chat_stream

    config = get_config()
    if not config.enabled:
        raise CodexOAuthUnavailable("Codex OAuth fallback disabled")
    if read_auth_state(config) is None:
        raise CodexOAuthUnavailable("Codex OAuth auth.json absent")
    requested = decision.model_id or body.get("model") or decision.alias
    slug = await model_catalog.reconcile(str(requested) if requested else None)
    opened = await _open_stream(decision, body, slug)
    async for chunk in _yield_openai_chat_stream(opened):
        yield chunk


async def refresh_catalog_once(*, force: bool = True) -> CatalogSnapshot | None:
    return await model_catalog.refresh(force=force)


async def _background_refresh_loop() -> None:
    config = get_config()
    while True:
        try:
            await refresh_catalog_once(force=False)
        except Exception as exc:
            logger.debug("[PANTHEON] Codex OAuth catalog background refresh failed: %s", exc)
        await asyncio.sleep(config.catalog_refresh_seconds)


def start_background_refresh() -> None:
    global _BACKGROUND_TASK
    config = get_config()
    if not config.enabled:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    if _BACKGROUND_TASK is None or _BACKGROUND_TASK.done():
        _BACKGROUND_TASK = loop.create_task(_background_refresh_loop())


async def stop_background_refresh() -> None:
    global _BACKGROUND_TASK
    if _BACKGROUND_TASK is None:
        return
    task = _BACKGROUND_TASK
    _BACKGROUND_TASK = None
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
