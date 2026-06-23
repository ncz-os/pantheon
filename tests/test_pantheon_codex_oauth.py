from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from mnemos.domain.pantheon import codex_oauth
from mnemos.domain.pantheon.router import RouteDecision


class _Resp:
    def __init__(self, status_code: int, payload: dict, text: str | None = None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload)
        self.headers = {}

    def json(self):
        return self._payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _decision(model_id: str = "openai/openai/gpt-5.3-codex") -> RouteDecision:
    return RouteDecision(alias=model_id, provider="eih", model_id=model_id, route_type="single", reason="r")


def _auth_payload(*, access: str = "access-old", refresh: str = "refresh-old", last_refresh: str = "2099-01-01T00:00:00Z") -> dict:
    return {
        "auth_mode": "chatgpt",
        "tokens": {
            "access_token": access,
            "refresh_token": refresh,
            "account_id": "acct_1",
        },
        "last_refresh": last_refresh,
    }


def test_catalog_reconcile_uses_cached_served_codex_slug(monkeypatch, tmp_path):
    cache = tmp_path / "models_cache.json"
    _write_json(
        cache,
        {
            "client_version": "0.131.0",
            "models": [
                {"slug": "gpt-5.5", "visibility": "list"},
                {"slug": "gpt-5.3-codex-spark", "visibility": "list", "supported_in_api": False},
            ],
        },
    )
    monkeypatch.setenv("PANTHEON_CODEX_OAUTH_MODELS_CACHE", str(cache))
    monkeypatch.setenv("PANTHEON_CODEX_OAUTH_SLUG_CACHE", str(tmp_path / "slug_cache.json"))
    monkeypatch.setenv("PANTHEON_CODEX_OAUTH_FALLBACK_ENABLED", "false")

    catalog = codex_oauth.ModelCatalog()
    slug = asyncio.run(catalog.reconcile("openai/openai/gpt-5.3-codex"))

    assert slug == "gpt-5.3-codex-spark"


def test_token_refresh_is_single_flight_and_uses_distributed_lock(monkeypatch, tmp_path):
    auth_path = tmp_path / "auth.json"
    _write_json(auth_path, _auth_payload(last_refresh="2000-01-01T00:00:00Z"))

    class _LockStore:
        def __init__(self):
            self.acquires = 0
            self.releases = 0

        async def atry_acquire_lock(self, name, owner, *, ttl_seconds=None):
            assert "codex-oauth" in name
            assert owner
            assert ttl_seconds is not None
            self.acquires += 1
            return True

        async def arelease_lock(self, name, owner):
            assert "codex-oauth" in name
            assert owner
            self.releases += 1

    class _Client:
        def __init__(self):
            self.posts = 0

        async def post(self, url, **kwargs):
            self.posts += 1
            assert url == "https://refresh.example/token"
            assert kwargs["data"]["refresh_token"] == "refresh-old"
            await asyncio.sleep(0)
            return _Resp(
                200,
                {
                    "access_token": "access-new",
                    "refresh_token": "refresh-new",
                    "account_id": "acct_1",
                },
            )

    lock_store = _LockStore()
    client = _Client()
    monkeypatch.setenv("PANTHEON_CODEX_OAUTH_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("PANTHEON_CODEX_OAUTH_AUTH_JSON", str(auth_path))
    monkeypatch.setenv("PANTHEON_CODEX_OAUTH_REFRESH_URL", "https://refresh.example/token")
    monkeypatch.setenv("PANTHEON_CODEX_OAUTH_TOKEN_REFRESH_SECONDS", "1")
    monkeypatch.setattr(codex_oauth, "_distributed_lock_store", lambda: lock_store)
    monkeypatch.setattr(codex_oauth, "get_http_client", lambda: client)

    async def _run():
        manager = codex_oauth.TokenManager()
        return await asyncio.gather(manager.access_state(), manager.access_state())

    states = asyncio.run(_run())
    persisted = json.loads(auth_path.read_text(encoding="utf-8"))

    assert [state.access_token for state in states] == ["access-new", "access-new"]
    assert persisted["tokens"]["refresh_token"] == "refresh-new"
    assert client.posts == 1
    assert lock_store.acquires == 1
    assert lock_store.releases == 1


def test_stale_token_refresh_fails_closed_without_distributed_lock(monkeypatch, tmp_path):
    auth_path = tmp_path / "auth.json"
    _write_json(auth_path, _auth_payload(last_refresh="2000-01-01T00:00:00Z"))

    class _Client:
        async def post(self, *_args, **_kwargs):
            raise AssertionError("refresh must not run without a distributed lock")

    monkeypatch.setenv("PANTHEON_CODEX_OAUTH_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("PANTHEON_CODEX_OAUTH_AUTH_JSON", str(auth_path))
    monkeypatch.setenv("PANTHEON_CODEX_OAUTH_TOKEN_REFRESH_SECONDS", "1")
    monkeypatch.setattr(codex_oauth, "_distributed_lock_store", lambda: None)
    monkeypatch.setattr(codex_oauth, "get_http_client", lambda: _Client())

    async def _run():
        manager = codex_oauth.TokenManager()
        await manager.access_state()

    with pytest.raises(codex_oauth.CodexOAuthUnavailable, match="distributed refresh lock"):
        asyncio.run(_run())


def test_model_not_supported_refreshes_catalog_and_retries_reconciled_slug(monkeypatch, tmp_path):
    auth_path = tmp_path / "auth.json"
    models_cache = tmp_path / "models_cache.json"
    slug_cache = tmp_path / "slug_cache.json"
    _write_json(auth_path, _auth_payload())
    _write_json(models_cache, {"client_version": "0.131.0", "models": [{"slug": "gpt-5.3-codex"}]})

    class _Client:
        def __init__(self):
            self.posts: list[dict] = []
            self.gets: list[dict] = []

        async def post(self, _url, **kwargs):
            self.posts.append(kwargs["json"])
            if len(self.posts) == 1:
                return _Resp(400, {"error": {"message": "model not supported"}}, "model not supported")
            return _Resp(
                200,
                {
                    "id": "resp_1",
                    "created_at": 1,
                    "model": "gpt-5.3-codex-spark",
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "ok"}],
                        }
                    ],
                    "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                },
            )

        async def get(self, _url, **kwargs):
            self.gets.append(kwargs)
            return _Resp(
                200,
                {
                    "client_version": "0.131.0",
                    "models": [{"slug": "gpt-5.3-codex-spark", "visibility": "list"}],
                },
            )

    client = _Client()
    monkeypatch.setenv("PANTHEON_CODEX_OAUTH_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("PANTHEON_CODEX_OAUTH_AUTH_JSON", str(auth_path))
    monkeypatch.setenv("PANTHEON_CODEX_OAUTH_MODELS_CACHE", str(models_cache))
    monkeypatch.setenv("PANTHEON_CODEX_OAUTH_SLUG_CACHE", str(slug_cache))
    monkeypatch.setenv("PANTHEON_CODEX_OAUTH_BASE_URL", "http://zc-gateway.test/v1")
    monkeypatch.setenv("PANTHEON_CODEX_OAUTH_CATALOG_URL", "http://codex.test/models")
    monkeypatch.setenv("PANTHEON_CODEX_OAUTH_CLIENT_VERSION", "0.131.0")
    monkeypatch.setenv("PANTHEON_CODEX_OAUTH_CATALOG_REFRESH_SECONDS", "999999999")
    monkeypatch.setattr(codex_oauth, "get_http_client", lambda: client)

    async def _run():
        monkeypatch.setattr(codex_oauth, "token_manager", codex_oauth.TokenManager())
        monkeypatch.setattr(codex_oauth, "model_catalog", codex_oauth.ModelCatalog())
        return await codex_oauth.forward_chat_completion(
            _decision(),
            {"messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 16},
        )

    out = asyncio.run(_run())

    assert out["model"] == "gpt-5.3-codex-spark"
    assert out["choices"][0]["message"]["content"] == "ok"
    assert [payload["model"] for payload in client.posts] == ["gpt-5.3-codex", "gpt-5.3-codex-spark"]
    assert client.gets[0]["params"] == {"client_version": "0.131.0"}
    assert json.loads(slug_cache.read_text(encoding="utf-8"))["models"][0]["slug"] == "gpt-5.3-codex-spark"
