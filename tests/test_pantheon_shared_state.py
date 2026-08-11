from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from mnemos.domain.pantheon.caps import ConsultationCapBucket
from mnemos.domain.pantheon.cooldown import CooldownManager, InMemoryCooldownStore
from mnemos.domain.pantheon.cooldown_nats import (
    NatsJetStreamCooldownStore,
    _cap_key,
)
from mnemos.domain.pantheon.runtime import RouterRuntime
from mnemos.domain.pantheon.shared_state import SharedStateUnavailable


class _NotFoundError(Exception):
    pass


class _WrongLastError(Exception):
    pass


class _Entry:
    def __init__(self, value: bytes, revision: int):
        self.value = value
        self.revision = revision


class _AtomicKv:
    def __init__(self) -> None:
        self.values: dict[str, tuple[bytes, int]] = {}
        self.revision = 0

    async def get(self, key: str):
        await asyncio.sleep(0)
        try:
            value, revision = self.values[key]
        except KeyError:
            raise _NotFoundError("not found")
        return _Entry(value, revision)

    async def create(self, key: str, value: bytes):
        await asyncio.sleep(0)
        if key in self.values:
            raise _WrongLastError("wrong last sequence")
        return await self.put(key, value)

    async def put(self, key: str, value: bytes):
        self.revision += 1
        self.values[key] = (value, self.revision)
        return self.revision

    async def update(self, key: str, value: bytes, *, last: int):
        await asyncio.sleep(0)
        if key not in self.values or self.values[key][1] != last:
            raise _WrongLastError("wrong last sequence")
        return await self.put(key, value)


class _BrokenKv:
    async def get(self, _key: str):
        raise RuntimeError("jetstream unavailable")

    async def put(self, _key: str, _value: bytes):
        raise RuntimeError("jetstream unavailable")


def test_consultation_cap_is_atomic_across_worker_stores() -> None:
    kv = _AtomicKv()
    first = NatsJetStreamCooldownStore(js=kv, sync_timeout=2.0)
    second = NatsJetStreamCooldownStore(js=kv, sync_timeout=2.0)

    async def consume() -> list[tuple[bool, int]]:
        calls = [
            (first if index % 2 else second).acheck_and_increment_consultation_cap(
                user_id="alice", session_id="trusted-session", cap=50
            )
            for index in range(100)
        ]
        return await asyncio.gather(*calls)

    try:
        results = asyncio.run(consume())
        assert sum(allowed for allowed, _used in results) == 50
        assert max(used for _allowed, used in results) == 50
        assert _cap_key("alice", "trusted-session") in kv.values
    finally:
        first.close()
        second.close()


def test_shared_state_failure_is_closed_by_default() -> None:
    store = NatsJetStreamCooldownStore(
        js=_BrokenKv(), sync_timeout=0.1, allow_local_fallback=False
    )
    try:
        with pytest.raises(SharedStateUnavailable, match="configure MNEMOS_NATS_URL"):
            asyncio.run(store.aready())
        with pytest.raises(SharedStateUnavailable, match="cooldown read"):
            asyncio.run(store.aget_cooled_until("tenant", "deployment"))
    finally:
        store.close()


def test_explicit_single_worker_setting_allows_local_cooldown_fallback() -> None:
    store = NatsJetStreamCooldownStore(
        js=_BrokenKv(),
        sync_timeout=0.1,
        allow_local_fallback=True,
    )
    store._l1.set_cooldown("tenant", "deployment", 123.0)
    try:
        asyncio.run(store.aready())
        assert asyncio.run(store.aget_cooled_until("tenant", "deployment")) == 123.0
    finally:
        store.close()


def test_shared_cap_keys_do_not_expose_principals() -> None:
    key = _cap_key("alice@example.com", "session-secret")
    for plaintext in ("alice", "example", "session", "secret"):
        assert plaintext not in key


def test_local_cap_fails_closed_for_multi_worker_without_nats(monkeypatch) -> None:
    from mnemos.domain.pantheon import gateway
    from mnemos.core.config import _reset_settings_for_tests

    monkeypatch.delenv("MNEMOS_PANTHEON_ALLOW_LOCAL_SHARED_STATE", raising=False)
    monkeypatch.setenv("MNEMOS_WORKERS", "2")
    _reset_settings_for_tests()
    gateway.set_runtime(
        RouterRuntime(CooldownManager(InMemoryCooldownStore()), clock=lambda: 0.0)
    )
    bucket = ConsultationCapBucket()
    try:
        with pytest.raises(SharedStateUnavailable, match="development/single-worker"):
            asyncio.run(bucket.acheck_and_increment(user_id="alice", session_id="s", cap=1))
    finally:
        gateway.set_runtime(None)
        _reset_settings_for_tests()


@pytest.mark.parametrize("profile", ["server", "edge", "dev", "personal"])
def test_default_single_worker_profiles_allow_local_state_without_nats(monkeypatch, profile) -> None:
    from mnemos.core.config import _reset_settings_for_tests
    from mnemos.domain.pantheon import gateway

    monkeypatch.delenv("MNEMOS_PANTHEON_ALLOW_LOCAL_SHARED_STATE", raising=False)
    monkeypatch.delenv("MNEMOS_NATS_URL", raising=False)
    monkeypatch.delenv("MNEMOS_WORKERS", raising=False)
    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
    monkeypatch.setenv("MNEMOS_PROFILE", profile)
    _reset_settings_for_tests()
    gateway.set_runtime(None)
    try:
        assert isinstance(gateway.get_runtime().cooldown._store, InMemoryCooldownStore)
        asyncio.run(gateway.require_shared_state())
    finally:
        gateway.set_runtime(None)
        _reset_settings_for_tests()


@pytest.mark.parametrize("worker_env", [("MNEMOS_WORKERS", "2"), ("WEB_CONCURRENCY", "4")])
def test_explicit_multi_worker_without_nats_does_not_allow_local_state(monkeypatch, worker_env) -> None:
    from mnemos.core.config import _reset_settings_for_tests
    from mnemos.domain.pantheon.shared_state import allow_local_shared_state

    monkeypatch.delenv("MNEMOS_PANTHEON_ALLOW_LOCAL_SHARED_STATE", raising=False)
    monkeypatch.delenv("MNEMOS_NATS_URL", raising=False)
    monkeypatch.setenv(*worker_env)
    _reset_settings_for_tests()
    try:
        assert allow_local_shared_state() is False
    finally:
        _reset_settings_for_tests()


def test_configured_nats_remains_authoritative_for_single_worker(monkeypatch) -> None:
    from mnemos.core.config import _reset_settings_for_tests
    from mnemos.domain.pantheon.shared_state import allow_local_shared_state

    monkeypatch.delenv("MNEMOS_PANTHEON_ALLOW_LOCAL_SHARED_STATE", raising=False)
    monkeypatch.setenv("MNEMOS_NATS_URL", "nats://127.0.0.1:4222")
    monkeypatch.setenv("MNEMOS_WORKERS", "1")
    _reset_settings_for_tests()
    try:
        assert allow_local_shared_state() is False
    finally:
        _reset_settings_for_tests()


def test_local_cap_opt_out_preserves_single_worker_behavior(monkeypatch) -> None:
    from mnemos.domain.pantheon import gateway

    monkeypatch.setenv("MNEMOS_PANTHEON_ALLOW_LOCAL_SHARED_STATE", "true")
    gateway.set_runtime(
        RouterRuntime(CooldownManager(InMemoryCooldownStore()), clock=lambda: 0.0)
    )
    bucket = ConsultationCapBucket()
    try:
        first = asyncio.run(bucket.acheck_and_increment(user_id="alice", session_id="s", cap=1))
        second = asyncio.run(bucket.acheck_and_increment(user_id="alice", session_id="s", cap=1))
        assert first.allowed is True
        assert second.allowed is False
    finally:
        gateway.set_runtime(None)


def test_generation_route_returns_503_when_shared_state_is_down(monkeypatch) -> None:
    from mnemos.api.routes import pantheon as routes
    from mnemos.domain.pantheon import gateway
    from mnemos.domain.pantheon.aliases import PantheonRoutingError

    async def _unavailable() -> None:
        raise SharedStateUnavailable("shared state down; configure MNEMOS_NATS_URL")

    monkeypatch.setattr(gateway, "require_shared_state", _unavailable)
    monkeypatch.setattr(
        routes,
        "_pantheon_imports",
        lambda: (
            None,
            gateway,
            SimpleNamespace(),
            PantheonRoutingError,
            SimpleNamespace(),
            lambda **_kwargs: ({}, {}),
            lambda *_args: None,
        ),
    )
    request = SimpleNamespace(
        headers={}, query_params={}, state=SimpleNamespace(), client=SimpleNamespace(host="test")
    )
    user = SimpleNamespace(user_id="alice", namespace="default")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            routes._chat_completions_impl(
                request,
                {"model": "gpt", "messages": [{"role": "user", "content": "hi"}]},
                user,
            )
        )

    assert exc_info.value.status_code == 503
    assert "MNEMOS_NATS_URL" in str(exc_info.value.detail)
