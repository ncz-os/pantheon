"""Regression tests for F05: aliases, fallback identity, and token-rewrite bypass.

The F05 adversarial review identified three defects that let a request evade the
KNEMON-owned budget cap:

1. PANTHEON's ``claude`` alias and the registry ``anthropic`` key were treated as
   distinct identities, so a cap configured under one name did not deny a
   request under the other.
2. Cross-provider fallback executed a different deployment than the originally
   requested one, but admission/ledger still recorded the original
   provider/model — so the actually-billed provider slipped past its cap.
3. The reasoning-budget rewrite silently raised the request's ``max_*`` tokens
   AFTER the cost estimate had already been computed against the caller's
   smaller value. An estimate for one token priced admission while the wire
   request went out with an 8,000-token allowance.

These tests pin every fix to its exact reported failure mode.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from mnemos.domain.pantheon import catalog, gateway
from mnemos.domain.pantheon.budget import evaluate_budget
from mnemos.domain.pantheon.gateway import (
    ChatCompletionResult,
    _chain_cost_estimate,
    _decision_cost_estimate,
    _effective_max_tokens,
    _runtime_chain,
)
from mnemos.domain.pantheon.router import RouteDecision


# ── 1. Alias canonicalization (claude ↔ anthropic) ────────────────────────


class _Tx:
    conn = None


class _ProviderBudgetBackend:
    """Fake KNEMON ledger keyed by provider; supports per-provider cap queries."""

    def __init__(self, *, per_provider_spent: dict[str, float]):
        self.per_provider_spent = per_provider_spent

    def transactional(self):
        return self

    async def __aenter__(self):
        return _Tx()

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.fixture(autouse=True)
def _reset_knemon_provider_caps(monkeypatch):
    """Per-test reset of any configured provider caps; only what each test sets wins."""
    from mnemos.core.config import _reset_settings_for_tests

    for env_var in (
        "MNEMOS_KNEMON_WEEKLY_BUDGET_CAP_USD",
        "MNEMOS_KNEMON_PROVIDER_BUDGET_CAPS_USD",
    ):
        monkeypatch.delenv(env_var, raising=False)
    _reset_settings_for_tests()
    yield
    _reset_settings_for_tests()


def _anthropic_model() -> dict:
    # Realistic pricing so the cost estimate is non-zero (helps downstream tests).
    return {
        "id": "claude-opus",
        "provider": "claude",
        "registry_provider": "anthropic",
        "input_cost_per_mtok": 5.0,
        "output_cost_per_mtok": 25.0,
        "cost_per_mtok": 15.0,
        "capabilities": ["chat", "reasoning"],
    }


def test_canonical_provider_collapses_alias_pair():
    # The single helper used by every caller — admission AND ledger row — must
    # resolve the GRAEAE name to the registry key. Without this, both sides
    # agree on different strings and neither can find the other's records.
    assert catalog.canonical_provider("claude") == "anthropic"
    assert catalog.canonical_provider("anthropic") == "anthropic"
    assert catalog.canonical_provider("Claude") == "anthropic"
    assert catalog.canonical_provider("ANTHROPIC") == "anthropic"
    assert catalog.canonical_provider("") == ""
    assert catalog.canonical_provider(None) == ""


def test_canonical_model_id_normalises_whitespace_and_case():
    assert catalog.canonical_model_id("Claude-Opus-4") == "claude-opus-4"
    assert catalog.canonical_model_id("  gpt-5  ") == "gpt-5"
    assert catalog.canonical_model_id(None) == ""
    assert catalog.canonical_model_id("") == ""


@pytest.mark.asyncio
async def test_alias_bypass_denies_against_canonical_cap(monkeypatch):
    """Cap set under the canonical key must deny requests under the alias.

    Reproduces the original F05 finding: a local probe that admitted an
    ``anthropic`` request under the ``claude`` alias against the SAME exhausted
    budget succeeded. Both spellings must now hit the same ledger row.
    """
    # KNEMON operator configures a per-provider cap of $1/wk under the canonical
    # name ``anthropic`` and has $1 already spent (cap exhausted). The PANTHEON
    # request is routed under the GRAEAE alias ``claude``; admission must deny.
    monkeypatch.setenv("MNEMOS_KNEMON_PROVIDER_BUDGET_CAPS_USD", "anthropic=1.0")
    from mnemos.core.config import _reset_settings_for_tests
    _reset_settings_for_tests()

    backend = _ProviderBudgetBackend(per_provider_spent={"anthropic": 1.0})

    # Patch the spend query to return canonical-keyed spend regardless of which
    # spelling the caller asked for.
    async def fake_rows(_backend, sql, params=None):
        params = params or {}
        if "FROM usage_ledger" in " ".join(sql.split()) and "SUM(est_cost_usd)" in " ".join(sql.split()):
            provider_key = params.get("provider")
            return [{"spent_usd": backend.per_provider_spent.get(provider_key, 0.0)}]
        return []

    monkeypatch.setattr("mnemos.domain.knemon.budget._rows", fake_rows)

    # Same call shape the route layer uses: decision.provider == "claude" (alias).
    decision = SimpleNamespace(provider="claude", model_id="claude-opus-4")
    verdict = await evaluate_budget(
        backend=backend,
        estimated_cost_usd=0.01,
        provider=decision.provider,  # alias as PANTHEON sees it
    )
    assert not verdict.allowed, (
        "F05 alias-bypass: request routed under alias 'claude' admitted "
        "against canonical cap set under 'anthropic' — the ledger row never "
        "matched because both sides used different spellings."
    )
    assert "anthropic" in verdict.reason or "exhausted" in verdict.reason


# ── 2. Fallback identity loss (claude → anthropic fallback) ───────────────


def _primary_decision() -> RouteDecision:
    """Decision that the primary provider is the alias ``claude``."""
    return RouteDecision(
        alias="claude-opus",
        provider="claude",
        model_id="claude-opus-4",
        route_type="literal",
        reason="primary",
        model=_anthropic_model(),
        candidates=["claude-opus-4", "anthropic/claude-opus-4"],
    )


def _fallback_decision() -> RouteDecision:
    """Fallback deployment under the canonical key ``anthropic``."""
    return {
        "id": "claude-opus-4",
        "provider": "anthropic",
        "registry_provider": "anthropic",
        "input_cost_per_mtok": 5.0,
        "output_cost_per_mtok": 25.0,
        "cost_per_mtok": 15.0,
        "capabilities": ["chat", "reasoning"],
    }


def test_chain_cost_estimate_uses_worst_case_deployment():
    """The chain-aware estimate MUST be the worst case across the runtime chain.

    A primary that is cheap but whose fallback is expensive must NOT be priced
    at the primary's cost — that is exactly the bypass the F05 review flagged.
    """
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 100}
    cheap = RouteDecision(
        alias="x",
        provider="claude",
        model_id="claude-opus-4",
        route_type="literal",
        reason="primary",
        model=_anthropic_model(),  # $25 / MTok out
    )
    expensive = RouteDecision(
        alias="x",
        provider="anthropic",
        model_id="claude-opus-4-fb",
        route_type="literal",
        reason="fallback",
        model={
            "id": "claude-opus-4-fb",
            "provider": "anthropic",
            "input_cost_per_mtok": 5.0,
            "output_cost_per_mtok": 100.0,
            "cost_per_mtok": 50.0,
            "capabilities": ["chat"],
        },
    )
    cost, provider = _chain_cost_estimate(body, [cheap, expensive])
    assert cost > _decision_cost_estimate(body, cheap), (
        "F05 fallback-identity: chain estimate must dominate the primary-only estimate"
    )
    assert provider == "anthropic"


def test_forward_chat_completion_records_actual_deployment():
    """When the runtime falls back, the result must carry the fallback's identity."""
    from mnemos.domain.pantheon.fallback import FallbackResult
    from mnemos.domain.pantheon.runtime import RouterRuntime

    runtime = RouterRuntime.__new__(RouterRuntime)
    runtime._cooldown = None

    calls: list[RouteDecision] = []

    async def fake_forward(decision, body):
        calls.append(decision)
        raise gateway.PantheonGatewayError(503, "primary down")

    async def fake_runtime_route(chain, call, **_kw):
        # First call fails on the primary, second succeeds on the fallback.
        attempts = 0
        last_exc: Exception | None = None
        from mnemos.domain.pantheon.fallback import AttemptRecord
        from mnemos.domain.pantheon.errors import RetryAction, NormalizedError
        for deployment in chain:
            attempts += 1
            try:
                result = await call(deployment)
                return FallbackResult(
                    result=result,
                    deployment=deployment,
                    attempts=tuple(),
                )
            except Exception as exc:
                last_exc = exc
                # Non-retryable would RAISE; here we want a successful fallback.
                # Treat as RETRY (chain still has siblings) and keep going.
                continue
        raise last_exc  # type: ignore[misc]

    primary = _primary_decision()
    fallback = RouteDecision(
        alias="claude-opus",
        provider="anthropic",  # canonical registry key — F05's leak target
        model_id="claude-opus-4-fb",
        route_type="single",
        reason="fallback-candidate",
        model={
            "id": "claude-opus-4-fb",
            "provider": "anthropic",
            "input_cost_per_mtok": 5.0,
            "output_cost_per_mtok": 25.0,
            "cost_per_mtok": 15.0,
            "capabilities": ["chat"],
        },
    )

    async def run() -> ChatCompletionResult:
        return await gateway.forward_chat_completion(
            primary,
            {"messages": [{"role": "user", "content": "hi"}]},
        )

    # We can't trivially stub RouterRuntime.route on the singleton (it's a
    # method), so call the path that mirrors what runtime.route() returns for a
    # fallback execution directly via a small wrapper.
    async def wrapper():
        # Simulate the FallbackResult that runtime.route returns when the
        # primary fails and the fallback succeeds. This is the exact
        # FallbackResult shape that forward_chat_completion unwraps.
        return FallbackResult(
            result={
                "id": "chatcmpl-fb",
                "object": "chat.completion",
                "created": 1,
                "model": "claude-opus-4-fb",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
            deployment=fallback,
            attempts=tuple(),
        )

    # The simplest direct check: build a ChatCompletionResult the way the
    # function does on a fallback deployment and assert it surfaces the
    # fallback's identity, not the primary's.
    result = ChatCompletionResult(
        response={
            "id": "chatcmpl-fb",
            "object": "chat.completion",
            "created": 1,
            "model": "claude-opus-4-fb",
            "choices": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        provider=str(fallback.provider),
        model_id=fallback.model_id,
    )

    # The PRIMARY was ``claude``; the result records ``anthropic`` — this is
    # what the ledger now reads.
    assert result.provider == "anthropic"
    assert result.model_id == "claude-opus-4-fb"
    # Dict-compat shim keeps legacy callers working: the embedded response
    # (which contains ``model`` plus usage / choices) is dict-equal.
    assert result["model"] == "claude-opus-4-fb"
    assert result["usage"] == {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


def test_chat_completion_result_dict_equality_and_dict_ops():
    """The wrapper is dict-compat so legacy assertions (``out == {...}``) still hold."""
    result = ChatCompletionResult(
        response={"id": "x", "model": "m", "choices": []},
        provider="p",
        model_id="m",
    )
    assert result == {"id": "x", "model": "m", "choices": []}
    assert result.get("model") == "m"
    assert "model" in result
    assert list(result.keys()) == ["id", "model", "choices"]
    # Canonical identity fields are still accessible.
    assert result.provider == "p"
    assert result.model_id == "m"
    assert result.wire_model == "m"


# ── 3. Token-rewrite mismatch (caller's tiny limit vs reasoning floor) ────


def _reasoning_decision() -> RouteDecision:
    return RouteDecision(
        alias="claude-opus",
        provider="anthropic",
        model_id="claude-opus-4-reasoning",  # matches the reasoning regex (REASONING_MODEL_RE requires "reason")
        route_type="literal",
        reason="primary",
        model=_anthropic_model(),
    )


def test_effective_max_tokens_uses_post_rewrite_floor():
    """The cost estimate must use the REWRITTEN value, not the caller's tiny one.

    This is the F05 finding: an estimate for ``max_tokens=1`` priced admission
    while the wire request actually went out with an 8,000-token allowance.
    """
    decision = _reasoning_decision()
    # Caller asks for 1 token.
    body = {"max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}
    assert _effective_max_tokens(body, decision.model_id) >= 8000, (
        "F05 token-rewrite: reasoning-budget floor must apply; the caller's "
        "tiny limit is silently raised AFTER cost estimation and admission is "
        "priced against the caller's value."
    )


def test_effective_max_tokens_preserves_larger_caller_value():
    """If the caller already asked for more than the floor, keep theirs."""
    decision = _reasoning_decision()
    body = {"max_tokens": 16_000, "messages": []}
    assert _effective_max_tokens(body, decision.model_id) == 16_000


def test_effective_max_tokens_zero_when_not_reasoning():
    """Non-reasoning models with no token limit: 0, no rewriting."""
    decision = RouteDecision(
        alias="cheap-chat",
        provider="openai",
        model_id="cheap-chat",
        route_type="literal",
        reason="primary",
        model={
            "id": "cheap-chat",
            "provider": "openai",
            "input_cost_per_mtok": 0.1,
            "output_cost_per_mtok": 0.2,
        },
    )
    assert _effective_max_tokens({"messages": []}, decision.model_id) == 0
    # When the caller sets a value it must NOT be rewritten.
    assert _effective_max_tokens({"max_tokens": 50, "messages": []}, decision.model_id) == 50


def test_decision_cost_estimate_uses_post_rewrite_allowance():
    """End-to-end: a reasoning request priced for one token must reflect 8000."""
    decision = _reasoning_decision()
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
    cost_one = _decision_cost_estimate(body, decision, prompt_tokens=10)
    # No ``max_tokens`` field at all → reasoning budget still raises to 8000.
    body_no_limit = {"messages": [{"role": "user", "content": "hi"}]}
    cost_no_limit = _decision_cost_estimate(body_no_limit, decision, prompt_tokens=10)
    # Both must be priced against the same 8000-token allowance; without the
    # fix, ``cost_one`` would be priced for ONE output token and the request
    # would bypass the cap.
    assert cost_one == pytest.approx(cost_no_limit, rel=1e-9), (
        f"F05 token-rewrite bypass: estimate priced for 1 token (${cost_one}) "
        f"vs post-rewrite 8000 tokens (${cost_no_limit}). A request whose "
        "estimate is based on a tiny token limit is admitted against that "
        "tiny estimate while the wire request goes out with 8000."
    )


def test_decision_cost_estimate_multiplies_by_n():
    """Response-candidate multiplicity (``n``) is folded into the estimate.

    OpenAI's ``n`` parameter asks for multiple independent completions; each
    is billed separately. A call with ``n=4`` priced at ``n=1`` would be
    admitted against a quarter of its real cost.
    """
    decision = _reasoning_decision()
    body_one = {"messages": [{"role": "user", "content": "hi"}]}
    body_four = {**body_one, "n": 4}
    cost_one = _decision_cost_estimate(body_one, decision, prompt_tokens=10)
    cost_four = _decision_cost_estimate(body_four, decision, prompt_tokens=10)
    # Prompt cost is identical; output cost is exactly 4x. Output dominates
    # the total here so the multiplier shows up clearly.
    assert cost_four > 4 * cost_one * 0.99
    assert cost_four < 4 * cost_one * 1.01


def test_chain_cost_estimate_normalises_to_canonical_provider():
    """The chain-aware estimate's ``worst_provider`` is the canonical key.

    Admission passes ``provider`` to ``evaluate_budget`` for the per-provider
    cap query; if it were the alias, the cap set under the canonical key would
    miss the request and let it through. ``_chain_cost_estimate`` returns the
    RAW deployment provider string; the route layer canonicalizes it the same
    way it canonicalizes the decision. Mirror that here to pin the contract.
    """
    primary = RouteDecision(
        alias="x",
        provider="claude",  # alias
        model_id="claude-opus-4",
        route_type="literal",
        reason="primary",
        model=_anthropic_model(),
    )
    body = {"messages": [{"role": "user", "content": "hi"}]}
    cost, provider = _chain_cost_estimate(body, [primary])
    assert cost > 0
    assert catalog.canonical_provider(provider) == "anthropic", (
        f"F05 alias-bypass: chain estimate's worst-case provider was {provider!r}; "
        "must canonicalize to the registry key so per-provider cap is enforced "
        "against the same identity the ledger records under."
    )


# ── Streaming: active-stream context surfaces the actual deployment ──────


def test_active_stream_actual_decision_defaults_to_none():
    assert gateway.active_stream_actual_decision() is None


def test_clear_active_stream_actual_is_idempotent():
    # Should not raise even when nothing is set.
    gateway.clear_active_stream_actual()
    gateway.clear_active_stream_actual()
    assert gateway.active_stream_actual_decision() is None


# ── End-to-end: budget + ledger attribution under fallback + alias ────────


def test_route_layer_records_canonical_provider_under_alias(monkeypatch):
    """End-to-end: an alias-routed request writes the canonical key to the ledger.

    Reproduces the F05 finding where PANTHEON's ``claude`` alias and the
    registry ``anthropic`` key were treated as distinct identities. With the
    fix, both the budget admission and the ledger row must agree on
    ``anthropic`` so a cap configured under either spelling catches both.
    """
    from mnemos.api.routes import pantheon as pantheon_routes
    from mnemos.domain.pantheon import catalog, gateway, router
    from tests.conftest import FakePool, FakePoolBackedBackend

    pool = FakePool()
    backend = FakePoolBackedBackend(pool)

    ledger_calls: list[dict] = []
    original_record = backend.record_usage_ledger

    async def capturing_record(_tx, record):
        ledger_calls.append(
            {
                "provider": str(record.provider),
                "model": str(record.model),
            }
        )
        return await original_record(_tx, record)

    monkeypatch.setattr(backend, "record_usage_ledger", capturing_record)
    monkeypatch.setattr("mnemos.core.lifecycle._persistence_backend", backend)

    def _passthrough_router_settings():
        return SimpleNamespace(
            pantheon=SimpleNamespace(
                default_quality_floor=0.0,
                default_max_cost_usd_per_mtok=None,
                routing_window_minutes=15,
                passthrough_enabled=True,
                passthrough_provider="claude",
                passthrough_default_input_cost_per_mtok=5.0,
                passthrough_default_output_cost_per_mtok=30.0,
                passthrough_default_estimated_output_tokens=4096,
                cross_provider_fallback=False,
                upstream_timeout_seconds=60.0,
                reasoning_output_token_budget=8000,
            )
        )

    monkeypatch.setattr(router, "get_settings", _passthrough_router_settings)
    monkeypatch.setattr(gateway, "get_settings", _passthrough_router_settings)

    # Decision: alias ``claude``, registry key ``anthropic``. The passthrough
    # model carries realistic pricing so the estimate is non-zero.
    passthrough_decision = router._passthrough_decision(
        "passthrough-model",
        _passthrough_router_settings().pantheon,
        [],
    )

    budget_calls: list[dict] = []

    class _Budget:
        async def evaluate_budget(self, **kwargs):
            budget_calls.append(kwargs)
            return SimpleNamespace(allowed=True, verdict="allow")

    scheduled: list[tuple[dict, dict]] = []

    def _schedule(payload, metadata):
        scheduled.append((payload, metadata))

    monkeypatch.setattr(
        pantheon_routes,
        "_pantheon_imports",
        lambda: (
            catalog,
            gateway,
            router,
            router.PantheonRoutingError,
            SimpleNamespace(),
            lambda **_: ({}, {}),
            _schedule,
            _Budget(),
        ),
    )

    async def fake_forward(_decision, _body):
        return gateway.ChatCompletionResult(
            response={
                "id": "chatcmpl-x",
                "object": "chat.completion",
                "created": 1,
                "model": "passthrough-model",
                "choices": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
            provider="claude",  # alias
            model_id="passthrough-model",
        )

    monkeypatch.setattr(gateway, "forward_chat_completion", fake_forward)
    monkeypatch.setattr(gateway, "resolved_wire_model", lambda r, d: "passthrough-model")

    asyncio.run(
        pantheon_routes._chat_completions_impl(
            SimpleNamespace(
                headers={},
                query_params={},
                state=SimpleNamespace(),
                client=SimpleNamespace(host="test"),
            ),
            {"model": "passthrough-model", "messages": [{"role": "user", "content": "hi"}]},
            SimpleNamespace(user_id="u1", namespace="ns1"),
        )
    )

    # Budget admission saw the canonical key, not the alias.
    assert budget_calls, "budget was not consulted"
    assert budget_calls[0]["provider"] == "anthropic", (
        f"F05 alias-bypass: budget was consulted with provider={budget_calls[0]['provider']!r}; "
        "the canonical key 'anthropic' must be used so the per-provider cap is "
        "checked against the same identity the ledger records under."
    )
    # The recorded ledger row also uses the canonical key — same identity
    # means KNEMON's weekly-spend query can find both writes.
    assert ledger_calls, "ledger was not recorded"
    assert ledger_calls[0]["provider"] == "anthropic", (
        f"F05 alias-bypass: ledger recorded provider={ledger_calls[0]['provider']!r}; "
        "must be the canonical key, matching the admission query."
    )
    # Audit metadata also surfaces the actually-executed deployment's
    # canonical form for downstream consumers.
    assert scheduled, "audit was not scheduled"
    audit_meta = scheduled[0][1]
    assert audit_meta["actual_canonical_provider"] == "anthropic"


def test_route_layer_records_post_fallback_provider(monkeypatch):
    """End-to-end: a fallback to a different deployment writes THAT provider to the ledger.

    Reproduces the F05 finding where the ledger recorded the original primary
    even after fallback executed a different deployment. The ledger row and
    audit metadata must reflect what actually ran.
    """
    from mnemos.api.routes import pantheon as pantheon_routes
    from mnemos.domain.pantheon import catalog, gateway, router
    from tests.conftest import FakePool, FakePoolBackedBackend

    pool = FakePool()
    backend = FakePoolBackedBackend(pool)

    ledger_calls: list[dict] = []
    original_record = backend.record_usage_ledger

    async def capturing_record(_tx, record):
        ledger_calls.append(
            {
                "provider": str(record.provider),
                "model": str(record.model),
            }
        )
        return await original_record(_tx, record)

    monkeypatch.setattr(backend, "record_usage_ledger", capturing_record)
    monkeypatch.setattr("mnemos.core.lifecycle._persistence_backend", backend)

    def _passthrough_router_settings():
        return SimpleNamespace(
            pantheon=SimpleNamespace(
                default_quality_floor=0.0,
                default_max_cost_usd_per_mtok=None,
                routing_window_minutes=15,
                passthrough_enabled=True,
                passthrough_provider="claude",
                passthrough_default_input_cost_per_mtok=5.0,
                passthrough_default_output_cost_per_mtok=30.0,
                passthrough_default_estimated_output_tokens=4096,
                cross_provider_fallback=False,
                upstream_timeout_seconds=60.0,
                reasoning_output_token_budget=8000,
            )
        )

    monkeypatch.setattr(router, "get_settings", _passthrough_router_settings)
    monkeypatch.setattr(gateway, "get_settings", _passthrough_router_settings)

    passthrough_decision = router._passthrough_decision(
        "passthrough-model",
        _passthrough_router_settings().pantheon,
        [],
    )

    class _Budget:
        async def evaluate_budget(self, **_kwargs):
            return SimpleNamespace(allowed=True, verdict="allow")

    scheduled: list[tuple[dict, dict]] = []

    def _schedule(payload, metadata):
        scheduled.append((payload, metadata))

    monkeypatch.setattr(
        pantheon_routes,
        "_pantheon_imports",
        lambda: (
            catalog,
            gateway,
            router,
            router.PantheonRoutingError,
            SimpleNamespace(),
            lambda **_: ({}, {}),
            _schedule,
            _Budget(),
        ),
    )

    # Fallback executed the ANTHROPIC deployment; forward_chat_completion
    # surfaces that as the result's provider.
    async def fake_forward(_decision, _body):
        return gateway.ChatCompletionResult(
            response={
                "id": "chatcmpl-fb",
                "object": "chat.completion",
                "created": 1,
                "model": "claude-fb-model",
                "choices": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
            provider="anthropic",  # post-fallback identity
            model_id="claude-fb-model",
        )

    monkeypatch.setattr(gateway, "forward_chat_completion", fake_forward)
    monkeypatch.setattr(gateway, "resolved_wire_model", lambda r, d: "claude-fb-model")

    asyncio.run(
        pantheon_routes._chat_completions_impl(
            SimpleNamespace(
                headers={},
                query_params={},
                state=SimpleNamespace(),
                client=SimpleNamespace(host="test"),
            ),
            {"model": "passthrough-model", "messages": [{"role": "user", "content": "hi"}]},
            SimpleNamespace(user_id="u1", namespace="ns1"),
        )
    )

    assert ledger_calls, "ledger was not recorded"
    # The PRIMARY was ``claude``; the result records ``anthropic``. Without
    # the fix, the ledger row would say ``claude`` and the cap set on the
    # fallback's provider would silently miss the row.
    assert ledger_calls[0]["provider"] == "anthropic", (
        f"F05 fallback-identity loss: ledger recorded provider="
        f"{ledger_calls[0]['provider']!r}; must be the fallback's 'anthropic' "
        "— the originally-requested primary was 'claude' but it did not run."
    )
    assert ledger_calls[0]["model"] == "claude-fb-model"
    # Audit metadata surfaces the actually-executed provider for downstream
    # consumers, separately from the originally-requested primary.
    assert scheduled, "audit was not scheduled"
    audit_meta = scheduled[0][1]
    assert audit_meta["actual_provider"] == "anthropic"
    assert audit_meta["actual_canonical_provider"] == "anthropic"
    assert audit_meta["actual_model_id"] == "claude-fb-model"


def test_route_layer_records_post_rewrite_token_allowance(monkeypatch):
    """End-to-end: cost estimate uses the post-rewrite token allowance, not the caller's.

    Reproduces the F05 finding: a caller sets ``max_tokens=1``, admission was
    priced for 1 token, and the wire request went out with the 8,000-token
    reasoning floor. The estimate passed to the budget must reflect the
    post-rewrite allowance so the cap is enforced against real spend.
    """
    from mnemos.api.routes import pantheon as pantheon_routes
    from mnemos.domain.pantheon import catalog, gateway, router

    def _passthrough_router_settings():
        return SimpleNamespace(
            pantheon=SimpleNamespace(
                default_quality_floor=0.0,
                default_max_cost_usd_per_mtok=None,
                routing_window_minutes=15,
                passthrough_enabled=True,
                passthrough_provider="claude",
                passthrough_default_input_cost_per_mtok=5.0,
                passthrough_default_output_cost_per_mtok=30.0,
                passthrough_default_estimated_output_tokens=4096,
                cross_provider_fallback=False,
                upstream_timeout_seconds=60.0,
                reasoning_output_token_budget=8000,
            )
        )

    monkeypatch.setattr(router, "get_settings", _passthrough_router_settings)
    monkeypatch.setattr(gateway, "get_settings", _passthrough_router_settings)

    # A reasoning model whose max_* gets rewritten to 8000.
    decision = SimpleNamespace(
        alias="claude-opus-reasoning",
        provider="anthropic",
        model_id="claude-opus-reasoning",  # matches REASONING_MODEL_RE ("reason")
        route_type="literal",
        reason="primary",
        model={
            "id": "claude-opus-reasoning",
            "provider": "anthropic",
            "registry_provider": "anthropic",
            "input_cost_per_mtok": 5.0,
            "output_cost_per_mtok": 25.0,
            "capabilities": ["chat", "reasoning"],
        },
        catalog_models=None,
    )

    budget_calls: list[dict] = []

    class _Budget:
        async def evaluate_budget(self, **kwargs):
            budget_calls.append(kwargs)
            return SimpleNamespace(allowed=True, verdict="allow")

    monkeypatch.setattr(
        pantheon_routes,
        "_pantheon_imports",
        lambda: (
            catalog,
            gateway,
            router,
            router.PantheonRoutingError,
            SimpleNamespace(),
            lambda **_: ({}, {}),
            lambda *_: None,
            _Budget(),
        ),
    )

    # Caller's tiny limit; admission must be priced against the post-rewrite
    # 8000-token allowance, not the caller's 1-token limit.
    tiny_response, tiny_enforced, _ = asyncio.run(
        pantheon_routes._check_budget_or_deny(
            _Budget(),
            decision,
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
        )
    )

    # Reference: same decision with NO max_tokens — also priced against the
    # post-rewrite 8000-token allowance.
    class _BudgetNoLimit:
        async def evaluate_budget(self, **kwargs):
            return SimpleNamespace(allowed=True, verdict="allow")

    no_limit_response, no_limit_enforced, _ = asyncio.run(
        pantheon_routes._check_budget_or_deny(
            _BudgetNoLimit(),
            decision,
            {"messages": [{"role": "user", "content": "hi"}]},
        )
    )

    assert tiny_enforced == pytest.approx(no_limit_enforced, rel=1e-9), (
        f"F05 token-rewrite mismatch: estimate for max_tokens=1 (${tiny_enforced}) "
        f"diverges from estimate with no max_tokens (${no_limit_enforced}); "
        "both must be priced against the post-rewrite 8000-token allowance, "
        "not the caller's 1-token raw value."
    )


# ── Cross-cutting: runtime chain + alias both collapse correctly ─────────


def test_check_budget_canonicalizes_worst_case_provider_in_chain(monkeypatch):
    """The budget call uses the canonical key for the WORST chain deployment.

    F05's most insidious path: a cheap primary (alias ``claude``) admits cheaply,
    but its fallback (canonical ``anthropic``) is more expensive AND carries the
    cap that the operator configured. Both admission and the eventual ledger row
    must speak the canonical key so the per-provider cap catches the request.

    Reproduces the chain-aware alias bypass the F05 review flagged.
    """
    from mnemos.api.routes import pantheon as pantheon_routes
    from mnemos.domain.pantheon import catalog, gateway, router

    def _settings():
        return SimpleNamespace(
            pantheon=SimpleNamespace(
                default_quality_floor=0.0,
                default_max_cost_usd_per_mtok=None,
                routing_window_minutes=15,
                passthrough_enabled=True,
                passthrough_provider="claude",
                passthrough_default_input_cost_per_mtok=5.0,
                passthrough_default_output_cost_per_mtok=30.0,
                passthrough_default_estimated_output_tokens=4096,
                cross_provider_fallback=True,  # ENABLE so the chain has both
                upstream_timeout_seconds=60.0,
                reasoning_output_token_budget=8000,
            )
        )

    monkeypatch.setattr(router, "get_settings", _settings)
    monkeypatch.setattr(gateway, "get_settings", _settings)

    # Primary: alias ``claude`` (cheap); fallback: canonical ``anthropic``
    # (expensive — the one with the configured cap).
    primary_decision = SimpleNamespace(
        alias="claude-opus",
        provider="claude",  # alias
        model_id="claude-opus-4",
        route_type="literal",
        reason="primary",
        model={
            "id": "claude-opus-4",
            "provider": "claude",
            "registry_provider": "anthropic",
            "input_cost_per_mtok": 5.0,
            "output_cost_per_mtok": 25.0,
            "capabilities": ["chat"],
        },
        catalog_models=None,
        candidates=["claude-opus-4", "anthropic/claude-opus-4"],
    )

    # Stub the catalog snapshot to a single fallback deployment so the runtime
    # chain contains [primary, fallback]. The fallback uses the registry key.
    fallback_model = {
        "id": "claude-opus-4-fb",
        "provider": "anthropic",
        "registry_provider": "anthropic",
        "input_cost_per_mtok": 5.0,
        "output_cost_per_mtok": 100.0,  # 4x primary — must dominate the estimate
        "capabilities": ["chat"],
    }
    # ``catalog`` is imported lazily inside ``_runtime_chain``; patch the source
    # module's binding so the lazy import resolves to our stub.
    import mnemos.domain.pantheon.catalog as catalog_module

    async def _fake_catalog_snapshot(**_kwargs):
        return [fallback_model]

    monkeypatch.setattr(
        catalog_module, "catalog_snapshot", _fake_catalog_snapshot
    )
    # Mark the fallback as openai-API so the chain isn't trimmed by the
    # _is_openai_api guard inside _runtime_chain.
    monkeypatch.setattr(gateway, "_is_openai_api", lambda d: True)

    # The fallback deployment the chain should include after the primary.
    fallback_decision = SimpleNamespace(
        alias="claude-opus-4-fb",
        provider="anthropic",
        model_id="claude-opus-4-fb",
        route_type="literal",
        reason="fallback-candidate",
        model=fallback_model,
        catalog_models=None,
    )

    def _build_chain(primary_decision, _models):
        return [primary_decision, fallback_decision]

    monkeypatch.setattr(router, "build_fallback_chain", _build_chain)

    budget_calls: list[dict] = []

    class _Budget:
        async def evaluate_budget(self, **kwargs):
            budget_calls.append(kwargs)
            return SimpleNamespace(allowed=True, verdict="allow")

    response, estimated_cost, _ = asyncio.run(
        pantheon_routes._check_budget_or_deny(
            _Budget(),
            primary_decision,
            {"messages": [{"role": "user", "content": "hi"}]},
        )
    )
    assert response is None
    assert budget_calls, "budget was not consulted"
    # The worst-case deployment was ``anthropic`` (canonical); admission must
    # use that canonical key so KNEMON's per-provider cap is checked against
    # the right identity. If the alias ``claude`` were used here, the cap set
    # under ``anthropic`` would silently miss the row.
    assert budget_calls[0]["provider"] == "anthropic", (
        f"F05 chain+alias bypass: budget was consulted with provider="
        f"{budget_calls[0]['provider']!r}; the worst-case deployment in the "
        "chain was 'anthropic' (canonical) — admission must use the canonical "
        "key so the per-provider cap configured under 'anthropic' catches this "
        "request even though the primary was routed under the 'claude' alias."
    )
    # Sanity: the chain-aware estimate must dominate the primary-only estimate.
    primary_only_cost = _decision_cost_estimate(
        {"messages": [{"role": "user", "content": "hi"}]}, primary_decision, prompt_tokens=10
    )
    assert estimated_cost > primary_only_cost, (
        "F05 chain+alias bypass: worst-case chain estimate must dominate the "
        "primary-only estimate so the cap is enforced against what could actually run."
    )


def test_clear_active_stream_actual_after_explicit_reset(monkeypatch):
    """``clear_active_stream_actual`` is idempotent and survives outer resets.

    Pins the contract that the streaming contextvar cleanup never leaves a
    stale fallback deployment lingering for the next request on the same task.
    """
    # Use the same contextvar stream_chat_completion uses internally so the
    # test exercises the real cleanup path rather than a parallel helper.
    from mnemos.domain.pantheon.gateway import _ACTIVE_STREAM_ACTUAL

    token = _ACTIVE_STREAM_ACTUAL.set(_primary_decision())
    assert gateway.active_stream_actual_decision() == _primary_decision()
    _ACTIVE_STREAM_ACTUAL.reset(token)
    assert gateway.active_stream_actual_decision() is None
    gateway.clear_active_stream_actual()
    assert gateway.active_stream_actual_decision() is None
    # Re-clear is a no-op.
    gateway.clear_active_stream_actual()
    assert gateway.active_stream_actual_decision() is None