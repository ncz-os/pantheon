"""Tests for gateway.forward_chat_completion routed through RouterRuntime.

Behavior-preserving wiring: success returns data, transient 5xx is retried then
succeeds, a non-retryable 400 still surfaces as the original PantheonGatewayError.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from mnemos.domain.pantheon import codex_oauth, gateway
from mnemos.domain.pantheon.cooldown import CooldownManager, InMemoryCooldownStore
from mnemos.domain.pantheon.gateway import PantheonGatewayError, forward_chat_completion
from mnemos.domain.pantheon.router import RouteDecision
from mnemos.domain.pantheon.runtime import RouterRuntime


async def _noop_sleep(_d):
    return None


@pytest.fixture(autouse=True)
def _no_sleep_runtime():
    gateway.set_runtime(
        RouterRuntime(
            CooldownManager(InMemoryCooldownStore()),
            clock=time.time,
            sleep=_noop_sleep,
            rng=lambda: 0.0,
        )
    )
    yield
    gateway.set_runtime(None)


def _decision():
    return RouteDecision(alias="a", provider="p", model_id="m", route_type="single", reason="r")


def _eih_decision():
    return RouteDecision(alias="a", provider="eih", model_id="gpt-5.5", route_type="single", reason="r")


def _force_openai(monkeypatch):
    monkeypatch.setattr(gateway, "_provider_config", lambda d: {"api": "openai", "url": "http://x"})


def test_success_returns_data(monkeypatch):
    _force_openai(monkeypatch)

    async def fake(decision, body):
        return {"model": "m", "ok": True}

    monkeypatch.setattr(gateway, "_forward_chat_once", fake)
    out = asyncio.run(forward_chat_completion(_decision(), {"messages": []}))
    assert out == {"model": "m", "ok": True}


def test_transient_5xx_is_retried_then_succeeds(monkeypatch):
    _force_openai(monkeypatch)
    calls = {"n": 0}

    async def fake(decision, body):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PantheonGatewayError(503, "upstream down")
        return {"model": "m", "recovered": True}

    monkeypatch.setattr(gateway, "_forward_chat_once", fake)
    out = asyncio.run(forward_chat_completion(_decision(), {"messages": []}))
    assert out["recovered"] is True
    assert calls["n"] == 2  # one failure + one retry on the same provider


def test_400_surfaces_unchanged(monkeypatch):
    _force_openai(monkeypatch)

    async def fake(decision, body):
        raise PantheonGatewayError(400, "bad request")

    monkeypatch.setattr(gateway, "_forward_chat_once", fake)
    with pytest.raises(PantheonGatewayError) as ei:
        asyncio.run(forward_chat_completion(_decision(), {"messages": []}))
    assert ei.value.status_code == 400
    assert ei.value.message == "bad request"


def test_400_is_not_retried(monkeypatch):
    _force_openai(monkeypatch)
    calls = {"n": 0}

    async def fake(decision, body):
        calls["n"] += 1
        raise PantheonGatewayError(400, "bad")

    monkeypatch.setattr(gateway, "_forward_chat_once", fake)
    with pytest.raises(PantheonGatewayError):
        asyncio.run(forward_chat_completion(_decision(), {"messages": []}))
    assert calls["n"] == 1  # non-retryable: single attempt, no retry


def test_codex_oauth_fallback_runs_for_eih_transport_outage(monkeypatch):
    _force_openai(monkeypatch)

    async def fake_primary(_decision, _body):
        raise httpx.ConnectError("connection refused")

    async def fake_codex(decision, body):
        assert decision.provider == "eih"
        assert body == {"messages": []}
        return {"model": "gpt-5.3-codex-spark", "fallback": "codex-oauth"}

    monkeypatch.setattr(gateway, "_forward_chat_once", fake_primary)
    monkeypatch.setattr(codex_oauth, "forward_chat_completion", fake_codex)

    out = asyncio.run(forward_chat_completion(_eih_decision(), {"messages": []}))
    assert out == {"model": "gpt-5.3-codex-spark", "fallback": "codex-oauth"}


def test_codex_oauth_fallback_runs_for_eih_503(monkeypatch):
    _force_openai(monkeypatch)
    calls = {"primary": 0, "codex": 0}

    async def fake_primary(_decision, _body):
        calls["primary"] += 1
        raise PantheonGatewayError(503, "upstream down")

    async def fake_codex(_decision, _body):
        calls["codex"] += 1
        return {"model": "gpt-5.3-codex-spark", "fallback": True}

    monkeypatch.setattr(gateway, "_forward_chat_once", fake_primary)
    monkeypatch.setattr(codex_oauth, "forward_chat_completion", fake_codex)

    out = asyncio.run(forward_chat_completion(_eih_decision(), {"messages": []}))
    assert out["fallback"] is True
    assert calls["primary"] == 3  # initial attempt + two same-deployment retries
    assert calls["codex"] == 1


@pytest.mark.parametrize("status", [400, 401, 403, 429])
def test_codex_oauth_fallback_does_not_hide_eih_4xx(monkeypatch, status):
    _force_openai(monkeypatch)

    async def fake_primary(_decision, _body):
        raise PantheonGatewayError(status, "client-side failure")

    async def fake_codex(_decision, _body):
        raise AssertionError("Codex OAuth fallback must not run for normal 4xx")

    monkeypatch.setattr(gateway, "_forward_chat_once", fake_primary)
    monkeypatch.setattr(codex_oauth, "forward_chat_completion", fake_codex)

    with pytest.raises(PantheonGatewayError) as ei:
        asyncio.run(forward_chat_completion(_eih_decision(), {"messages": []}))
    assert ei.value.status_code == status


def test_codex_oauth_fallback_is_limited_to_ngc_eih_outages(monkeypatch):
    _force_openai(monkeypatch)

    async def fake_primary(_decision, _body):
        raise PantheonGatewayError(503, "other provider down")

    async def fake_codex(_decision, _body):
        raise AssertionError("Codex OAuth fallback must not run for non-EIH/NGC providers")

    monkeypatch.setattr(gateway, "_forward_chat_once", fake_primary)
    monkeypatch.setattr(codex_oauth, "forward_chat_completion", fake_codex)

    with pytest.raises(PantheonGatewayError) as ei:
        asyncio.run(forward_chat_completion(_decision(), {"messages": []}))
    assert ei.value.status_code == 503


def test_codex_oauth_fallback_does_not_launder_primary_400_through_later_503(monkeypatch):
    _force_openai(monkeypatch)
    sibling = RouteDecision(alias="b", provider="deepseek-direct", model_id="m2", route_type="single", reason="r")

    async def fake_chain(_decision):
        return [_eih_decision(), sibling]

    async def fake_primary(decision, _body):
        if decision.provider == "eih":
            raise PantheonGatewayError(400, "bad request")
        raise PantheonGatewayError(503, "fallback provider down")

    async def fake_codex(_decision, _body):
        raise AssertionError("Codex OAuth fallback must not run after a target-provider 400")

    monkeypatch.setattr(gateway, "_runtime_chain", fake_chain)
    monkeypatch.setattr(gateway, "_forward_chat_once", fake_primary)
    monkeypatch.setattr(codex_oauth, "forward_chat_completion", fake_codex)

    with pytest.raises(PantheonGatewayError) as ei:
        asyncio.run(forward_chat_completion(_eih_decision(), {"messages": []}))
    assert ei.value.status_code == 503


def test_codex_oauth_fallback_does_not_run_on_config_error_503(monkeypatch):
    # A local config fault (no endpoint configured) surfaces as 503 from a helper
    # inside _forward_chat_once, but must fail closed: the Codex-OAuth fallback
    # must NOT launder a misconfiguration into a live call, even for a
    # fallback-eligible provider (eih).
    _force_openai(monkeypatch)

    async def fake_chain(_decision):
        return [_eih_decision()]

    async def fake_primary(_decision, _body):
        raise PantheonGatewayError(503, "provider has no chat endpoint configured", config_error=True)

    async def fake_codex(_decision, _body):
        raise AssertionError("Codex OAuth fallback must not run on a local config error")

    monkeypatch.setattr(gateway, "_runtime_chain", fake_chain)
    monkeypatch.setattr(gateway, "_forward_chat_once", fake_primary)
    monkeypatch.setattr(codex_oauth, "forward_chat_completion", fake_codex)

    with pytest.raises(PantheonGatewayError) as ei:
        asyncio.run(forward_chat_completion(_eih_decision(), {"messages": []}))
    assert ei.value.status_code == 503
    assert ei.value.config_error is True


def test_classify_preserves_config_error_through_normalization():
    # The config_error marker must survive classify() so it is not lost when the
    # runtime records an AttemptRecord (a NormalizedError) for a failed deployment.
    from mnemos.domain.pantheon.http_bridge import classify

    cfg_err = classify(PantheonGatewayError(503, "provider has no chat endpoint configured", config_error=True))
    assert cfg_err.config_error is True
    assert cfg_err.status_code == 503

    outage = classify(PantheonGatewayError(503, "upstream down"))
    assert outage.config_error is False


def test_route_failure_trigger_rejects_normalized_config_error_attempt():
    # The target-attempts path (eih/ngc/nvidia) reads NormalizedError off
    # AttemptRecord, NOT the original exception. A config-error attempt there must
    # still block the Codex-OAuth fallback.
    from types import SimpleNamespace

    from mnemos.domain.pantheon.errors import ErrorClass, NormalizedError, RetryAction
    from mnemos.domain.pantheon.fallback import AllDeploymentsFailed, AttemptRecord

    def _failure(config_error: bool) -> AllDeploymentsFailed:
        err = NormalizedError(
            ErrorClass.SERVICE_UNAVAILABLE, 503, True, True, "no endpoint", provider="eih", config_error=config_error
        )
        record = AttemptRecord(SimpleNamespace(provider="eih"), err, RetryAction.FALLOVER, 0)
        return AllDeploymentsFailed([record], last_exception=None)

    # config-error attempt -> NOT eligible for fallback (fail closed)
    assert gateway._codex_oauth_route_failure_trigger(_eih_decision(), _failure(True)) is False
    # genuine outage attempt -> eligible (unchanged behavior)
    assert gateway._codex_oauth_route_failure_trigger(_eih_decision(), _failure(False)) is True


def test_config_error_is_terminal_no_fallover():
    # A config_error is terminal in decide(): never RETRY, never FALLOVER, even
    # with healthy siblings available. Surfaces immediately so the gateway fails
    # closed instead of masquerading as a transient outage.
    from mnemos.domain.pantheon.errors import ErrorClass, NormalizedError, RetryAction, decide

    cfg_err = NormalizedError(ErrorClass.SERVICE_UNAVAILABLE, 503, True, True, "no endpoint", config_error=True)
    assert decide(cfg_err, num_deployments=3, has_fallbacks=True) is RetryAction.RAISE

    # Control: a genuine 503 outage with siblings still falls over (unchanged).
    outage = NormalizedError(ErrorClass.SERVICE_UNAVAILABLE, 503, True, True, "down")
    assert decide(outage, num_deployments=3, has_fallbacks=True) is not RetryAction.RAISE


def test_missing_api_key_is_config_error_and_blocks_fallback():
    # A missing local credential is a config fault: it must carry config_error and
    # therefore be ineligible for the Codex-OAuth fallback after normalization.
    from mnemos.domain.pantheon.http_bridge import classify

    err = PantheonGatewayError(503, "missing api_key for provider key_name='nvidia'", config_error=True)
    assert classify(err).config_error is True
    assert gateway._codex_oauth_fallback_trigger(_eih_decision(), err) is False


def test_mixed_outage_and_config_error_attempts_block_fallback():
    # Fail closed: a genuine eih 503 outage paired with a LATER non-target
    # config-error attempt must NOT be laundered into Codex-OAuth fallback. The
    # first-pass any-config_error guard catches it before target filtering.
    from types import SimpleNamespace

    from mnemos.domain.pantheon.errors import ErrorClass, NormalizedError, RetryAction
    from mnemos.domain.pantheon.fallback import AllDeploymentsFailed, AttemptRecord

    eih_outage = NormalizedError(ErrorClass.SERVICE_UNAVAILABLE, 503, True, True, "down", provider="eih")
    other_cfg = NormalizedError(
        ErrorClass.SERVICE_UNAVAILABLE, 503, True, True, "no endpoint", provider="deepseek-direct", config_error=True
    )
    failure = AllDeploymentsFailed(
        [
            AttemptRecord(SimpleNamespace(provider="eih"), eih_outage, RetryAction.FALLOVER, 0),
            AttemptRecord(SimpleNamespace(provider="deepseek-direct"), other_cfg, RetryAction.RAISE, 0),
        ],
        last_exception=None,
    )
    assert gateway._codex_oauth_route_failure_trigger(_eih_decision(), failure) is False


def test_is_openai_api_keeps_config_faulted_candidate_in_chain(monkeypatch):
    # A config-faulted candidate is KEPT (True) so it stays in the runtime chain,
    # fails terminally when attempted, and records its config_error in the attempt
    # trail (blocking fallback laundering). A healthy openai provider is kept; a
    # resolved non-openai (Graeae) provider is filtered out.
    def fake_provider_config(decision):
        if decision.provider == "broken":
            raise PantheonGatewayError(503, "provider has no endpoint configured", config_error=True)
        if decision.provider == "gemini":
            return {"api": "gemini"}
        return {"api": "openai", "url": "https://ok.test/v1"}

    monkeypatch.setattr(gateway, "_provider_config", fake_provider_config)
    assert gateway._is_openai_api(RouteDecision(alias="a", provider="broken", model_id="m", route_type="single", reason="r")) is True
    assert gateway._is_openai_api(RouteDecision(alias="a", provider="gemini", model_id="m", route_type="single", reason="r")) is False
    assert gateway._is_openai_api(RouteDecision(alias="a", provider="nvidia", model_id="m", route_type="single", reason="r")) is True


def test_config_error_is_non_cooldownable_and_non_retryable():
    # A config fault must not trip a cooldown (which would pre-filter the bad
    # deployment out of later requests and hide its config_error from the attempt
    # trail) nor be retryable.
    from mnemos.domain.pantheon.http_bridge import classify

    cfg_err = classify(PantheonGatewayError(503, "provider has no endpoint configured", config_error=True))
    assert cfg_err.config_error is True
    assert cfg_err.cooldownable is False
    assert cfg_err.retryable is False

    # A genuine 503 outage is still cooldownable/retryable (unchanged).
    outage = classify(PantheonGatewayError(503, "upstream down"))
    assert outage.cooldownable is True
    assert outage.retryable is True


def test_consensus_still_delegates(monkeypatch):
    captured = {}

    async def fake_consensus(decision, body):
        captured["hit"] = True
        return {"consensus": True}

    monkeypatch.setattr(gateway, "consensus_chat_completion", fake_consensus)
    dec = RouteDecision(alias="c", provider="p", model_id=None, route_type="consensus", reason="r")
    out = asyncio.run(forward_chat_completion(dec, {"messages": []}))
    assert out == {"consensus": True}
    assert captured["hit"] is True
