"""Durable admission at the actual deployment attempt boundary."""

from __future__ import annotations

import json
from functools import wraps

from mnemos.domain.knemon.attempts import Attempt, finite_cost
from mnemos.domain.knemon.budget import (
    CostReservationConflict,
    CostReservationSchemaMissing,
)


async def begin_attempt(decision, body, *, embeddings=False):
    from mnemos.domain.pantheon import gateway

    model = decision.model or {}

    def price(*keys):
        for key in keys:
            if key in model:
                value = finite_cost(model[key])
                if value is not None:
                    return value
        return None

    input_price = price("input_cost_per_mtok", "price_in", "cost_per_mtok")
    output_price = (
        0.0
        if embeddings
        else price("output_cost_per_mtok", "price_out", "cost_per_mtok")
    )
    # UTF-8 byte count bounds token count for text tokenizers; word-count
    # estimates can under-reserve non-English, code, tools, and JSON prompts.
    tokens_in = max(1, len(json.dumps(body, default=str).encode("utf-8")))
    tokens_out = (
        0 if embeddings else gateway._effective_max_tokens(body, decision.model_id)
    )
    if tokens_out <= 0 and not embeddings:
        tokens_out = 4096
    estimate = (
        None
        if input_price is None or output_price is None
        else (
            tokens_in * input_price
            + tokens_out * output_price * gateway._candidate_multiplicity(body)
        )
        / 1_000_000
    )
    attempt = Attempt(
        provider=str(decision.provider or ""),
        model=decision.model_id or decision.alias,
        subsystem="pantheon",
        estimate=estimate,
        input_price=input_price,
        output_price=output_price,
        task_kind="embeddings" if embeddings else "chat",
    )
    try:
        return await attempt.begin()
    except CostReservationConflict as exc:
        raise gateway.PantheonGatewayError(429, str(exc), config_error=True) from exc
    except CostReservationSchemaMissing as exc:
        raise gateway.PantheonGatewayError(503, str(exc), config_error=True) from exc


def accounted(*, embeddings=False):
    def decorate(function):
        @wraps(function)
        async def wrapped(decision, body):
            attempt = await begin_attempt(decision, body, embeddings=embeddings)
            outcome = "err"
            try:
                response = await function(decision, body)
                attempt.observe(response)
                outcome = "ok"
                return response
            except Exception as exc:
                if getattr(exc, "config_error", False):
                    outcome = "unused"
                raise
            finally:
                await attempt.finish(outcome)

        return wrapped

    return decorate
