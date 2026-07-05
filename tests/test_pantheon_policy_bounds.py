from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from mnemos.domain.pantheon import catalog, router


def _settings(*, default_quality_floor: float = 0.8, default_max_cost: float | None = 10.0) -> SimpleNamespace:
    return SimpleNamespace(
        pantheon=SimpleNamespace(
            default_quality_floor=default_quality_floor,
            default_max_cost_usd_per_mtok=default_max_cost,
            routing_window_minutes=15,
            passthrough_enabled=True,
        )
    )


async def _select_first(_pool, _alias, candidates, **_kwargs):
    return SimpleNamespace(
        selected=candidates[0],
        candidates=[candidate["id"] for candidate in candidates],
        scores={},
        selection_reason="test-select-first",
    )


@pytest.mark.parametrize(
    "body",
    [
        {"max_cost": 999.0},
        {"max_cost": None},
        {"pantheon": {"max_cost": None}},
    ],
)
def test_request_max_cost_cannot_relax_operator_cost_ceiling(
    monkeypatch: pytest.MonkeyPatch,
    body: dict[str, object],
) -> None:
    async def _models():
        return [
            {
                "id": "expensive-reasoner",
                "provider": "paid",
                "capabilities": ["reasoning"],
                "quality_score": 0.95,
                "cost_per_mtok": 20.0,
            }
        ]

    monkeypatch.setattr(catalog, "list_models", _models)
    monkeypatch.setattr(router, "get_settings", lambda: _settings(default_max_cost=10.0))

    with pytest.raises(router.PantheonRoutingError) as exc:
        asyncio.run(router.route_model("auto:reasoning", body))

    assert exc.value.status_code == 404


def test_request_quality_floor_cannot_lower_operator_quality_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _models():
        return [
            {
                "id": "low-quality-reasoner",
                "provider": "cheap",
                "capabilities": ["reasoning"],
                "quality_score": 0.2,
                "cost_per_mtok": 0.1,
            },
            {
                "id": "policy-quality-reasoner",
                "provider": "paid",
                "capabilities": ["reasoning"],
                "quality_score": 0.9,
                "cost_per_mtok": 1.0,
            },
        ]

    monkeypatch.setattr(catalog, "list_models", _models)
    monkeypatch.setattr(router, "get_settings", lambda: _settings(default_quality_floor=0.8))
    monkeypatch.setattr(router, "resolve_with_policy", _select_first)

    decision = asyncio.run(router.route_model("auto:reasoning", {"quality_floor": 0.1}))

    assert decision.model_id == "policy-quality-reasoner"
    assert decision.candidates == ["policy-quality-reasoner"]
