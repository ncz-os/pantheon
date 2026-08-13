from __future__ import annotations

import pytest

from mnemos.domain.pantheon import catalog


@pytest.mark.asyncio
async def test_catalog_snapshot_reuses_loader_and_isolates_mutation(monkeypatch):
    calls = 0

    async def loader(*, use_synced_cache=True):
        nonlocal calls
        calls += 1
        return [{"id": "model", "capabilities": ["chat"]}]

    monkeypatch.setattr(catalog, "list_models", loader)
    monkeypatch.setenv("MNEMOS_PANTHEON_CATALOG_SNAPSHOT_TTL_SECONDS", "30")
    catalog.invalidate_catalog_snapshot()

    first = await catalog.catalog_snapshot()
    first[0]["capabilities"].append("mutated")
    second = await catalog.catalog_snapshot()

    assert calls == 1
    assert second == [{"id": "model", "capabilities": ["chat"]}]


@pytest.mark.asyncio
async def test_catalog_snapshot_can_be_disabled_and_invalidated(monkeypatch):
    calls = 0

    async def loader(*, use_synced_cache=True):
        nonlocal calls
        calls += 1
        return [{"id": str(calls)}]

    monkeypatch.setattr(catalog, "list_models", loader)
    monkeypatch.setenv("MNEMOS_PANTHEON_CATALOG_SNAPSHOT_TTL_SECONDS", "0")
    catalog.invalidate_catalog_snapshot()
    assert await catalog.catalog_snapshot() == [{"id": "1"}]
    assert await catalog.catalog_snapshot() == [{"id": "2"}]

    monkeypatch.setenv("MNEMOS_PANTHEON_CATALOG_SNAPSHOT_TTL_SECONDS", "30")
    assert await catalog.catalog_snapshot() == [{"id": "3"}]
    catalog.invalidate_catalog_snapshot()
    assert await catalog.catalog_snapshot() == [{"id": "4"}]
