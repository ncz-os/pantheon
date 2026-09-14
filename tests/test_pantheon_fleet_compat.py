from __future__ import annotations

from pathlib import Path
import tomllib


def test_fleet_dependencies_are_bounded_to_supported_release_lines() -> None:
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = set(pyproject["project"]["dependencies"])

    assert "mnemos-core>=6.2,<7" in dependencies
    assert "mnemos-graeae>=0.1.0,<0.2" in dependencies


def test_supported_current_core_and_graeae_import_together() -> None:
    import mnemos.core.resilience as resilience
    from mnemos.domain.pantheon.fleet_compat import install_core_graeae_compat

    install_core_graeae_compat()

    assert not hasattr(resilience, "_get_lifecycle_redis_client")
    from mnemos.domain.graeae.engine import GraeaeEngine

    assert GraeaeEngine is not None


def test_compat_does_not_replace_core_symbols(monkeypatch) -> None:
    import mnemos.core.resilience as resilience
    from mnemos.domain.pantheon.fleet_compat import install_core_graeae_compat

    sentinel = lambda _settings: "core"  # noqa: E731
    monkeypatch.setattr(resilience, "_redis_requested", sentinel, raising=False)

    install_core_graeae_compat()

    assert resilience._redis_requested is sentinel


def test_compat_does_not_synthesize_deleted_core_symbols(monkeypatch) -> None:
    import mnemos.core.resilience as resilience
    from mnemos.domain.pantheon.fleet_compat import install_core_graeae_compat

    monkeypatch.delattr(resilience, "_get_lifecycle_redis_client", raising=False)
    install_core_graeae_compat()
    assert not hasattr(resilience, "_get_lifecycle_redis_client")
