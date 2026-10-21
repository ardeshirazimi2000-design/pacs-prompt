from __future__ import annotations

from collections.abc import Callable

from ctpipeline.adapters.base import ModelAdapter
from ctpipeline.adapters.dummy import DummyThresholdAdapter
from ctpipeline.common.errors import ModelError

_REGISTRY: dict[str, Callable[[], ModelAdapter]] = {
    DummyThresholdAdapter.name: DummyThresholdAdapter,
}


def register(name: str, factory: Callable[[], ModelAdapter]) -> None:
    _REGISTRY[name] = factory


def available_models() -> list[str]:
    return sorted(_REGISTRY)


def create_adapter(name: str) -> ModelAdapter:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise ModelError(f"Unknown model '{name}'", code="UNKNOWN_MODEL") from None
