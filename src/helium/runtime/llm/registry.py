from collections.abc import Callable, Iterable
from typing import TypeVar

from class_registry import ClassRegistry

from helium.runtime.llm import BaseLLM, LLMServiceConfig

T = TypeVar("T")


class LLMRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, name: str) -> Callable[[T], T]:
        return cls.registry.register(name)

    @classmethod
    def keys(cls) -> Iterable[str]:
        return cls.registry.keys()

    @classmethod
    def items(cls) -> dict[str, BaseLLM]:
        return dict(cls.registry.items())

    @classmethod
    def get(cls, key: str, *args, config: LLMServiceConfig, **kwargs) -> BaseLLM:
        return cls.registry.get(key, *args, config=config, **kwargs)
