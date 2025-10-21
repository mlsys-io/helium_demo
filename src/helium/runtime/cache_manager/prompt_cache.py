import sys
from collections.abc import Sequence
from typing import Hashable

from cachetools import LRUCache

from helium.common import Message
from helium.runtime.worker.worker_input import WorkerInput


class PromptCacheManager:
    def __init__(self, maxsize: int = sys.maxsize) -> None:
        self._maxsize: int = maxsize
        self._cache: dict[
            type[WorkerInput], LRUCache[Hashable, str | list[Message]]
        ] = {}
        self._is_frozen: bool = False

    @property
    def is_frozen(self) -> bool:
        return self._is_frozen

    def freeze(self) -> None:
        self._is_frozen = True

    def unfreeze(self) -> None:
        self._is_frozen = False

    def query(self, inp: WorkerInput, key: Hashable) -> str | list[Message] | None:
        inp_type = type(inp)
        if inp_type not in self._cache:
            return None
        return self._cache[inp_type].get(key, None)

    def batch_query(
        self, inp: WorkerInput, keys: Sequence[Hashable]
    ) -> list[str | None] | list[list[Message] | None]:
        inp_type = type(inp)
        if inp_type not in self._cache:
            return [None] * len(keys)  # type: ignore
        cache = self._cache[inp_type]
        return [cache.get(key, None) for key in keys]  # type: ignore

    def store(
        self,
        inp: WorkerInput,
        key: Hashable,
        value: str | list[Message],
        overwrite: bool = False,
    ) -> None:
        if self._is_frozen:
            return
        inp_type = type(inp)
        if inp_type in self._cache:
            cache = self._cache[inp_type]
        else:
            cache = LRUCache(maxsize=self._maxsize)
            self._cache[inp_type] = cache
        if overwrite or key not in cache:
            cache[key] = value

    def batch_store_cache(
        self,
        inp: WorkerInput,
        batch: dict[Hashable, str] | dict[Hashable, list[Message]],
        overwrite: bool = False,
    ) -> None:
        if self._is_frozen:
            return
        inp_type = type(inp)
        if inp_type in self._cache:
            cache = self._cache[inp_type]
        else:
            cache = LRUCache(maxsize=self._maxsize)
            self._cache[inp_type] = cache
        for key, value in batch.items():
            if overwrite or key not in cache:
                cache[key] = value

    def clear(self) -> None:
        self._cache.clear()
