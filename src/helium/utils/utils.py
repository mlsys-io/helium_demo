import asyncio
import itertools
from collections.abc import AsyncGenerator, Coroutine, Iterable, Sequence
from threading import Thread
from typing import Any, TypeVar, cast

import shortuuid

T = TypeVar("T")
V = TypeVar("V")


def unique_id() -> str:
    return str(shortuuid.uuid())


def identity(x: T) -> T:
    return x


def iter_batch(iterable: Iterable[T], batch_size: int) -> Iterable[list[T]]:
    """Yield batches of items from an iterable."""
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, batch_size))
        if not batch:
            break
        yield batch


def partition(lst: list[T], n: int) -> Iterable[list[T]]:
    """Partition a list into n approximately equal parts."""
    k, m = divmod(len(lst), n)
    for i in range(n):
        start = i * k + min(i, m)
        end = (i + 1) * k + min(i + 1, m)
        yield lst[start:end]


def check_and_cast_list(t: type[T], lst: Sequence[V]) -> list[T]:
    if not isinstance(lst, list):
        raise ValueError(f"Expected a list, got {type(lst)}")
    if len(lst) == 0:
        return cast(list[T], lst)
    item = lst[0]
    if not isinstance(item, t):
        raise ValueError(f"Expected a list of {t}, got a list of {type(item)}")
    return cast(list[T], lst)


def indices_to_list(indices: Iterable[int] | slice) -> list[int]:
    return (
        list(range(indices.stop))[indices]
        if isinstance(indices, slice)
        else list(indices)
    )


async def execute_unordered(
    tasks: Iterable[asyncio.Task[T]],
) -> AsyncGenerator[set[asyncio.Task[T]], None]:
    pending = set(tasks)
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        yield done


class _AsyncRunner:
    def __init__(self) -> None:
        self._main_loop = asyncio.get_event_loop()
        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=self._run_event_loop)
        self._thread.start()
        self._is_stopped: bool = False

    def _run_event_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, task: Coroutine[Any, Any, T]) -> T:
        if self._is_stopped:
            raise RuntimeError("Event loop has stopped")
        if asyncio.get_event_loop() is self._main_loop:
            return asyncio.run_coroutine_threadsafe(task, self._loop).result()
        return asyncio.run_coroutine_threadsafe(task, self._main_loop).result()

    def stop(self):
        if not self._is_stopped:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join()
            self._is_stopped = True


_async_runner: _AsyncRunner | None = None


def async_runner() -> _AsyncRunner:
    global _async_runner
    if _async_runner is None:
        _async_runner = _AsyncRunner()
    return _async_runner


def run_coroutine_blocking(coro: Coroutine[Any, Any, T]) -> T:
    return async_runner().run(coro)


def stop_async_runner() -> None:
    global _async_runner
    if _async_runner is not None:
        _async_runner.stop()
        _async_runner = None
