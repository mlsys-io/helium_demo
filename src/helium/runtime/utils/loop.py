import asyncio
import multiprocessing as mp
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Generic, TypeVar

from helium.runtime.utils.logger import log_on_exception_async
from helium.runtime.utils.pool import AsyncPool, MPAsyncPool
from helium.runtime.utils.queue import AIOQueue, AsyncQueue, MPQueue
from helium.utils import unique_id

E = TypeVar("E")
R = TypeVar("R")
Ctx = TypeVar("Ctx")


class _EventBatch(Generic[E]):
    __slots__ = ("keys", "events")

    def __init__(self, keys: list[str] | None, events: list[E]) -> None:
        self.keys = keys
        self.events = events


class EventLoop(ABC, Generic[E, R, Ctx]):
    """
    Abstract base class for event-driven processing loops.

    This class provides a framework for processing events asynchronously using a handler function.
    Events are queued in an input channel and processed sequentially or concurrently depending
    on the implementation. Results can optionally be stored in a result pool for retrieval.

    Type Parameters:
        E: Event type - the type of events to be processed
        R: Result type - the type of results produced by event processing
        Ctx: Context type - the type of context passed to the handler function

    Args:
        handler_func: Async function that processes events and returns results
        in_channel: Queue for incoming events
        result_collector: Optional pool or queue for storing results with string keys
        context_manager: Optional async context manager for handler execution
        key_prefix: Optional prefix for generated result keys
    """

    def __init__(
        self,
        handler_func: Callable[[E, Ctx | None], Awaitable[R]],
        in_channel: AsyncQueue,
        result_collector: AsyncPool[str, R] | AsyncQueue[tuple[str, R]] | None = None,
        context_manager: AbstractAsyncContextManager[Ctx] | None = None,
        key_prefix: str | None = None,
    ):
        self._handler_func = handler_func
        self._in_channel = in_channel
        self._result_collector = result_collector
        self._context_manager = context_manager
        self._key_prefix = key_prefix

        self._id = unique_id()
        self._counter = 0

        self._is_started: bool = False
        self._is_stopped: bool = False

        self._put_result = self._get_put_result_func(result_collector)

    def is_started(self) -> bool:
        return self._is_started

    def is_stopped(self) -> bool:
        return self._is_stopped

    def is_running(self) -> bool:
        return self.is_started() and not self.is_stopped()

    def _check_running(self) -> None:
        if not self.is_started():
            raise ValueError("Event loop has not been started.")
        if self.is_stopped():
            raise ValueError("Event loop has been stopped.")

    async def start(self) -> None:
        if self.is_started():
            return
        await self._start_handler()
        self._is_started = True

    async def stop(self) -> None:
        self._in_channel.close()
        await self.join()
        if isinstance(self._result_collector, AsyncQueue):
            self._result_collector.close()

    async def join(self) -> None:
        if self.is_stopped():
            return
        await self._join_handler()
        self._is_stopped = True

    async def add_event(self, event: E) -> str | None:
        self._check_running()
        if self._result_collector is None:
            await self._in_channel.put(event)
            return None
        key = self._new_key()
        await self._in_channel.put((key, event))
        return key

    async def add_event_batch(self, events: list[E]) -> list[str] | None:
        self._check_running()
        if self._result_collector is None:
            await self._in_channel.put(_EventBatch(keys=None, events=events))
            return None
        keys = [self._new_key() for _ in range(len(events))]
        await self._in_channel.put(_EventBatch(keys=keys, events=events))
        return keys

    async def get_result(self, key: str) -> R:
        if isinstance(self._result_collector, AsyncPool):
            return await self._result_collector.pop(key)
        raise ValueError(
            "Expect result_collector to be a pool. "
            f"Got {type(self._result_collector)}"
        )

    async def get_result_no_wait(self, key: str) -> R | None:
        if isinstance(self._result_collector, AsyncPool):
            return await self._result_collector.pop_nowait(key)
        raise ValueError(
            "Expect result_collector to be a pool. "
            f"Got {type(self._result_collector)}"
        )

    async def pop_result(self) -> tuple[str, R]:
        if isinstance(self._result_collector, AsyncQueue):
            return await self._result_collector.get()
        raise ValueError(
            "Expect result_collector to be a queue. "
            f"Got {type(self._result_collector)}"
        )

    def pop_result_no_wait(self) -> tuple[str, R] | None:
        if isinstance(self._result_collector, AsyncQueue):
            return self._result_collector.get_nowait()
        raise ValueError(
            "Expect result_collector to be a queue. "
            f"Got {type(self._result_collector)}"
        )

    async def pop_all_results(self) -> list[tuple[str, R]]:
        if isinstance(self._result_collector, AsyncQueue):
            return await self._result_collector.get_all()
        raise ValueError(
            "Expect result_collector to be a queue. "
            f"Got {type(self._result_collector)}"
        )

    async def process_event(self, event: E) -> R:
        key = await self.add_event(event)
        if key is None:
            raise ValueError("No result available.")
        if isinstance(self._result_collector, AsyncPool):
            return await self.get_result(key)
        k, result = await self.pop_result()
        if k != key:
            raise ValueError(f"Result key mismatch: expected {key}, got {k}")
        return result

    def _new_key(self) -> str:
        self._counter += 1
        if self._key_prefix is None:
            return f"{self._id}-{self._counter}"
        return f"{self._key_prefix}-{self._id}-{self._counter}"

    @log_on_exception_async(ignore=[asyncio.CancelledError])
    async def loop(self, *args, **kwargs) -> None:
        loop_func = (
            self._loop_func_no_result
            if self._result_collector is None
            else self._loop_func_with_result
        )
        if self._context_manager is None:
            await loop_func(None, *args, **kwargs)
        else:
            async with self._context_manager as ctx:
                await loop_func(ctx, *args, **kwargs)

    def _get_put_result_func(
        self, collector: AsyncPool[str, R] | AsyncQueue[tuple[str, R]] | None
    ) -> Callable[[str, R], Awaitable[None]]:
        async def raise_error(key: str, result: R) -> None:
            raise ValueError("Result collector is not initialized.")

        if isinstance(collector, AsyncPool):
            return lambda key, result: collector.put(key, result)
        if isinstance(collector, AsyncQueue):
            return lambda key, result: collector.put((key, result))
        return raise_error

    @abstractmethod
    async def _start_handler(self) -> None:
        pass

    @abstractmethod
    async def _join_handler(self) -> None:
        pass

    @abstractmethod
    async def _loop_func_no_result(self, ctx: Ctx | None, *args, **kwargs) -> None:
        pass

    @abstractmethod
    async def _loop_func_with_result(self, ctx: Ctx | None, *args, **kwargs) -> None:
        pass


class AsyncEventLoop(EventLoop[E, R, Ctx]):
    """
    Asynchronous event loop implementation that processes events sequentially.

    This implementation runs within the current async event loop and processes events
    one at a time in the order they arrive. The processing is handled by an asyncio Task
    that continuously reads from the input channel and invokes the handler function.
    """

    def __init__(
        self,
        handler_func: Callable[[E, Ctx | None], Awaitable[R]],
        in_channel: AsyncQueue | None = None,
        result_collector: AsyncPool[str, R] | AsyncQueue[tuple[str, R]] | None = None,
        context_manager: AbstractAsyncContextManager[Ctx] | None = None,
        key_prefix: str | None = None,
    ):
        super().__init__(
            handler_func=handler_func,
            in_channel=in_channel or AIOQueue(),
            result_collector=result_collector,
            context_manager=context_manager,
            key_prefix=key_prefix,
        )
        self._loop_task: asyncio.Task | None = None

    @property
    def loop_task(self) -> asyncio.Task:
        if self._loop_task is None:
            raise ValueError("Event loop has not been started.")
        return self._loop_task

    async def _start_handler(self) -> None:
        assert self._loop_task is None
        self._loop_task = asyncio.create_task(self.loop())

    async def _join_handler(self) -> None:
        assert self._loop_task is not None
        await self._loop_task
        self._loop_task = None

    async def _loop_func_no_result(self, ctx: Ctx | None, *args, **kwargs) -> None:
        assert self._in_channel is not None
        while True:
            item = await self._in_channel.get()
            if isinstance(item, _EventBatch):
                for event in item.events:
                    await self._handler_func(event, ctx)
            else:
                await self._handler_func(item, ctx)

    async def _loop_func_with_result(self, ctx: Ctx | None, *args, **kwargs) -> None:
        assert self._in_channel is not None
        while True:
            item = await self._in_channel.get()
            if isinstance(item, _EventBatch):
                assert item.keys is not None
                for key, event in zip(item.keys, item.events):
                    result = await self._handler_func(event, ctx)
                    await self._put_result(key, result)
            else:
                key, event = item
                result = await self._handler_func(event, ctx)
                await self._put_result(key, result)


class AsyncConcurrentEventLoop(
    AsyncEventLoop[E, R, tuple[set[asyncio.Task], Ctx | None]]
):
    """
    Asynchronous concurrent event loop implementation that processes events in parallel.

    This implementation extends AsyncEventLoop to support concurrent event processing
    within a single async event loop. Multiple events can be processed simultaneously
    using asyncio tasks, allowing for better utilization of async I/O operations.
    """

    def __init__(
        self,
        handler_func: Callable[[E, Ctx | None], Coroutine[Any, Any, R]],
        in_channel: AsyncQueue | None = None,
        result_collector: AsyncPool[str, R] | AsyncQueue[tuple[str, R]] | None = None,
        context_manager: AbstractAsyncContextManager[Ctx] | None = None,
        key_prefix: str | None = None,
    ):
        super().__init__(
            handler_func=self._handler_func_wrapper(handler_func, result_collector),
            in_channel=in_channel,
            result_collector=result_collector,
            context_manager=_task_ctx(context_manager),
            key_prefix=key_prefix,
        )
        setattr(self, "_loop_func_with_result", self._loop_func_no_result)

    def _handler_func_wrapper(
        self,
        handler_func: Callable[[E, Ctx | None], Coroutine[Any, Any, R]],
        result_collector: AsyncPool[str, R] | AsyncQueue[tuple[str, R]] | None,
    ) -> Callable[[Any, tuple[set[asyncio.Task], Ctx | None] | None], Awaitable[R]]:
        async def wrapper_with_result(event: tuple[str, E], ctx: Ctx | None) -> None:
            k, e = event
            res = await handler_func(e, ctx)
            await self._put_result(k, res)

        wrapper_func = handler_func if result_collector is None else wrapper_with_result

        async def wrapper_cleanup(
            event: Any, wrapper_ctx: tuple[set[asyncio.Task], Ctx | None] | None
        ) -> R:
            assert wrapper_ctx is not None
            tasks, context = wrapper_ctx
            task = asyncio.create_task(wrapper_func(event, context))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
            return None  # type: ignore

        return wrapper_cleanup


class MPEventLoop(EventLoop[E, R, Ctx]):
    """
    Multiprocessing event loop implementation that processes events in a separate process.

    This implementation runs the event processing loop in a dedicated Python process,
    completely separate from the main process. Communication between processes occurs
    through multiprocessing-safe queues and pools.
    """

    def __init__(
        self,
        handler_func: Callable[[E, Ctx | None], Awaitable[R]],
        in_channel: AsyncQueue | None = None,
        result_collector: AsyncPool[str, R] | AsyncQueue[tuple[str, R]] | None = None,
        context_manager: AbstractAsyncContextManager[Ctx] | None = None,
        key_prefix: str | None = None,
    ):
        super().__init__(
            handler_func=handler_func,
            in_channel=in_channel or MPQueue(),
            result_collector=result_collector,
            context_manager=context_manager,
            key_prefix=key_prefix,
        )
        self._loop_proc: mp.Process | None = None
        self._out_channel: MPQueue[tuple[str | None, R]] | None = None
        self._pulling_loop: AsyncEventLoop[R, R, Ctx] | None = None

    async def _start_handler(self) -> None:
        assert (
            self._loop_proc is None
            and self._out_channel is None
            and self._pulling_loop is None
        )
        out_channel: MPQueue[tuple[str | None, R]] = MPQueue()

        # Set the out channel
        self._out_channel = out_channel

        # Start the pulling loop
        if self._result_collector is not None:
            self._pulling_loop = AsyncEventLoop(
                handler_func=self._pulling_func,
                in_channel=out_channel,
                result_collector=self._result_collector,
            )
            await self._pulling_loop.start()

        # Start the main loop
        ready_queue: MPQueue[bool] = MPQueue()
        self._loop_proc = mp.Process(target=lambda: asyncio.run(self.loop(ready_queue)))
        self._loop_proc.start()
        is_ready = await ready_queue.get()

        if not is_ready:
            raise RuntimeError("Failed to start the event loop process.")

    async def _join_handler(self) -> None:
        assert not (self._loop_proc is None or self._out_channel is None)

        # Join the main loop
        self._loop_proc.join()
        self._loop_proc = None

        # Stop the pulling loop
        if self._pulling_loop is not None:
            await self._pulling_loop.stop()
            self._pulling_loop = None

        # Reset the out channel
        self._out_channel = None

    async def _loop_func_no_result(
        self, ctx: Ctx | None, ready_queue: MPQueue, *args, **kwargs
    ) -> None:
        assert self._in_channel is not None
        await ready_queue.put(True)
        while True:
            item = await self._in_channel.get()
            if isinstance(item, _EventBatch):
                for event in item.events:
                    await self._handler_func(event, ctx)
            else:
                await self._handler_func(item, ctx)

    async def _loop_func_with_result(
        self, ctx: Ctx | None, ready_queue: MPQueue, *args, **kwargs
    ) -> None:
        assert self._in_channel is not None and self._out_channel is not None
        await ready_queue.put(True)
        while True:
            item = await self._in_channel.get()
            if isinstance(item, _EventBatch):
                assert item.keys is not None
                for key, event in zip(item.keys, item.events):
                    result = await self._handler_func(event, ctx)
                    await self._out_channel.put((key, result))
            else:
                key, event = item
                result = await self._handler_func(event, ctx)
                await self._out_channel.put((key, result))

    async def _pulling_func(self, result: R, _) -> R:
        return result


class MPConcurrentEventLoop(MPEventLoop[E, R, tuple[set[asyncio.Task], Ctx | None]]):
    """
    Multiprocessing concurrent event loop implementation that combines process isolation with concurrency.

    This implementation extends MPEventLoop to support concurrent event processing within
    a separate process. It provides both the benefits of true parallel processing through
    multiprocessing and concurrent I/O operations within that process.
    """

    def __init__(
        self,
        handler_func: Callable[[E, Ctx | None], Coroutine[Any, Any, R]],
        in_channel: AsyncQueue | None = None,
        result_collector: AsyncPool[str, R] | AsyncQueue[tuple[str, R]] | None = None,
        context_manager: AbstractAsyncContextManager[Ctx] | None = None,
        key_prefix: str | None = None,
    ):
        super().__init__(
            handler_func=self._handler_func_wrapper(handler_func, result_collector),
            in_channel=in_channel,
            result_collector=result_collector,
            context_manager=_task_ctx(context_manager),
            key_prefix=key_prefix,
        )

    def _handler_func_wrapper(
        self,
        handler_func: Callable[[E, Ctx | None], Coroutine[Any, Any, R]],
        result_collector: AsyncPool[str, R] | AsyncQueue[tuple[str, R]] | None = None,
    ) -> Callable[[Any, tuple[set[asyncio.Task], Ctx | None] | None], Awaitable[R]]:
        async def wrapper_with_result(event: tuple[str, E], ctx: Ctx | None) -> None:
            assert self._out_channel is not None
            k, e = event
            res = await handler_func(e, ctx)
            await self._out_channel.put((k, res))

        wrapper_func = handler_func if result_collector is None else wrapper_with_result

        async def wrapper_cleanup(
            event: Any, wrapper_ctx: tuple[set[asyncio.Task], Ctx | None] | None
        ) -> R:
            assert wrapper_ctx is not None
            tasks, context = wrapper_ctx
            task = asyncio.create_task(wrapper_func(event, context))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
            return None  # type: ignore

        return wrapper_cleanup

    async def _loop_func_with_result(
        self,
        ctx: tuple[set[asyncio.Task], Ctx | None] | None,
        ready_queue: MPQueue,
        *args,
        **kwargs,
    ) -> None:
        assert self._in_channel is not None
        await ready_queue.put(True)
        while True:
            item = await self._in_channel.get()
            if isinstance(item, _EventBatch):
                assert item.keys is not None
                for key, event in zip(item.keys, item.events):
                    await self._handler_func((key, event), ctx)  # type: ignore
            else:
                await self._handler_func(item, ctx)


class AsyncMPEventLoop(EventLoop[E, R, Ctx]):
    """
    Asynchronous multiprocessing event loop implementation with simplified interface.

    This implementation provides a hybrid approach that combines multiprocessing queues
    for cross-process communication with async processing within the current process.
    It uses multiprocessing-safe pools and queues while maintaining async execution.
    """

    def __init__(
        self,
        handler_func: Callable[[E, Ctx | None], Awaitable[R]],
        has_result: bool = True,
        in_channel: MPQueue | None = None,
        context_manager: AbstractAsyncContextManager[Ctx] | None = None,
        key_prefix: str | None = None,
    ):
        super().__init__(
            handler_func=handler_func,
            in_channel=in_channel or MPQueue(),
            result_collector=MPAsyncPool() if has_result else None,
            context_manager=context_manager,
            key_prefix=key_prefix,
        )
        self._loop_task: asyncio.Task | None = None
        self._start_event = mp.Event()
        self._stop_event = mp.Event()

    @property
    def loop_task(self) -> asyncio.Task:
        if self._loop_task is None:
            raise ValueError("Event loop has not been started.")
        return self._loop_task

    def is_started(self) -> bool:
        return self._start_event.is_set()

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    async def join(self) -> None:
        await super().join()
        self._stop_event.set()

    async def _start_handler(self) -> None:
        assert self._loop_task is None
        self._loop_task = asyncio.create_task(self.loop())

    async def _join_handler(self) -> None:
        assert self._loop_task is not None
        await self._loop_task
        self._loop_task = None

    async def _loop_func_no_result(self, ctx: Ctx | None, *args, **kwargs) -> None:
        assert self._in_channel is not None
        self._start_event.set()
        while True:
            item = await self._in_channel.get()
            if isinstance(item, _EventBatch):
                for event in item.events:
                    await self._handler_func(event, ctx)
            else:
                await self._handler_func(item, ctx)

    async def _loop_func_with_result(self, ctx: Ctx | None, *args, **kwargs) -> None:
        assert self._in_channel is not None and self._result_collector is not None
        self._start_event.set()
        while True:
            item = await self._in_channel.get()
            if isinstance(item, _EventBatch):
                assert item.keys is not None
                for key, event in zip(item.keys, item.events):
                    result = await self._handler_func(event, ctx)
                    await self._put_result(key, result)
            else:
                key, event = item
                result = await self._handler_func(event, ctx)
                await self._put_result(key, result)


@asynccontextmanager
async def _task_ctx(
    context_manager: AbstractAsyncContextManager[Ctx] | None = None,
) -> AsyncGenerator[tuple[set[asyncio.Task], Ctx | None], None]:
    tasks: set[asyncio.Task] = set()
    try:
        if context_manager is None:
            yield tasks, None
        else:
            async with context_manager as ctx:
                yield tasks, ctx
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        raise
