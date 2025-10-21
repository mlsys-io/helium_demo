import asyncio
import time

from helium.runtime.protocol import HeliumSystemProfile
from helium.runtime.utils.logger import Logger, init_child_logger
from helium.runtime.utils.loop import AsyncEventLoop
from helium.runtime.utils.pool import AsyncBucket
from helium.runtime.utils.queue import AIOQueue, AsyncQueue, MPQueue


class ProfilingEvent:
    def __init__(self, sender: str, elapsed_time: float) -> None:
        self.sender = sender
        self.elapsed_time = elapsed_time


class ProfilingTaskEvent(ProfilingEvent):
    def __init__(
        self, sender: str, task_id: str, iteration: int, elapsed_time: float
    ) -> None:
        super().__init__(sender, elapsed_time)
        self.task_id = task_id
        self.iteration = iteration


class ProfilingRangeEvent(ProfilingEvent):
    def __init__(self, sender: str, range_id: str, elapsed_time: float) -> None:
        super().__init__(sender, elapsed_time)
        self.range_id = range_id


class WorkerProfiler:
    def __init__(
        self,
        logger: Logger,
        disable_task_profile: bool = False,
        disable_range_profile: bool = False,
        use_mp: bool = False,
    ) -> None:
        self.name = "WorkerProfiler"
        self.logger = init_child_logger(self.name, logger)

        self._task_result_bucket: AsyncBucket[str, tuple[int, float]] | None
        self._task_start: dict[tuple[str, int], tuple[str, float]] | None
        if disable_task_profile:
            self._task_result_bucket = None
            self._task_start = None
        else:
            self._task_result_bucket = AsyncBucket()
            self._task_start = {}

        self._range_result_bucket: AsyncBucket[str, float] | None
        self._range_start: dict[str, tuple[str, float]] | None
        if disable_range_profile:
            self._range_result_bucket = None
            self._range_start = None
        else:
            self._range_result_bucket = AsyncBucket()
            self._range_start = {}
        self._range_cond = asyncio.Condition()

        self._profiler_channel: AsyncQueue[ProfilingEvent] | None
        if disable_task_profile and disable_range_profile:
            self._profiler_channel = None
            self._pulling_loop = None
        else:
            self._profiler_channel = MPQueue() if use_mp else AIOQueue()
            self._pulling_loop = AsyncEventLoop(
                self._pulling_func, self._profiler_channel
            )

    async def start(self) -> None:
        if self._pulling_loop is not None:
            await self._pulling_loop.start()
        self.logger.debug("Profiler started.")

    async def stop(self) -> None:
        if self._pulling_loop is not None:
            await self._pulling_loop.stop()
        self._check_memory_leak()
        self.logger.debug("Profiler terminated.")

    async def _pulling_func(self, event: ProfilingEvent, _) -> None:
        if isinstance(event, ProfilingTaskEvent):
            if self._task_result_bucket is not None:
                await self._task_result_bucket.put(
                    event.task_id, (event.iteration, event.elapsed_time)
                )
        elif isinstance(event, ProfilingRangeEvent):
            if self._range_result_bucket is not None:
                await self._range_result_bucket.put(event.range_id, event.elapsed_time)

    def start_task(self, sender: str, task_id: str, iteration: int) -> None:
        if self._task_start is None:
            return
        key = (task_id, iteration)
        if key in self._task_start:
            raise RuntimeError(f"Task {task_id} iteration {iteration} already started")
        self._task_start[key] = sender, time.perf_counter()

    async def finish_task(self, sender: str, task_id: str, iteration: int) -> None:
        if self._task_start is None:
            return
        assert self._profiler_channel is not None
        key = (task_id, iteration)
        if key not in self._task_start:
            raise RuntimeError(f"Task {task_id} iteration {iteration} not started")
        start_sender, start_time = self._task_start.pop(key)
        if start_sender != sender:
            raise RuntimeError(
                f"Task {task_id} iteration {iteration} started by {start_sender} "
                f"but finished by {sender}"
            )
        elapsed_time = time.perf_counter() - start_time
        event = ProfilingTaskEvent(sender, task_id, iteration, elapsed_time)
        await self._profiler_channel.put(event)

    def start_range(self, sender: str, name: str) -> None:
        if self._range_start is None:
            return
        if name in self._range_start:
            raise RuntimeError(f"Range {name} already started")
        self._range_start[name] = sender, time.perf_counter()

    async def finish_range(self, sender: str, name: str) -> None:
        if self._range_start is None:
            return
        assert self._profiler_channel is not None
        if name not in self._range_start:
            raise RuntimeError(f"Range {name} not started")
        start_sender, start_time = self._range_start.pop(name)
        if start_sender != sender:
            raise RuntimeError(
                f"Range {name} started by {start_sender} but finished by {sender}"
            )
        elapsed_time = time.perf_counter() - start_time
        event = ProfilingRangeEvent(sender, name, elapsed_time)
        await self._profiler_channel.put(event)

        async with self._range_cond:
            # Notify tasks waiting to get range results
            self._range_cond.notify_all()

    async def get_task_result(self, task_id: str) -> list[tuple[int, float]] | None:
        assert not (self._task_start is None or self._task_result_bucket is None)
        keys = list(self._task_start)
        for k in keys:
            if k[0] == task_id:
                del self._task_start[k]
        return await self._task_result_bucket.pop_nowait(task_id)

    async def get_all_task_results(
        self, task_ids: list[str]
    ) -> dict[str, list[tuple[int, float]] | None]:
        results = {}
        for task_id in task_ids:
            results[task_id] = await self.get_task_result(task_id)
        return results

    async def get_range_result(self) -> dict[str, float]:
        assert not (self._range_start is None or self._range_result_bucket is None)
        async with self._range_cond:
            # Wait until all started ranges are finished
            await self._range_cond.wait_for(
                lambda: self._range_start is None or len(self._range_start) == 0
            )
        elapsed_times = self._range_result_bucket.dump_and_clear()
        ret: dict[str, float] = {}
        for range_id, time_list in elapsed_times.items():
            ret[range_id] = sum(time_list)
        return ret

    async def get_profiling_result(self, task_ids: list[str]) -> HeliumSystemProfile:
        ret: HeliumSystemProfile = {}
        if self._task_start is not None:
            ret["task_profile"] = await self.get_all_task_results(task_ids)
        if self._range_start is not None:
            ret["range_profile"] = await self.get_range_result()
        return ret

    def _check_memory_leak(self) -> None:
        task_bucket_leaked = not (
            self._task_result_bucket is None or self._task_result_bucket.empty()
        )
        range_bucket_leaked = not (
            self._range_result_bucket is None or self._range_result_bucket.empty()
        )
        task_start_leaked = not (self._task_start is None or len(self._task_start) == 0)
        range_start_leaked = not (
            self._range_start is None or len(self._range_start) == 0
        )

        if (
            task_bucket_leaked
            or range_bucket_leaked
            or task_start_leaked
            or range_start_leaked
        ):
            self.logger.warning("Memory leak detected in %s.", self.name)

        if task_bucket_leaked:
            assert self._task_result_bucket is not None
            self.logger.debug(
                "Task result bucket's dump: %s", self._task_result_bucket.dump()
            )
        if range_bucket_leaked:
            assert self._range_result_bucket is not None
            self.logger.debug(
                "Range result bucket's dump: %s", self._range_result_bucket.dump()
            )
        if task_start_leaked:
            self.logger.debug("Task start dict's dump: %s", self._task_start)
        if range_start_leaked:
            self.logger.debug("Range start dict's dump: %s", self._range_start)


class RequestProfiler:
    def __init__(self, logger: Logger) -> None:
        self.name = "RequestProfiler"
        self.logger = init_child_logger(self.name, logger)

        self._is_stopped: bool = True
        # range_name -> elapsed_time
        self._range_result: dict[str, float] = {}
        # list(range_name, start_time)
        self._range_starts: list[tuple[str, float]] = []

    def start(self) -> None:
        self._is_stopped = False

    def push_range(self, name: str) -> None:
        if self._is_stopped:
            return
        self._range_starts.append((name, time.perf_counter()))

    def pop_range(self) -> None:
        if self._is_stopped:
            return
        range_starts = self._range_starts
        if len(range_starts) == 0:
            raise RuntimeError("No ranges to pop")
        range_name, start_time = range_starts.pop()
        self._range_result[range_name] = time.perf_counter() - start_time

    def stop(self) -> dict[str, float]:
        if self._is_stopped:
            raise RuntimeError("Profiler has not been started.")
        self._is_stopped = True

        # Check if there are unfinished ranges
        range_starts = self._range_starts
        if len(range_starts) > 0:
            raise RuntimeError(
                f"Unfinished ranges found: {[name for name, _ in range_starts]}"
            )
        range_result = self._range_result

        # Clear the results for the next profiling session
        self._range_result = {}
        self._range_starts = []

        return range_result
