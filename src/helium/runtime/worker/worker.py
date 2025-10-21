import asyncio
import multiprocessing as mp
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import asynccontextmanager
from multiprocessing.synchronize import Event
from typing import Any, Hashable

from helium.common import Message
from helium.runtime.cache_manager import (
    CacheManagerConfig,
    KVCacheClient,
    KVCacheManager,
    PromptCacheManager,
)
from helium.runtime.data import Data
from helium.runtime.functional import FnInput, FnInputBatch
from helium.runtime.llm import LLMServiceConfig
from helium.runtime.profiler import WorkerProfiler
from helium.runtime.protocol import HeliumSystemProfile
from helium.runtime.state_manager import StateManager
from helium.runtime.utils import (
    AIOQueue,
    AsyncPool,
    AsyncQueue,
    MPQueue,
    RcStreamingPool,
)
from helium.runtime.utils.logger import (
    Logger,
    LogLevel,
    init_child_logger,
    log_on_exception_async,
    log_on_exception_async_generator,
)
from helium.runtime.utils.loop import AsyncConcurrentEventLoop, AsyncEventLoop
from helium.runtime.worker.worker_input import (
    ResultPuller,
    ResultStreamIterator,
    WorkerArg,
    WorkerInput,
    WorkerInputBatch,
    WorkerRequest,
    WorkerResponse,
)
from helium.utils import execute_unordered, unique_id


class TaskEnd:
    pass


TASK_END = TaskEnd()


class WorkerManager:
    class Result:
        def __init__(self, resp: WorkerResponse) -> None:
            self.data: Data | None = resp.data
            self.has_next: bool = resp.has_next
            self.new_ticket: int | None = resp.new_ticket

        def add(self, other: "WorkerManager.Result") -> None:
            assert self.data is not None
            data = other.data
            if data is not None:
                self.data += data
            self.has_next = self.has_next and other.has_next
            self.new_ticket = other.new_ticket

        def add_resp(self, resp: WorkerResponse) -> None:
            data = resp.data
            if data is not None:
                assert self.data is not None
                self.data += data
            self.has_next = self.has_next and resp.has_next
            self.new_ticket = resp.new_ticket

        @property
        def is_eos(self) -> bool:
            return self.data is None and self.has_next

        def __repr__(self) -> str:
            return (
                f"{self.__class__.__name__}"
                f"({', '.join(str(k) + '=' + str(v) for k, v in vars(self).items())})"
            )

    class ResultIterator(ResultStreamIterator):
        """Iterator for pulling data from the destination worker in a stream"""

        def __init__(
            self,
            manager: "WorkerManager",
            looping: bool,
            request: WorkerRequest,
        ) -> None:
            """
            Parameters
            ----------
            manager : WorkerManager
                Worker manager
            looping : bool
                Whether the caller is a looping task. If True, the iterator will
                stop at the end of the current iteration. Otherwise, it will continue
                until there is no more iteration.
            request: WorkerRequest
                Initial request for pulling data
            """
            self.manager = manager
            self.out_channel = manager._out_channels[request.dst]
            self.looping = looping
            self.request: WorkerRequest | None = request.copy()
            self.has_next_iteration: bool = True

        def __aiter__(self) -> "WorkerManager.ResultIterator":
            return self

        async def __anext__(self) -> Data | None:
            """Pulls data from the destination worker in a stream"""
            request = self.request
            if request is None:
                raise StopAsyncIteration

            manager = self.manager
            looping = self.looping
            indices = request.indices_as_list()
            out_channel = self.out_channel

            if indices is not None and len(indices) == 0:
                if looping:
                    # Unsub from the current iteration
                    unsub = WorkerRequest.UnsubMode.ITER
                else:
                    # Unsub from all future iterations
                    unsub = WorkerRequest.UnsubMode.TASK
                unsub_request = request.to_unsubscribe(request.ticket, unsub)
                await out_channel.put(unsub_request)
                raise StopAsyncIteration

            # Pull data for the current iteration
            result = await self._request_next(request)

            if result.is_eos:
                # Early commit has occurred
                if looping or not self.has_next_iteration:
                    raise StopAsyncIteration
                # Request the next iteration now
                request = WorkerRequest(
                    request.src,
                    request.dst,
                    request.requesting,
                    request.requested,
                    request.iteration + 1,
                    indices,
                    0,
                )
                result = await self._request_next(request)
                assert not result.is_eos, "Guaranteed by the worker's implementation"

            if result.new_ticket is None:
                # No more data in the current iteration
                if looping or not self.has_next_iteration:
                    # Last data in the stream
                    self.request = None
                    return result.data
                # Proceed to the next iteration
                new_ticket = 0
                next_iteration = request.iteration + 1
            else:
                # Proceed to the next ticket in the same iteration
                new_ticket = result.new_ticket
                next_iteration = request.iteration
            next_indices = manager._filter_indices(indices, result)

            self.request = WorkerRequest(
                request.src,
                request.dst,
                request.requesting,
                request.requested,
                next_iteration,
                next_indices,
                new_ticket,
            )
            return result.data

        async def _request_next(self, request: WorkerRequest) -> "WorkerManager.Result":
            # get_debug_logger().debug("Pulling data with request: %s", request)
            await self.out_channel.put(request)
            result = await self.manager._task_results.pop(request.message_id)
            if not result.is_eos:
                # Only update if the result is not EOS.
                # It is guaranteed by the worker's implementation that at least one
                # non-EOS result is returned before an EOS, and self.has_next_iteration
                # must have reflected that.
                assert self.has_next_iteration or not result.has_next
                self.has_next_iteration = result.has_next
            # get_debug_logger().debug(
            #     "Pulled data for request: %s (result=%s) (has_next_iteration=%s)",
            #     request,
            #     result,
            #     self.has_next_iteration,
            # )
            return result

    def __init__(
        self,
        llm_service_configs: list[LLMServiceConfig],
        cache_manager_config: CacheManagerConfig,
        name: str = "Manager",
        logger: Logger | None = None,
        log_level: LogLevel | None = None,
    ) -> None:
        self.name = name
        self.logger: Logger = init_child_logger(self.name, logger, log_level)
        # For data from workers
        self._in_channels: dict[str, AsyncQueue[WorkerResponse]] = {}
        # For pull requests from workers
        self._out_channels: dict[str, AsyncQueue[WorkerRequest]] = {}
        # For caching data from workers
        # data is associated with a flag indicating whether it has a next iteration.
        # (requesting, requested, iteration) -> (result)
        self._task_results: AsyncPool[tuple[str, str, int], WorkerManager.Result] = (
            AsyncPool()
        )
        self._use_mp: bool = False

        # Tasks pulling data from in_channels
        self._pulling_tasks: dict[str, asyncio.Task] = {}
        self._is_closed: bool = False

        # Profiler for benchmarking
        self._profiler: WorkerProfiler | None = None

        # State manager for tracking job states
        self._state_manager: StateManager

        self._kv_cache_manager: KVCacheManager | None
        if cache_manager_config.enable_proactive_kv_cache:
            # Initialize KV cache manager
            if cache_manager_config.kv_cache_manager is None:
                use_lmcache = any(
                    cfg.args.get("kv_transfer_config") is not None
                    for cfg in llm_service_configs
                )
                self._kv_cache_manager = KVCacheManager(
                    cache_manager_config.kv_cache_config_file,
                    use_lmcache=use_lmcache,
                    logger=logger,
                    log_level=log_level,
                )
                self._external_kv_cache_manager = False
            else:
                self._kv_cache_manager = cache_manager_config.kv_cache_manager
                self._external_kv_cache_manager = True
        else:
            self._kv_cache_manager = None
            self._external_kv_cache_manager = False

        self._prompt_cache_manager: PromptCacheManager | None
        if cache_manager_config.enable_prompt_cache:
            # Initialize prompt cache manager
            if cache_manager_config.prompt_cache_manager is None:
                self._prompt_cache_manager = PromptCacheManager()
                self._external_prompt_cache_manager = False
            else:
                self._prompt_cache_manager = cache_manager_config.prompt_cache_manager
                self._external_prompt_cache_manager = True
        else:
            self._prompt_cache_manager = None
            self._external_prompt_cache_manager = False

    @property
    def kv_cache_manager(self) -> KVCacheManager | None:
        return self._kv_cache_manager

    @property
    def prompt_cache_manager(self) -> PromptCacheManager | None:
        return self._prompt_cache_manager

    def get_kv_cache_client(self) -> KVCacheClient:
        if self._kv_cache_manager is None:
            raise ValueError("KV cache manager is not initialized")
        return self._kv_cache_manager.controller_client

    def register(self, worker: "Worker"):
        """Registers the worker so that it can communicate with other registered
        workers
        """
        if isinstance(worker, MPWorker) and not self._use_mp:
            self._use_mp = True
        elif self._use_mp and isinstance(worker, AIOWorker):
            raise ValueError(
                "Cannot use asynchronous worker in multiprocessing context"
            )
        elif self._profiler is not None:
            raise RuntimeError("Profiler has already been started")

        self._in_channels[worker.name] = worker.in_channel
        self._out_channels[worker.name] = worker.out_channel

    def start(self, worker: "Worker") -> None:
        """Starts the manager for the worker, allowing it to communicate with other
        workers

        This method should be called by all registered workers before they start.
        """
        if not (self._kv_cache_manager is None or self._external_kv_cache_manager):
            self._kv_cache_manager.start()

        if worker.name not in self._pulling_tasks:
            self._pulling_tasks[worker.name] = asyncio.create_task(
                self._pulling_loop(worker.name)
            )
            self._state_manager.start(worker.name)

    async def start_state_manager(self) -> None:
        """Starts the state manager

        This should be called only after registering all the workers and before
        starting the workers.
        """
        if hasattr(self, "_state_manager"):
            self.logger.warning("State manager has already been initialized")
            return
        self._state_manager = await StateManager.create_and_init(
            list(self._in_channels), use_mp=self._use_mp, logger=self.logger
        )

    def init_profiler(self) -> None:
        if self._profiler is not None:
            self.logger.warning("Profiler has already been initialized")
            return
        self._profiler = WorkerProfiler(
            self.logger,
            disable_task_profile=True,
            disable_range_profile=False,
            use_mp=self._use_mp,
        )

    async def start_profiling(self) -> None:
        if self._profiler is None:
            self.logger.warning("Profiler has not been initialized")
            return
        await self._profiler.start()

    @log_on_exception_async(ignore=[asyncio.CancelledError])
    async def _pulling_loop(self, worker_name: str) -> None:
        """Pulls data from in_channels and puts them into task_results"""
        while True:
            resp = await self._in_channels[worker_name].get()
            resp_id = resp.id
            result = await self._task_results.get_nowait(resp_id)
            if result is None:
                result = WorkerManager.Result(resp)
                await self._task_results.put(resp_id, result)
            else:
                result.add_resp(resp)

    def _filter_indices(
        self, indices: list[int] | None, result: "WorkerManager.Result"
    ) -> list[int] | None:
        if indices is None:
            return None
        if result.data is None:
            return indices
        return [i for i in indices if i not in result.data.indices]

    def result_iterator(
        self, worker_name: str, looping: bool, request: WorkerRequest
    ) -> ResultStreamIterator:
        assert worker_name == request.src, f"{worker_name} != {request.src}"
        return self.__class__.ResultIterator(self, looping, request)

    async def send_result(self, worker_name: str, resp: WorkerResponse) -> None:
        """Sends data to the destination worker"""
        assert worker_name == resp.src
        # get_debug_logger().debug("Sending response: %s", resp)
        await self._in_channels[resp.dst].put(resp)

    async def unsubscribe(self, worker_name: str, unsub_request: WorkerRequest) -> None:
        """Unsubscribes from the given request"""
        assert worker_name == unsub_request.src
        assert unsub_request.unsubscribe is not None
        # get_debug_logger().debug("Unsubscribing with request: %s", unsub_request)
        await self._out_channels[unsub_request.dst].put(unsub_request)

    async def close(self) -> None:
        """Closes the manager and sends a termination signal to all the workers

        It does not wait for the workers to finish.
        """
        if not self._is_closed:
            self._is_closed = True
            for in_channel in self._in_channels.values():
                in_channel.close()
            for out_channel in self._out_channels.values():
                out_channel.close()
            for task in self._pulling_tasks.values():
                await task
            if not (self._kv_cache_manager is None or self._external_kv_cache_manager):
                self._kv_cache_manager.stop()
            if self._profiler is not None:
                await self._profiler.stop()
            if hasattr(self, "_state_manager"):
                await self._state_manager.close()

    def start_profiling_task(
        self, worker_name: str, op_id: str, iteration: int
    ) -> None:
        if self._profiler is None:
            return
        self._profiler.start_task(worker_name, op_id, iteration)

    async def stop_profiling_task(
        self, worker_name: str, op_id: str, iteration: int
    ) -> None:
        if self._profiler is None:
            return
        await self._profiler.finish_task(worker_name, op_id, iteration)

    def start_profiling_range(self, worker_name: str, range_id: str) -> None:
        if self._profiler is None:
            return
        self._profiler.start_range(worker_name, range_id)

    async def stop_profiling_range(self, worker_name: str, range_id: str) -> None:
        if self._profiler is None:
            return
        await self._profiler.finish_range(worker_name, range_id)

    async def get_profiling_results(self, op_ids: list[str]) -> HeliumSystemProfile:
        if self._profiler is None:
            return {}
        return await self._profiler.get_profiling_result(op_ids)

    def track_job_state(self, job_id: str) -> None:
        """Tracks the job state for the given job ID"""
        self._state_manager.track_job(job_id)

    def untrack_job_state(self, job_id: str) -> None:
        """Untracks the job state for the given job ID"""
        self._state_manager.untrack_job(job_id)

    async def get_job_state(
        self, worker_name: str, job_id: str, scope: str, key: str
    ) -> Any:
        return await self._state_manager.get_state(worker_name, job_id, scope, key)

    async def pop_job_state(
        self, worker_name: str, job_id: str, scope: str, key: str
    ) -> Any:
        return await self._state_manager.pop_state(worker_name, job_id, scope, key)

    async def set_job_state(
        self, worker_name: str, job_id: str, scope: str, key: str, value: Any
    ) -> None:
        await self._state_manager.set_state(worker_name, job_id, scope, key, value)

    async def dump_job_state(
        self, worker_name: str, job_id: str, scope: str | None
    ) -> dict[str, Any]:
        return await self._state_manager.dump_state(worker_name, job_id, scope)

    async def clear_job_state(
        self, worker_name: str, job_id: str, scope: str | None
    ) -> None:
        await self._state_manager.clear_state(worker_name, job_id, scope)

    def query_cache(
        self, inp: WorkerInput, key: Hashable
    ) -> str | list[Message] | None:
        if self._prompt_cache_manager is None:
            return None
        return self._prompt_cache_manager.query(inp, key)

    def batch_query_cache(
        self, inp: WorkerInput, keys: Sequence[Hashable]
    ) -> list[str | None] | list[list[Message] | None]:
        if self._prompt_cache_manager is None:
            return [None] * len(keys)  # type: ignore[return-value]
        return self._prompt_cache_manager.batch_query(inp, keys)

    def store_cache(
        self,
        inp: WorkerInput,
        key: Hashable,
        value: str | list[Message],
        overwrite: bool = False,
    ) -> None:
        if self._prompt_cache_manager is None:
            return
        self._prompt_cache_manager.store(inp, key, value, overwrite)

    def batch_store_cache(
        self,
        inp: WorkerInput,
        batch: dict[Hashable, str] | dict[Hashable, list[Message]],
        overwrite: bool = False,
    ) -> None:
        if self._prompt_cache_manager is None:
            return
        self._prompt_cache_manager.batch_store_cache(inp, batch, overwrite)


class InputRegistry:
    def __init__(self, prefix: str | None = None) -> None:
        self.prefix = prefix or ""
        self._worker_inp_counter: int = 0
        self._fn_inp_registry: dict[type[FnInput], int] = {}
        self._pregistererd_task_ids: set[str] = set()

    def register(self, inp: WorkerInput) -> None:
        """Registers the worker input and assigns a task ID to it

        Returns immediately if the input has been preregistered.
        """
        if inp.is_registered():
            if inp.task_id not in self._pregistererd_task_ids:
                raise ValueError("Task ID has already been registered")
            self._pregistererd_task_ids.remove(inp.task_id)
            return
        self._register_unchecked(inp)

    def preregister(self, inp_cls: type[WorkerInput]) -> str:
        """Preregisters the worker input of the given class

        This is necessary because some Ops, like FutureOp, need to obtain task IDs
        for the worker inputs of the future Ops they are referring to before the
        worker inputs are actually created and allocated a task ID.

        See how the HeliumScheduler handles Op preregistration for more detail.

        Parameters
        ----------
        inp_cls : type[WorkerInput]
            The class of the worker input to be preregistered

        Returns
        -------
        str
            The task ID of the preregistered worker input
        """
        task_id = (
            self._register_fn_input(inp_cls)
            if issubclass(inp_cls, FnInput)
            else self._register_worker_input()
        )
        self._pregistererd_task_ids.add(task_id)
        return task_id

    def _register_unchecked(self, inp: WorkerInput) -> None:
        """Registers the worker input without checking if it has been registered/
        preregistered

        It modifies the `task_id` of `inp`.

        Parameters
        ----------
        inp : WorkerInput
            The worker input to be registered
        """
        task_id = (
            self._register_fn_input(type(inp))
            if isinstance(inp, FnInput)
            else self._register_worker_input()
        )
        inp.task_id = task_id

    def _register_worker_input(self) -> str:
        """Allocates a task ID for a worker input

        Returns
        -------
        str
            The allocated task ID
        """
        task_id = f"{self.prefix}-{self._worker_inp_counter}"
        self._worker_inp_counter += 1
        return task_id

    def _register_fn_input(self, fn_inp_cls: type[FnInput]) -> str:
        """Allocates a task ID for a function input of the given class

        It keeps track of each FnInput class separately for understandability of
        the task ID.

        Parameters
        ----------
        fn_inp_cls : type[FnInput]
            The class of the function input

        Returns
        -------
        str
            The allocated task ID
        """
        if fn_inp_cls not in self._fn_inp_registry:
            self._fn_inp_registry[fn_inp_cls] = 0
        fn_no = self._fn_inp_registry[fn_inp_cls]
        self._fn_inp_registry[fn_inp_cls] += 1
        task_id = f"{self.prefix}-{fn_inp_cls.NAME}-{fn_no}"
        return task_id


WorkerOutput = Data | None | TaskEnd


class Worker(ABC, ResultPuller):
    def __init__(
        self,
        manager: WorkerManager,
        in_channel: AsyncQueue[WorkerResponse],
        out_channel: AsyncQueue[WorkerRequest],
        name: str | None = None,
        logger: Logger | None = None,
        log_level: LogLevel | None = None,
    ):
        self.name: str = "wkr-" + (unique_id() if name is None else name)
        self.logger: Logger = init_child_logger(self.name, logger, log_level)
        self.in_channel: AsyncQueue[WorkerResponse] = (
            in_channel  # For data from workers
        )
        self.out_channel: AsyncQueue[WorkerRequest] = (
            out_channel  # For pull requests from workers
        )
        self.input_registry: InputRegistry = InputRegistry(prefix=self.name)
        self.manager: WorkerManager = manager

        self.manager.register(self)

    @log_on_exception_async_generator(ignore=[asyncio.CancelledError])
    async def execute(
        self, inputs: WorkerInputBatch
    ) -> AsyncGenerator[Sequence[tuple[WorkerInput, WorkerOutput]], None]:
        self.logger.debug("Executing tasks %s", [inp.task_id for inp in inputs])

        tasks = {asyncio.create_task(inp.resolve(self)) for inp in inputs}
        async for done in execute_unordered(tasks):
            resolved = [task.result() for task in done]
            alive_inputs: list[WorkerInput] = []
            dead_inputs: list[WorkerInput] = []
            for inp in resolved:
                if inp.is_dead():
                    dead_inputs.append(inp)
                else:
                    alive_inputs.append(inp)

            # Yield dead inputs
            yield [(inp, None) for inp in dead_inputs]

            # Allow streaming partially completed results
            to_execute = inputs.wrap(alive_inputs)
            try:
                async for inp_out_list in self._execute(to_execute):
                    yield [(inp, out) for inp, out in inp_out_list]
                yield [(inp, TASK_END) for inp in alive_inputs]
            except Exception as e:
                err_inputs = to_execute.inputs
                await self._handle_error(err_inputs, e)
                yield [(inp, None) for inp in err_inputs]

    def result_iterator(
        self, worker_name: str, looping: bool, request: WorkerRequest
    ) -> ResultStreamIterator:
        return self.manager.result_iterator(worker_name, looping, request)

    async def unsubscribe(self, worker_name: str, unsub_request: WorkerRequest) -> None:
        return await self.manager.unsubscribe(worker_name, unsub_request)

    def start_profiling_task(
        self, worker_name: str, op_id: str, iteration: int
    ) -> None:
        self.manager.start_profiling_task(worker_name, op_id, iteration)

    async def stop_profiling_task(
        self, worker_name: str, op_id: str, iteration: int
    ) -> None:
        await self.manager.stop_profiling_task(worker_name, op_id, iteration)

    def start_profiling_range(self, worker_name: str, range_id: str) -> None:
        self.manager.start_profiling_range(worker_name, range_id)

    async def stop_profiling_range(self, worker_name: str, range_id: str) -> None:
        await self.manager.stop_profiling_range(worker_name, range_id)

    async def get_cached_results(
        self, inp: WorkerInput, keys: Sequence[Hashable]
    ) -> list[str | None] | list[list[Message] | None]:
        return self.manager.batch_query_cache(inp, keys)

    async def cache_results(
        self,
        inp: WorkerInput,
        batch: dict[Hashable, str] | dict[Hashable, list[Message]],
        overwrite: bool = False,
    ) -> None:
        self.manager.batch_store_cache(inp, batch, overwrite)

    def preregister(self, inp_cls: type[WorkerInput]) -> str:
        return self.input_registry.preregister(inp_cls)

    @abstractmethod
    async def start(self) -> None:
        pass

    @abstractmethod
    async def add_task(
        self,
        inputs: WorkerInputBatch | FnInputBatch,
        puller: ResultPuller | None = None,
    ) -> list[WorkerArg]:
        pass

    @abstractmethod
    def _execute(
        self, inputs: WorkerInputBatch
    ) -> AsyncIterator[Sequence[tuple[WorkerInput, WorkerOutput]]]:
        pass

    @abstractmethod
    async def join(self) -> None:
        pass

    async def _handle_error(
        self, err_inputs: Sequence[WorkerInput], e: BaseException
    ) -> None:
        task_ids: list[str] = []
        for inp in err_inputs:
            inp.mark_error()
            task_ids.append(inp.task_id)
        self.logger.exception("Error while executing tasks: %s", task_ids)
        err_inp_key = ",".join(task_ids)
        await self.manager.set_job_state(
            self.name, err_inputs[0].request_id, "error", err_inp_key, e
        )


class AsyncExecutor(Worker, ABC):
    class ResultPool:
        """
        Pool for storing results from tasks

        It additionally tracks results across iterations and handle unsubscription
        requests.
        """

        K = tuple[str, int]  # (task_id, iteration)
        V = tuple[Data | None, bool]  # (data, has_next)

        class TrackerEntry:
            def __init__(self) -> None:
                self.task_unsub_count: int = 0
                self.iter_unsub_count: dict[int, int] = {}
                self.last_iteration: int | None = None

            def unsub_task(self) -> None:
                self.task_unsub_count += 1

            def unsub_iter(self, iteration) -> None:
                if iteration not in self.iter_unsub_count:
                    self.iter_unsub_count[iteration] = 0
                self.iter_unsub_count[iteration] += 1

            def get_unsub_count(self, iteration: int) -> int:
                return self.task_unsub_count + self.iter_unsub_count.get(iteration, 0)

            def set_last_iter(self, iteration: int) -> None:
                if self.last_iteration is None:
                    self.last_iteration = iteration
                elif iteration != self.last_iteration:
                    raise ValueError("Inconsistent last iteration")

            def __repr__(self) -> str:
                return (
                    f"{self.__class__.__name__}"
                    f"({', '.join(str(k) + '=' + str(v) for k, v in vars(self).items())})"
                )

        def __init__(self, worker_name: str, manager: WorkerManager) -> None:
            self._worker_name = worker_name
            self._manager = manager
            self._pool: RcStreamingPool[
                AsyncExecutor.ResultPool.K, AsyncExecutor.ResultPool.V
            ] = RcStreamingPool(merge_func=self.__class__._merge_result)
            self._task_tracker: dict[str, AsyncExecutor.ResultPool.TrackerEntry] = {}

        def track(self, inp: WorkerInput) -> None:
            """Tracks the worker input

            This should always be called before starting processing the input.

            Parameters
            ----------
            inp : WorkerInput
                Worker input to track
            """
            if inp.task_id in self._task_tracker:
                raise ValueError(f"Task {inp.task_id} is already being tracked")
            self._task_tracker[inp.task_id] = self.__class__.TrackerEntry()

        def _untrack(self, task_id: str) -> None:
            if task_id not in self._task_tracker:
                raise ValueError(f"Task {task_id} is not being tracked")
            self._task_tracker.pop(task_id)

        async def put(self, inp: WorkerInput, data: Data | None) -> None:
            # get_debug_logger().debug(
            #     "Putting %s (data=%s)",
            #     (inp.task_id, inp.iteration),
            #     None if data is None else data.data,
            # )
            task_id = inp.task_id
            if task_id not in self._task_tracker:
                raise ValueError(f"Task {task_id} is not being tracked")

            result_id = task_id, inp.iteration
            has_next = inp.looping and data is not None
            tracker_entry = self._task_tracker[task_id]
            unsub_count = tracker_entry.get_unsub_count(inp.iteration)
            ref_count = inp.ref_count - unsub_count
            await self._pool.put(result_id, (data, has_next), ref_count)

            if data is None:
                # Dead data received. Mark the current iteration as the last iteration.
                tracker_entry.set_last_iter(inp.iteration)

        async def pop(
            self, task_id: str, iteration: int, ticket: int
        ) -> tuple[int | None, V] | None:
            result_id = task_id, iteration
            return await self._pool.pop(result_id, ticket)

        async def pop_all(self, task_id: str, iteration: int) -> V:
            result_id = task_id, iteration
            return await self._pool.pop_all(result_id)

        async def commit(self, inp: WorkerInput) -> None:
            looping = inp.looping
            task_id = inp.task_id
            iteration = inp.iteration
            # get_debug_logger().debug("Committing %s (iteration=%s)", task_id, iteration)
            await self._manager.stop_profiling_task(
                self._worker_name, inp.op_id, iteration
            )
            await self._pool.commit((task_id, iteration))
            if not looping or iteration == self._task_tracker[task_id].last_iteration:
                # iteration is set only when looping
                self._untrack(task_id)

        async def unsubscribe(
            self,
            task_id: str,
            iteration: int,
            ticket: int,
            unsub: WorkerRequest.UnsubMode,
        ) -> None:
            # Unsub from this iteration
            await self._pool.unsubscribe((task_id, iteration), ticket)

            if task_id not in self._task_tracker:
                # Task has been untracked.
                # There are no more iterations to unsub from.
                return

            unsub_entry = self._task_tracker[task_id]
            if unsub == WorkerRequest.UnsubMode.ITER:
                unsub_entry.unsub_iter(iteration)
                return
            unsub_entry.unsub_task()

            unsub_result_ids = [
                (tid, it)
                for tid, it in self._pool.keys()
                if tid == task_id and it > iteration
            ]
            # Unsub from the first item of each subsequent iteration
            for result_id in unsub_result_ids:
                await self._pool.unsubscribe(result_id, 0)

        def empty(self) -> bool:
            return self._pool.empty() and len(self._task_tracker) == 0

        def dump(self) -> tuple[dict[K, list[V]], dict[str, TrackerEntry]]:
            return self._pool.dump(), self._task_tracker

        @staticmethod
        def _merge_result(x: V, y: V) -> V:
            x_data, x_has_next = x
            y_data, y_has_next = y
            if x_data is None:
                ret_data = y_data
            elif y_data is None:
                ret_data = x_data
            else:
                ret_data = x_data + y_data
            return (ret_data, x_has_next and y_has_next)

    def __init__(
        self,
        manager: WorkerManager,
        fn_inp_channel: AsyncQueue[FnInputBatch],
        in_channel: AsyncQueue[WorkerResponse],
        out_channel: AsyncQueue[WorkerRequest],
        name: str | None = None,
        logger: Logger | None = None,
        log_level: LogLevel | None = None,
    ) -> None:
        super().__init__(
            manager=manager,
            in_channel=in_channel,
            out_channel=out_channel,
            name=name,
            logger=logger,
            log_level=log_level,
        )

        self._result_pool: AsyncExecutor.ResultPool = AsyncExecutor.ResultPool(
            self.name, self.manager
        )

        self._fn_handling_loop = AsyncEventLoop(
            self._fn_handling_func,
            in_channel=fn_inp_channel,
            context_manager=self._fn_handling_ctx(),
        )
        self._pull_request_loop = AsyncConcurrentEventLoop(
            self._handle_pull_request,
            in_channel=out_channel,
            context_manager=self._pull_request_ctx(),
        )
        self._is_started: bool = False

    async def _start(self) -> bool:
        to_start = not self._is_started
        if to_start:
            self.manager.start(self)
            await self._fn_handling_loop.start()
            await self._pull_request_loop.start()
            self._is_started = True
        return to_start

    async def _join(self) -> bool:
        to_join = self._is_started
        if to_join:
            await self._fn_handling_loop.join()
            await self._pull_request_loop.join()
            self._is_started = False
        return to_join

    @asynccontextmanager
    async def _fn_handling_ctx(self) -> AsyncGenerator[set[asyncio.Task], None]:
        tasks: set[asyncio.Task] = set()
        try:
            yield tasks
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            raise

    async def _fn_handling_func(
        self, fn_inputs: FnInputBatch, tasks: set[asyncio.Task] | None
    ) -> None:
        assert tasks is not None
        self.logger.debug(
            "Executing tasks %s", [fn_inp.task_id for fn_inp in fn_inputs]
        )
        for fn_inp in fn_inputs:
            self._result_pool.track(fn_inp)
            task = asyncio.create_task(self._handle_fn(fn_inp))
            tasks.add(task)
            task.add_done_callback(tasks.discard)

    @log_on_exception_async(ignore=[asyncio.CancelledError])
    async def _handle_fn(self, fn_inp: FnInput) -> None:
        try:
            async for data in fn_inp.run(self):
                await self._result_pool.put(fn_inp, data)
            is_dead = False
        except Exception as e:
            await self._handle_error([fn_inp], e)
            await self._result_pool.put(fn_inp, None)
            is_dead = True

        await self._result_pool.commit(fn_inp)
        # Handling looping tasks.
        if fn_inp.looping:
            while not is_dead:
                is_dead = False
                try:
                    async for data in fn_inp.run(self):
                        await self._result_pool.put(fn_inp, data)
                        if data is None:
                            is_dead = True
                            break
                except Exception as e:
                    await self._handle_error([fn_inp], e)
                    await self._result_pool.put(fn_inp, None)
                    is_dead = True
                await self._result_pool.commit(fn_inp)
            await fn_inp.unsubscribe(self)

    @asynccontextmanager
    async def _pull_request_ctx(self) -> AsyncGenerator[None, None]:
        try:
            yield None
        except asyncio.CancelledError:
            await self._fn_handling_loop.stop()
            raise

    @log_on_exception_async(ignore=[asyncio.CancelledError])
    async def _handle_pull_request(self, request: WorkerRequest, _) -> None:
        if request.unsubscribe is not None:
            await self._result_pool.unsubscribe(
                request.requested,
                request.iteration,
                request.ticket,
                request.unsubscribe,
            )
            return

        popped = await self._result_pool.pop(
            request.requested, request.iteration, request.ticket
        )
        if popped is None:
            # Late commit occurred
            # get_debug_logger().debug("Late commit occurred for %s", request)
            resp = WorkerResponse.eos(request)
        else:
            new_ticket, (data, has_next) = popped
            indices = request.indices_as_list()
            if data is not None and indices is not None:
                # Send only data with requested indices
                data = data.get_by_indices(indices, uncheck=True)
            resp = WorkerResponse.create(request, data, has_next, new_ticket)
        await self.manager.send_result(self.name, resp)

    def _check_memory_leak(self) -> None:
        if not self._result_pool.empty():
            self.logger.warning("Memory leak detected in %s.", self.name)
            self.logger.debug("Result pool's dump: %s", self._result_pool.dump())
        if not self.manager._task_results.empty():
            self.logger.warning(
                "Memory leak detected in %s's manager.",
                self.name,
            )


class AsyncWorker(AsyncExecutor, ABC):
    def __init__(
        self,
        manager: WorkerManager,
        worker_inp_channel: AsyncQueue[WorkerInputBatch],
        fn_inp_channel: AsyncQueue[FnInputBatch],
        in_channel: AsyncQueue[WorkerResponse],
        out_channel: AsyncQueue[WorkerRequest],
        name: str | None = None,
        logger: Logger | None = None,
        log_level: LogLevel | None = None,
    ) -> None:
        super().__init__(
            manager=manager,
            fn_inp_channel=fn_inp_channel,
            in_channel=in_channel,
            out_channel=out_channel,
            name=name,
            logger=logger,
            log_level=log_level,
        )

        self._worker_inp_channel: AsyncQueue[WorkerInputBatch] = (
            worker_inp_channel  # For worker input from main,
        )
        self._worker_task: asyncio.Task | None = None

    async def add_task(
        self,
        inputs: WorkerInputBatch | FnInputBatch,
        puller: ResultPuller | None = None,
    ) -> list[WorkerArg]:
        # Prepare inputs
        for inp in inputs:
            if inp.is_eager:
                if inp.looping:
                    raise ValueError("Cannot execute looping tasks eagerly")
                # The results will be resolved only by this call
                inp.ref_count = 1
            self.input_registry.register(inp)

        # Dispatch inputs for execution
        if isinstance(inputs, FnInputBatch):
            await self._fn_handling_loop.add_event(inputs)
        else:
            await self._worker_inp_channel.put(inputs)

        if puller is None:
            puller = self

        # Create WorkerArgs
        out: list[WorkerArg] = []
        resolving_tasks: list[asyncio.Task] = []
        for inp in inputs:
            arg = WorkerArg(
                op_id=inp.op_id, src_worker=self.name, src_task_id=inp.task_id
            )
            out.append(arg)

            if inp.is_eager:
                # Resolve eagerly executed tasks
                resolving_tasks.append(
                    asyncio.create_task(
                        arg.resolve(
                            puller,
                            f"{arg.src_task_id}-eager",
                            looping=False,
                            eager=True,
                        )
                    )
                )

        if len(resolving_tasks) > 0:
            # Wait for all eagerly executed tasks to be resolved
            await asyncio.gather(*resolving_tasks)

        return out

    def _terminate_worker_loop(self, _: Any) -> None:
        self._worker_inp_channel.close()

    async def _start(self) -> bool:
        to_start = await super()._start()
        if to_start:
            self._worker_task = asyncio.create_task(self._worker_loop())
            self._pull_request_loop.loop_task.add_done_callback(
                self._terminate_worker_loop
            )
        return to_start

    async def _join(self) -> bool:
        to_join = await super()._join()
        if to_join:
            assert self._worker_task is not None
            await self._worker_task
            self._worker_task = None
        return to_join

    @log_on_exception_async(ignore=[asyncio.CancelledError])
    async def _worker_loop(self) -> None:
        looping_tasks: list[WorkerInputBatch] = []
        while True:
            if len(looping_tasks) == 0:
                worker_inputs = await self._worker_inp_channel.get()
                for inp in worker_inputs:
                    self._result_pool.track(inp)
            else:
                # Looping tasks have already been tracked
                worker_inputs = looping_tasks.pop(0)
                looping_tasks = []
            looping_inputs = []
            committed_inputs = set()
            async for results in self.execute(worker_inputs):
                for inp, data in results:
                    # Handle looping tasks.
                    is_alive = data is not None
                    if inp.looping and is_alive:
                        looping_inputs.append(inp)
                    task_end = isinstance(data, TaskEnd)
                    if not task_end:
                        await self._result_pool.put(inp, data)
                    if data is None or task_end:
                        committed_inputs.add(inp)
                        await self._result_pool.commit(inp)
                        if inp.looping and inp.is_dead():
                            await inp.unsubscribe(self)
            assert committed_inputs == set(worker_inputs), (
                "Not all inputs were committed: "
                f"Expect {set(worker_inputs)}, got {committed_inputs}"
            )

            if len(looping_inputs) > 0:
                # TODO: Support looping for batches of interdependent tasks
                looping_tasks.append(worker_inputs.wrap(looping_inputs))


class AIOWorker(AsyncWorker, ABC):
    def __init__(
        self,
        manager: WorkerManager,
        name: str | None = None,
        logger: Logger | None = None,
        log_level: LogLevel | None = None,
    ):
        super().__init__(
            manager=manager,
            worker_inp_channel=AIOQueue(),
            fn_inp_channel=AIOQueue(),
            in_channel=AIOQueue(),
            out_channel=AIOQueue(),
            name=name,
            logger=logger,
            log_level=log_level,
        )

    async def start(self) -> None:
        await self._start()
        self.logger.debug("Worker %s started.", self.name)

    async def join(self) -> None:
        await self._join()
        self._check_memory_leak()
        self.logger.debug("Worker %s terminated.", self.name)


class MPWorker(AsyncWorker, ABC):
    def __init__(
        self,
        manager: WorkerManager,
        name: str | None = None,
        logger: Logger | None = None,
        log_level: LogLevel | None = None,
    ):
        super().__init__(
            manager=manager,
            worker_inp_channel=MPQueue(),
            fn_inp_channel=MPQueue(),
            in_channel=MPQueue(),
            out_channel=MPQueue(),
            name=name,
            logger=logger,
            log_level=log_level,
        )
        self._main_proc: mp.Process | None = None

    async def start(self) -> None:
        if self._main_proc is None:
            ready_event = mp.Event()
            self._main_proc = mp.Process(
                target=lambda: asyncio.run(self._run_main_proc(ready_event))
            )
            self._main_proc.start()
            ready_event.wait()

    async def _run_main_proc(self, ready_event: Event) -> None:
        await self._start()
        ready_event.set()  # Signals the parent process that it is ready
        self.manager.logger = init_child_logger("Worker", self.logger)
        self.logger.debug("Worker %s started on process %s.", self.name, os.getpid())
        # Wait for all the running tasks to terminate
        await self._join()
        self._check_memory_leak()

    async def join(self) -> None:
        if self._main_proc is not None:
            self._main_proc.join()
        self.logger.debug("Worker %s terminated.", self.name)


DefaultAsyncWorker = AIOWorker
