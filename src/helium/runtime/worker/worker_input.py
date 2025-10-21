import asyncio
import copy
import enum
import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Iterator, Sequence
from contextlib import asynccontextmanager
from typing import Generic, Hashable, Self, TypeVar, cast

from helium.common import Message
from helium.runtime.data import Data, DataType, MessageList
from helium.runtime.utils.queue import AIOQueue
from helium.utils import utils


class WorkerRequest:
    src: str
    """Sender's name"""
    dst: str
    """Destination's name"""
    requesting: str
    """Requesting task ID"""
    requested: str
    """Requested task ID"""
    iteration: int
    """Iteration number"""
    indices: Iterable[int] | slice | None
    """Requested data indices"""
    ticket: int
    """Ticket number for streaming data"""
    unsubscribe: "WorkerRequest.UnsubMode | None"
    """Unsubscribe from the task or iteration"""

    class UnsubMode(enum.Enum):
        TASK = enum.auto()
        ITER = enum.auto()

    def __init__(
        self,
        src: str,
        dst: str,
        requesting: str,
        requested: str,
        iteration: int,
        indices: Iterable[int] | slice | None = None,
        ticket: int = 0,
        unsubscribe: "WorkerRequest.UnsubMode | None" = None,
    ):
        self.src = src
        self.dst = dst
        self.requesting = requesting
        self.requested = requested
        self.iteration = iteration
        self.indices = indices
        self.ticket = ticket
        self.unsubscribe = unsubscribe

    @property
    def message_id(self) -> tuple[str, str, int]:
        return self.requesting, self.requested, self.iteration

    def indices_as_list(self) -> list[int] | None:
        return None if self.indices is None else utils.indices_to_list(self.indices)

    def copy(self) -> "WorkerRequest":
        return WorkerRequest(
            src=self.src,
            dst=self.dst,
            requesting=self.requesting,
            requested=self.requested,
            iteration=self.iteration,
            indices=self.indices,
            ticket=self.ticket,
            unsubscribe=self.unsubscribe,
        )

    def to_unsubscribe(
        self, ticket: int, unsub: "WorkerRequest.UnsubMode"
    ) -> "WorkerRequest":
        return WorkerRequest(
            src=self.src,
            dst=self.dst,
            requesting=self.requesting,
            requested=self.requested,
            iteration=self.iteration,
            indices=self.indices,
            ticket=ticket,
            unsubscribe=unsub,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"({', '.join(str(k) + '=' + str(v) for k, v in vars(self).items())})"
        )


class WorkerResponse:
    src: str
    """Sender's name"""
    dst: str
    """Destination's name"""
    requesting: str
    """Requesting task ID"""
    requested: str
    """Requested task ID"""
    data: Data | None
    """Data payload"""
    iteration: int
    """Iteration number"""
    has_next: bool
    """Whether there are more responses"""
    new_ticket: int | None
    """New ticket number for streaming data"""

    def __init__(
        self,
        src: str,
        dst: str,
        requesting: str,
        requested: str,
        data: Data | None,
        iteration: int,
        has_next: bool,
        new_ticket: int | None,
    ):
        if data is None:
            # TODO: Verify this
            new_ticket = None  # Dead. No more data to be sent.
        self.src = src
        self.dst = dst
        self.requesting = requesting
        self.requested = requested
        self.data = data
        self.iteration = iteration
        self.has_next = has_next
        self.new_ticket = new_ticket

    @property
    def id(self) -> tuple[str, str, int]:
        return self.requesting, self.requested, self.iteration

    @classmethod
    def create(
        cls,
        request: WorkerRequest,
        data: Data | None,
        has_next: bool,
        new_ticket: int | None,
    ) -> "WorkerResponse":
        if data is None:
            if has_next:
                raise ValueError(
                    f"Dead data detected, but has_next is {has_next} (expected False). "
                    "If you are signaling end of stream, use WorkerResponse.eos()."
                )
        return cls(
            src=request.dst,
            dst=request.src,
            requesting=request.requesting,
            requested=request.requested,
            data=data,
            iteration=request.iteration,
            has_next=has_next,
            new_ticket=new_ticket,
        )

    @classmethod
    def eos(cls, request: WorkerRequest) -> "WorkerResponse":
        return cls(
            src=request.dst,
            dst=request.src,
            requesting=request.requesting,
            requested=request.requested,
            data=None,
            iteration=request.iteration,
            has_next=True,
            new_ticket=None,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"({', '.join(str(k) + '=' + str(v) for k, v in vars(self).items())})"
        )


class ResultStreamIterator(ABC, AsyncIterator[Data | None]):
    pass


class InputStreamIterator(ABC, AsyncIterator["WorkerInput"]):
    pass


class ResultPuller:
    name: str

    async def pull_result(self, looping: bool, request: WorkerRequest) -> Data | None:
        data = None
        async for new_data in self.result_iterator(self.name, looping, request):
            if new_data is None:
                assert data is None, "Dead signal must be yielded only once"
                break
            if data is None:
                data = new_data
            else:
                data += new_data
        return data

    @abstractmethod
    def result_iterator(
        self, worker_name: str, looping: bool, request: WorkerRequest
    ) -> ResultStreamIterator:
        """Returns an async iterator that yields result from the requested worker

        Parameters
        ----------
        worker_name : str
            Name of the worker that makes the request
        looping : bool
            Whether the caller is a looping task. If True, the iterator will
            stop at the end of the current iteration. Otherwise, it will continue
            until there is no more iteration.
        request : WorkerRequest
            Initial request for pulling data

        Returns
        -------
        ResultStreamIterator
            An async iterator that yields data from the requested worker
        """
        raise NotImplementedError()

    @abstractmethod
    async def unsubscribe(self, worker_name: str, unsub_request: WorkerRequest) -> None:
        raise NotImplementedError()

    @abstractmethod
    def start_profiling_task(
        self, worker_name: str, op_id: str, iteration: int
    ) -> None:
        """Starts profiling a task

        Parameters
        ----------
        worker_name : str
            Name of the worker that makes the request
        op_id : str
            Op ID
        iteration : int
            Iteration number
        """
        raise NotImplementedError()

    @abstractmethod
    async def stop_profiling_task(
        self, worker_name: str, op_id: str, iteration: int
    ) -> None:
        """Stops profiling a task

        Parameters
        ----------
        worker_name : str
            Name of the worker that makes the request
        op_id : str
            Op ID
        iteration : int
            Iteration number
        """
        raise NotImplementedError()

    @abstractmethod
    def start_profiling_range(self, worker_name: str, range_id: str) -> None:
        """Starts profiling a range

        Parameters
        ----------
        worker_name : str
            Name of the worker that makes the request
        range_id : str
            Range ID
        """
        raise NotImplementedError()

    @abstractmethod
    async def stop_profiling_range(self, worker_name: str, range_id: str) -> None:
        """Stops profiling a range

        Parameters
        ----------
        worker_name : str
            Name of the worker that makes the request
        range_id : str
            Range ID
        """
        raise NotImplementedError()

    @abstractmethod
    async def get_cached_results(
        self, inp: "WorkerInput", keys: Sequence[Hashable]
    ) -> list[str | None] | list[list[Message] | None]:
        """Fetch cached results for the given keys.

        Parameters
        ----------
        keys : Sequence[Hashable]
            A sequence of cache keys to fetch.

        Returns
        -------
        list[str | None] | list[list[Message] | None]
            A list of cached results corresponding to the input keys. Each element is either
            the cached result (str or list of Message) or None if the key was not found.
        """
        raise NotImplementedError()

    @abstractmethod
    async def cache_results(
        self,
        inp: "WorkerInput",
        batch: dict[Hashable, str] | dict[Hashable, list[Message]],
        overwrite: bool = False,
    ) -> None:
        """Cache results for the given keys.

        Parameters
        ----------
        inp : WorkerInput
            The worker input associated with the cache.
        keys : Sequence[Hashable]
            A sequence of cache keys to store.
        results : Sequence[str | list[Message] | None]
            A sequence of results to be cached. Each element corresponds to the input keys.
            If an element is None, it indicates that there is no result to cache for that key.
        """
        raise NotImplementedError()


class WorkerArg:
    def __init__(
        self,
        *,
        op_id: str,
        data: Data | None = None,
        src_worker: str | None = None,
        src_task_id: str | None = None,
    ):
        if (src_worker is None) != (src_task_id is None):
            raise ValueError("Both src_worker and src_task_id must be set or unset")
        self._iteration: int = 0
        self.op_id = op_id
        self._original_data: Data | None = data
        """Original data without any modifications"""
        self._data: Data | None = data
        self._src_worker: str | None = src_worker
        self._src_task_id: str | None = src_task_id

    def __gt__(self, other: "WorkerArg") -> bool:
        # Don't sort WorkerArg
        return True

    @property
    def data(self) -> Data:
        if self._data is None:
            raise ValueError("Data has not been computed")
        return self._data

    @property
    def original_data(self) -> Data:
        if self._original_data is None:
            raise ValueError("There is no original data")
        return self._original_data

    def copy(self) -> Self:
        return copy.copy(self)

    def is_available(self) -> bool:
        return self._data is not None

    def is_dead(self) -> bool:
        return self._data is None and self._src_task_id is None

    def set_dead(self) -> None:
        self._data = self._src_worker = self._src_task_id = None

    def set_data(self, data: Data) -> None:
        self._data = data

    def mark_resolve(self) -> None:
        self._iteration += 1

    @classmethod
    def dead(cls, op_id: str) -> "WorkerArg":
        return cls(op_id=op_id)

    def into_empty(self, op_id: str, dtype: DataType) -> "WorkerArg":
        return self.__class__(op_id=op_id, data=self.data.into_empty(dtype))

    def into_empty_text(self, op_id: str) -> "WorkerArg":
        return self.into_empty(op_id, DataType.TEXT)

    def into_empty_message(self, op_id: str) -> "WorkerArg":
        return self.into_empty(op_id, DataType.MESSAGE)

    @property
    def src_worker(self) -> str | None:
        return self._src_worker

    @property
    def src_task_id(self) -> str | None:
        return self._src_task_id

    async def resolve(
        self, puller: ResultPuller, task_id: str, looping: bool, eager: bool = False
    ) -> Data | None:
        self.mark_resolve()
        if self.is_dead():
            return None
        if self._data is not None and self.src_worker is None:
            return self._data
        assert (self._src_worker is not None) and (self._src_task_id is not None)
        request = WorkerRequest(
            src=puller.name,
            dst=self._src_worker,
            requesting=task_id,
            requested=self._src_task_id,
            iteration=self._iteration,
        )
        result = await puller.pull_result(looping, request)
        if result is None:
            self.set_dead()
        else:
            self.set_data(result)

        if eager:
            self._src_worker = self._src_task_id = None
            self._original_data = self._data

        return result


class WorkerInput:
    class _InputIterator(InputStreamIterator):
        def __init__(
            self,
            worker_input: "WorkerInput",
            puller: ResultPuller,
            resolving_tasks: set[asyncio.Task],
            is_concat: bool = False,
        ) -> None:
            self.puller = puller
            self.worker_input = worker_input
            self.is_concat = is_concat

            # Resolve each task ID only once.
            all_args = worker_input.get_all_args()
            # (src_worker, src_task_id) -> list[WorkerArg]
            self.tasks_to_resolve = self.worker_input._get_tasks_to_resolve(all_args)
            self.to_resolve_count = len(self.tasks_to_resolve)

            self.resolved_data: dict[str, Data] = {}
            # To track the number of args resolved for each data index
            self.resolved_count_by_index: defaultdict[int, int] = defaultdict(int)
            # For resolving tasks to send their results
            self.result_queue: AIOQueue[tuple[str, str, Data | None]] = AIOQueue()
            self.resolving_tasks = resolving_tasks

            self.initially_ready_data = {
                arg: arg.original_data.copy()
                for arg in self.worker_input._get_ready_args(all_args)
            }

            self.is_dead: bool = False

            self.not_ready: bool = True

        def start_profiling_task(self) -> None:
            if self.not_ready:
                self.puller.start_profiling_task(
                    self.puller.name,
                    self.worker_input.op_id,
                    self.worker_input.iteration,
                )
                self.not_ready = False

        def __aiter__(self) -> AsyncIterator["WorkerInput"]:
            async def resolve_arg(
                result_queue: AIOQueue[tuple[str, str, Data | None]],
                src_worker: str,
                src_task_id: str,
            ) -> None:
                async for data in self.worker_input.resolve_arg_stream(
                    self.puller, src_worker, src_task_id
                ):
                    await result_queue.put((src_worker, src_task_id, data))

            async def iterate_async() -> AsyncGenerator["WorkerInput", None]:
                self.start_profiling_task()
                yield self.worker_input

            def done_callback(task: asyncio.Task) -> None:
                self.resolving_tasks.discard(task)
                if len(self.resolving_tasks) == 0:
                    self.result_queue.close()

            if self.to_resolve_count == 0:
                # All args are already resolved
                return iterate_async()

            # Resolve tasks concurrently
            for src_worker, src_task_id in self.tasks_to_resolve:
                task = asyncio.create_task(
                    resolve_arg(self.result_queue, src_worker, src_task_id)
                )
                self.resolving_tasks.add(task)
                # To determine when all tasks are resolved
                task.add_done_callback(done_callback)

            return self

        async def __anext__(self) -> "WorkerInput":
            if (
                len(self.resolving_tasks) == 0 and self.result_queue.empty()
            ) or self.is_dead:
                raise StopAsyncIteration

            while True:
                try:
                    src_worker, src_task_id, data = await self.result_queue.get()
                except asyncio.CancelledError:
                    raise StopAsyncIteration

                if data is None:
                    # The current task is dead
                    for arg in self.tasks_to_resolve[src_worker, src_task_id]:
                        arg.set_dead()
                    # Return dead input only once
                    self.is_dead = True
                    self.start_profiling_task()
                    return self.worker_input

                # The current task is alive
                if self.is_concat:
                    to_return = self._resolve_data_concat(data, src_task_id)
                else:
                    to_return = self._resolve_data_no_concat(data, src_task_id)

                if to_return:
                    # There is data to return
                    self.start_profiling_task()
                    return self.worker_input

        def _resolve_data_concat(self, data: Data, src_task_id: str) -> bool:
            """
            Returns
            -------
            bool
                Whether there is data to be returned
            """
            data = data.copy()
            for (_, task_id), args in self.tasks_to_resolve.items():
                if task_id == src_task_id:
                    for arg in args:
                        arg.set_data(data)
                else:
                    for arg in args:
                        arg.set_data(data.into_empty(None))
            for arg, data in self.initially_ready_data.items():
                data.pop_by_indices(data.indices)
                arg.set_data(data)
            return True

        def _resolve_data_no_concat(self, data: Data, src_task_id: str) -> bool:
            """
            Returns
            -------
            bool
                Whether there is data to be returned
            """
            tasks_to_resolve = self.tasks_to_resolve
            resolved_data = self.resolved_data
            resolved_count_by_index = self.resolved_count_by_index
            to_resolve_count = self.to_resolve_count

            initially_ready_data = self.initially_ready_data

            # Accumulate data
            if src_task_id not in resolved_data:
                # Copy here as we need to modify the data in place and others
                # may be keeping a reference to the same data
                resolved_data[src_task_id] = data.copy()
            else:
                resolved_data[src_task_id] += data
            # Increment the number of resolved args for each data index
            for i in data.indices:
                resolved_count_by_index[i] += 1
            # Check if all args are resolved for any data index
            ready_indices = [
                i
                for i, count in resolved_count_by_index.items()
                if count == to_resolve_count
            ]
            if len(ready_indices) == 0:
                return False
            for i in ready_indices:
                del resolved_count_by_index[i]
            # Return the input with resolved data
            for (_, src_task_id), args in tasks_to_resolve.items():
                data = resolved_data[src_task_id]
                # Guarantee all args' data have consistent order
                ready_data = data.pop_by_indices(ready_indices)
                for arg in args:
                    arg.set_data(ready_data)
            for arg, data in initially_ready_data.items():
                ready_data = data.pop_by_indices(ready_indices)
                arg.set_data(ready_data)
            return True

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        max_iter: int | None,
        args: list[WorkerArg] | None = None,
        kwargs: dict[str, WorkerArg] | None = None,
    ):
        """
        Parameters
        ----------
        request_id : str
            Request ID.
        op_id : str
            Op ID.
        is_eager : bool
            Whether this input is to be executed eagerly.
        ref_count : int
            Number of references to this input.
        max_iter : int | None
            Maximum number of iterations. There are three cases:
            - None: No looping.
            - -1: Infinite looping.
            - n: Looping for n times.
        args : list[WorkerArg] | None
            Positional arguments.
        kwargs : dict[str, WorkerArg] | None
            Keyword arguments.
        """
        self._task_id: str | None = None
        self._iteration: int = 0
        self._is_error: bool = False
        self.request_id = request_id
        self.op_id = op_id
        self.is_eager = is_eager
        self.ref_count = ref_count
        self.max_iter = max_iter
        self.args: list[WorkerArg] = args or []
        self.kwargs: dict[str, WorkerArg] = kwargs or {}

    @property
    def looping(self) -> bool:
        return self.max_iter is not None

    @property
    def task_id(self) -> str:
        if self._task_id is None:
            raise ValueError("Task ID has not been set")
        return self._task_id

    @task_id.setter
    def task_id(self, tid: str) -> None:
        if self._task_id is not None:
            raise ValueError("Task ID has already been set")
        self._task_id = tid

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    @abstractmethod
    def output_type(self) -> DataType:
        raise NotImplementedError()

    def empty_output(self) -> Data:
        if self.output_type == DataType.TEXT:
            return Data.text([], [])
        return Data.message(MessageList.from_messages([]), [])

    def get_all_args(self) -> set[WorkerArg]:
        return set(itertools.chain(self.args, self.kwargs.values()))

    def _get_tasks_to_resolve(
        self, all_args: set[WorkerArg] | None = None
    ) -> dict[tuple[str, str], list[WorkerArg]]:
        if all_args is None:
            all_args = self.get_all_args()
        to_resolve: dict[tuple[str, str], list[WorkerArg]] = defaultdict(list)
        for arg in all_args:
            arg.mark_resolve()
            if arg.src_worker is not None and arg.src_task_id is not None:
                to_resolve[arg.src_worker, arg.src_task_id].append(arg)
        return to_resolve

    def _get_ready_args(self, all_args: set[WorkerArg] | None = None) -> set[WorkerArg]:
        if all_args is None:
            all_args = self.get_all_args()
        return {
            arg for arg in all_args if arg.src_worker is None or arg.src_task_id is None
        }

    def _get_worker_request(
        self, puller: ResultPuller, src_worker: str, src_task_id: str
    ) -> WorkerRequest:
        return WorkerRequest(
            src=puller.name,
            dst=src_worker,
            requesting=self.task_id,
            requested=src_task_id,
            iteration=self._iteration,
        )

    def _get_unsub_request(
        self, puller: ResultPuller, src_worker: str, src_task_id: str
    ) -> WorkerRequest:
        request = self._get_worker_request(puller, src_worker, src_task_id)
        return request.to_unsubscribe(0, WorkerRequest.UnsubMode.TASK)

    async def resolve_arg(
        self, puller: ResultPuller, src_worker: str, src_task_id: str
    ) -> tuple[str, Data | None]:
        request = self._get_worker_request(puller, src_worker, src_task_id)
        result = await puller.pull_result(self.looping, request)
        return src_task_id, result

    async def resolve(self, puller: ResultPuller) -> Self:
        self._iteration += 1

        # Resolve each task ID only once.
        tasks_to_resolve = self._get_tasks_to_resolve()
        resolved_data = dict(
            await asyncio.gather(
                *[
                    self.resolve_arg(puller, src_worker, src_task_id)
                    for src_worker, src_task_id in tasks_to_resolve
                ]
            )
        )
        is_alive = True
        all_args = list(itertools.chain(*tasks_to_resolve.values()))
        for arg in all_args:
            assert arg.src_task_id is not None
            data = resolved_data[arg.src_task_id]
            if data is None:
                arg.set_dead()
                is_alive = False
            else:
                arg.set_data(data)

        if is_alive and len(all_args) > 1:
            # In case all arguments are alive and there are multiple arguments,
            # sort the data to ensure consistent order among multiple arguments.
            for arg in all_args:
                # We can assume that all arguments' data are available because
                # the arguments are alive and have been resolved.
                arg.set_data(arg.data.sort())

        puller.start_profiling_task(puller.name, self.op_id, self._iteration)
        return self

    async def resolve_arg_stream(
        self, puller: ResultPuller, src_worker: str, src_task_id: str
    ) -> AsyncGenerator[Data | None, None]:
        request = self._get_worker_request(puller, src_worker, src_task_id)
        async for data in puller.result_iterator(puller.name, self.looping, request):
            yield data

    async def unsubscribe(self, puller: ResultPuller) -> None:
        """Unsubscribes from arguments' data for future iteration

        This is first introduced to signal EnterOp that the looping tasks are dead.
        """

        assert self.looping, "Unsubscribe is only allowed for looping tasks"
        # Unsubscribe from all tasks for the next iteration
        self._iteration += 1
        tasks_to_resolve = self._get_tasks_to_resolve()

        for src_worker, src_task_id in tasks_to_resolve:
            request = self._get_unsub_request(puller, src_worker, src_task_id)
            await puller.unsubscribe(puller.name, request)

    @asynccontextmanager
    async def input_iterator(
        self, puller: ResultPuller, is_concat: bool = False
    ) -> AsyncGenerator[AsyncIterator[Self], None]:
        self._iteration += 1
        resolving_tasks: set[asyncio.Task] = set()
        try:
            yield cast(
                AsyncIterator[Self],
                self._InputIterator(self, puller, resolving_tasks, is_concat),
            )
        finally:
            while resolving_tasks:
                task = resolving_tasks.pop()
                await task

    def is_registered(self) -> bool:
        return self._task_id is not None

    def is_dead(self) -> bool:
        return self._is_error or any(arg.is_dead() for arg in self.get_all_args())

    def mark_error(self) -> None:
        self._is_error = True

    def get_cache_keys(self) -> dict[int, Hashable] | None:
        return None


InputType = TypeVar("InputType", bound=WorkerInput)


class BaseWorkerInputBatch(Generic[InputType]):
    def __init__(self, inputs: Sequence[InputType]):
        self.inputs = inputs

    def __iter__(self) -> Iterator[InputType]:
        return iter(self.inputs)

    def wrap(self, inputs: Sequence[InputType]) -> Self:
        return self.__class__(inputs)

    def filter_dead(self, inputs: Sequence[InputType]) -> tuple[Self, list[InputType]]:
        """Filters the given worker inputs for alive and dead inputs

        Parameters
        ----------
        inputs : Sequence[InputType]
            Worker inputs to filter

        Returns
        -------
        Self
            Batch of alive inputs
        list[InputType]
            Dead inputs
        """
        alive_inputs: list[InputType] = []
        dead_inputs: list[InputType] = []
        for inp in inputs:
            if inp.is_dead():
                dead_inputs.append(inp)
            else:
                alive_inputs.append(inp)
        return self.__class__(alive_inputs), dead_inputs


class WorkerInputBatch(BaseWorkerInputBatch[WorkerInput]):
    pass
