import asyncio
import enum
from typing import Any

from helium.runtime.utils.logger import (
    Logger,
    LogLevel,
    init_child_logger,
    log_on_exception_async,
)
from helium.runtime.utils.loop import AsyncEventLoop
from helium.runtime.utils.pool import AsyncPool
from helium.runtime.utils.queue import AIOQueue, AsyncQueue, MPQueue
from helium.utils import unique_id


class _StateRequestOp(enum.Enum):
    GET = enum.auto()
    POP = enum.auto()
    SET = enum.auto()
    DUMP = enum.auto()
    CLEAR = enum.auto()


class _StateRequest:
    def __init__(
        self,
        sender: str,
        op: _StateRequestOp,
        job_id: str,
        scope: str | None = None,
        key: str | None = None,
        value: Any = None,
    ):
        self.req_id = unique_id()
        self.sender = sender
        self.op = op
        self.job_id = job_id
        self.scope = scope
        self.key = key
        self.value = value

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(req_id={self.req_id}, sender={self.sender}, "
            f"op={self.op}, job_id={self.job_id}, scope={self.scope}, key={self.key}, "
            f"value={self.value})"
        )


class _StateResponse:
    def __init__(self, req_id: str, state: Any = None, error: str | None = None):
        self.req_id = req_id
        self.state = state
        self.error = error

    def unwrap(self) -> Any:
        if self.error is not None:
            raise RuntimeError(f"Error in state response: {self.error}")
        return self.state

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(req_id={self.req_id}, state={self.state}, "
            f"error={self.error})"
        )


class StateManager:
    """A class to manage global states of the jobs."""

    _DEFAULT = object()

    def __init__(
        self,
        worker_names: list[str],
        use_mp: bool = False,
        name: str = "StateManager",
        logger: Logger | None = None,
        log_level: LogLevel | None = None,
    ) -> None:
        self.name = name
        self.logger = init_child_logger(self.name, logger, log_level)
        self._use_mp = use_mp
        # Stores the states of jobs.
        self._job_states: dict[str, dict[str, dict[str, Any]]] = {}
        # To send responses to the workers.
        self._worker_channels: dict[str, AsyncQueue[_StateResponse]] = (
            {worker_name: MPQueue() for worker_name in worker_names}
            if self._use_mp
            else {worker_name: AIOQueue() for worker_name in worker_names}
        )
        # To be used by workers to pull results.
        self._pulling_tasks: dict[str, asyncio.Task] = {}
        self._result_pool: AsyncPool[str, _StateResponse] = AsyncPool()
        self._is_closed: bool = False
        # To send requests to the manager.
        self._handling_loop = AsyncEventLoop(
            self._handling_func, in_channel=MPQueue() if self._use_mp else AIOQueue()
        )

    @classmethod
    async def create_and_init(
        cls,
        worker_names: list[str],
        use_mp: bool = False,
        name: str = "StateManager",
        logger: Logger | None = None,
        log_level: LogLevel | None = None,
    ) -> "StateManager":
        state_manager = cls(
            worker_names=worker_names,
            use_mp=use_mp,
            name=name,
            logger=logger,
            log_level=log_level,
        )
        await state_manager._handling_loop.start()
        return state_manager

    def register(self, worker_name: str) -> None:
        """Registers the worker so that it can communicate with the manager"""
        self._worker_channels[worker_name] = MPQueue() if self._use_mp else AIOQueue()

    def start(self, worker_name: str) -> None:
        """Starts the pulling task for the worker to receive results from the manager"""
        if worker_name not in self._pulling_tasks:
            self._pulling_tasks[worker_name] = asyncio.create_task(
                self._pulling_loop(worker_name)
            )

    async def close(self) -> None:
        if not self._is_closed:
            self._is_closed = True
            await self._handling_loop.stop()
            for channel in self._worker_channels.values():
                channel.close()
            for task in self._pulling_tasks.values():
                await task
            if len(self._job_states) > 0:
                self.logger.warning(
                    f"{len(self._job_states)} job states are still tracked when "
                    "closing the manager."
                )

    def track_job(self, job_id: str) -> None:
        self._job_states[job_id] = {}

    def untrack_job(self, job_id: str) -> None:
        if job_id in self._job_states:
            del self._job_states[job_id]
        else:
            raise RuntimeError(f"Job {job_id} has not been tracked.")

    async def _process_request(self, request: _StateRequest) -> Any:
        await self._handling_loop.add_event(request)
        res = await self._result_pool.get(request.req_id)
        return res.unwrap()

    async def get_state(
        self, worker_name: str, job_id: str, scope: str, key: str
    ) -> Any:
        request = _StateRequest(worker_name, _StateRequestOp.GET, job_id, scope, key)
        return await self._process_request(request)

    async def pop_state(
        self, worker_name: str, job_id: str, scope: str, key: str
    ) -> Any:
        request = _StateRequest(worker_name, _StateRequestOp.POP, job_id, scope, key)
        return await self._process_request(request)

    async def set_state(
        self, worker_name: str, job_id: str, scope: str, key: str, value: Any
    ) -> None:
        request = _StateRequest(
            worker_name, _StateRequestOp.SET, job_id, scope, key, value
        )
        return await self._process_request(request)

    async def dump_state(
        self, worker_name: str, job_id: str, scope: str | None
    ) -> dict[str, Any]:
        request = _StateRequest(worker_name, _StateRequestOp.DUMP, job_id, scope)
        return await self._process_request(request)

    async def clear_state(
        self, worker_name: str, job_id: str, scope: str | None
    ) -> None:
        request = _StateRequest(worker_name, _StateRequestOp.CLEAR, job_id, scope)
        return await self._process_request(request)

    async def _handling_func(self, request: _StateRequest, _) -> None:
        req_id = request.req_id
        job_id = request.job_id
        request_scope = request.scope
        request_key = request.key
        request_value = request.value

        def get_job_state() -> dict[str, dict[str, Any]]:
            if job_id not in self._job_states:
                raise RuntimeError(f"Job {job_id} has not been tracked.")
            return self._job_states[job_id]

        def get_scope_state(
            request_scope: str, raise_on_err: bool, pop_on_exist: bool
        ) -> dict[str, Any]:
            job_state = get_job_state()
            scope_state: dict[str, Any]
            found = request_scope in job_state
            if not found and raise_on_err:
                raise RuntimeError(f"Scope {request_scope} not found in job {job_id}.")
            if pop_on_exist:
                scope_state = job_state.pop(request_scope) if found else {}
            elif found:
                scope_state = job_state[request_scope]
            else:
                scope_state = {}
                job_state[request_scope] = scope_state
            return scope_state

        def check_request_scope(job_state: dict[str, dict[str, Any]]) -> None:
            if request_scope not in job_state:
                raise RuntimeError(f"Scope {request_scope} not found in job {job_id}.")

        def check_request_key(scope_state: dict[str, Any]) -> None:
            if request_key not in scope_state:
                raise RuntimeError(
                    f"Key {request_key} not found in job {job_id} with scope "
                    f"{request_scope}."
                )

        try:
            match request.op:
                case _StateRequestOp.GET:
                    if request_scope is None:
                        raise RuntimeError("Scope must be provided for GET operation")
                    if request_key is None:
                        raise RuntimeError("Key must be provided for GET operation")
                    scope_state = get_scope_state(request_scope, True, False)
                    check_request_key(scope_state)
                    state = scope_state[request_key]
                    res = _StateResponse(req_id, state)
                case _StateRequestOp.POP:
                    if request_scope is None:
                        raise RuntimeError("Scope must be provided for POP operation")
                    if request_key is None:
                        raise RuntimeError("Key must be provided for POP operation")
                    scope_state = get_scope_state(request_scope, True, False)
                    check_request_key(scope_state)
                    state = scope_state.pop(request_key)
                    res = _StateResponse(req_id, state)
                case _StateRequestOp.SET:
                    if request_scope is None:
                        raise RuntimeError("Scope must be provided for SET operation")
                    if request_key is None or request_value is None:
                        raise RuntimeError(
                            "Key and value must be provided for SET operation"
                        )
                    scope_state = get_scope_state(request_scope, False, False)
                    scope_state[request_key] = request_value
                    res = _StateResponse(req_id)
                case _StateRequestOp.DUMP:
                    job_state = get_job_state()
                    if request_scope is None:
                        state = job_state
                    else:
                        state = get_scope_state(request_scope, False, True)
                    res = _StateResponse(req_id, state.copy())
                case _StateRequestOp.CLEAR:
                    job_state = get_job_state()
                    if request_scope is None:
                        job_state.clear()
                    else:
                        check_request_scope(job_state)
                        del job_state[request_scope]
                    res = _StateResponse(req_id)
        except RuntimeError as e:
            res = _StateResponse(req_id, error=str(e))

        await self._worker_channels[request.sender].put(res)

    @log_on_exception_async(ignore=[asyncio.CancelledError])
    async def _pulling_loop(self, worker_name: str) -> None:
        """A loop to pull results from the manager."""
        while True:
            res = await self._worker_channels[worker_name].get()
            await self._result_pool.put(res.req_id, res)
