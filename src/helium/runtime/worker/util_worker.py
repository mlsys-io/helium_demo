import asyncio
from collections.abc import AsyncIterator, Sequence

from helium.runtime.functional import FnInputBatch
from helium.runtime.utils.logger import Logger, LogLevel
from helium.runtime.utils.queue import AIOQueue, MPQueue
from helium.runtime.worker.worker import (
    AsyncExecutor,
    DefaultAsyncWorker,
    MPWorker,
    WorkerInput,
    WorkerInputBatch,
    WorkerManager,
    WorkerOutput,
)
from helium.runtime.worker.worker_input import ResultPuller, WorkerArg


class AsyncFnExecutor(AsyncExecutor):
    """
    Output worker must run on the main process.
    """

    def __init__(
        self,
        manager: WorkerManager,
        name: str | None = None,
        logger: Logger | None = None,
        log_level: LogLevel | None = None,
    ) -> None:
        super().__init__(
            manager=manager,
            fn_inp_channel=AIOQueue(),
            in_channel=MPQueue() if DefaultAsyncWorker is MPWorker else AIOQueue(),
            out_channel=MPQueue() if DefaultAsyncWorker is MPWorker else AIOQueue(),
            name=name,
            logger=logger,
            log_level=log_level,
        )

    async def add_task(
        self,
        inputs: WorkerInputBatch | FnInputBatch,
        puller: ResultPuller | None = None,
    ) -> list[WorkerArg]:
        if not isinstance(inputs, FnInputBatch):
            raise ValueError("AsyncFnExecutor only accepts FnInputBatch")

        # Prepare inputs
        for inp in inputs:
            if inp.is_eager:
                if inp.looping:
                    raise ValueError("Cannot execute looping tasks eagerly")
                # The results will be resolved only by this call
                inp.ref_count = 1
            self.input_registry.register(inp)

        # Dispatch inputs for execution
        await self._fn_handling_loop.add_event(inputs)

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
                            self, f"{arg.src_task_id}-eager", looping=False, eager=True
                        )
                    )
                )

        if len(resolving_tasks) > 0:
            # Wait for all eagerly executed tasks to be resolved
            await asyncio.gather(*resolving_tasks)

        return out

    def _execute(
        self, inputs: WorkerInputBatch
    ) -> AsyncIterator[Sequence[tuple[WorkerInput, WorkerOutput]]]:
        raise ValueError("AsyncFnExecutor cannot be used as a worker")

    async def start(self) -> None:
        await self._start()
        self.logger.debug("Worker %s started.", self.name)

    async def join(self) -> None:
        await self._join()
        self._check_memory_leak()
        self.logger.debug("Worker %s terminated.", self.name)
