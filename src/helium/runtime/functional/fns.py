from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator

from helium.runtime.data import Data, DataType
from helium.runtime.worker.worker_input import (
    BaseWorkerInputBatch,
    ResultPuller,
    WorkerArg,
    WorkerInput,
)


class FnInput(WorkerInput, ABC):
    NAME: str = "fn"

    async def run(self, puller: ResultPuller) -> AsyncGenerator[Data | None, None]:
        async with self.input_iterator(puller) as input_iterator:
            async for inp in input_iterator:
                if (
                    self.max_iter is not None
                    and self.max_iter >= 0
                    and self._iteration > self.max_iter
                ):
                    # Exceeds the maximum number of iterations.
                    yield None
                elif inp.is_dead():
                    yield None
                else:
                    async for data in inp._run(puller):
                        yield data

    @abstractmethod
    def _run(self, puller: ResultPuller) -> AsyncIterator[Data | None]:
        pass


class FnInputBatch(BaseWorkerInputBatch[FnInput]):
    pass


class DataFnInput(FnInput):
    NAME: str = "data"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        data: list[str],
    ) -> None:
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=None,
        )
        self.data = data

    @property
    def output_type(self) -> DataType:
        return DataType.TEXT

    async def _run(self, puller: ResultPuller) -> AsyncGenerator[Data | None, None]:
        yield Data.text(self.data, indices=None)


class InputFnInput(FnInput):
    NAME: str = "input"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        inputs: list[str],
    ) -> None:
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=None,
        )
        self.inputs = inputs

    @property
    def output_type(self) -> DataType:
        return DataType.TEXT

    async def _run(self, puller: ResultPuller) -> AsyncGenerator[Data | None, None]:
        yield Data.text(self.inputs, indices=None)


class OutputFnInput(FnInput):
    NAME: str = "output"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        output: WorkerArg,
    ) -> None:
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=None,
            args=[output],
        )

    @property
    def output_type(self) -> DataType:
        raise ValueError("OutputFnInput has dynamic output type")

    async def _run(self, puller: ResultPuller) -> AsyncGenerator[Data | None, None]:
        yield self.args[0].data
