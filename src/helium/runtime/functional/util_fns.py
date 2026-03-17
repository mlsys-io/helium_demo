import itertools
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager
from typing import Self

from helium.common import Slice
from helium.ops import SingleDtype
from helium.runtime.data import Data, DataType
from helium.runtime.functional.fns import FnInput
from helium.runtime.worker.worker_input import ResultPuller, WorkerArg, WorkerRequest
from helium.utils import utils


class FormatFnInput(FnInput):
    NAME: str = "format"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        max_iter: int | None,
        template: str,
        format_args: list[WorkerArg],
        format_kwargs: dict[str, WorkerArg],
    ):
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=max_iter,
            args=format_args,
            kwargs=format_kwargs,
        )
        self.template = template

    @property
    def output_type(self) -> DataType:
        return DataType.TEXT

    async def _run(self, puller: ResultPuller) -> AsyncGenerator[Data | None, None]:
        template = self.template
        format_args = [arg.data.as_text() for arg in self.args]
        format_kwargs = {k: v.data.as_text() for k, v in self.kwargs.items()}
        cur_outputs = []
        first_data = next(iter(itertools.chain(self.args, self.kwargs.values()))).data
        for i in range(len(first_data)):
            formatted = template.format(
                *[arg[i] for arg in format_args],
                **{k: v[i] for k, v in format_kwargs.items()},
            )
            cur_outputs.append(formatted)
        yield first_data.into_text(cur_outputs)


class LambdaFnInput(FnInput):
    NAME: str = "lambda"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        max_iter: int | None,
        inputs: list[WorkerArg],
        fn: Callable[[tuple[SingleDtype, ...]], str],
    ) -> None:
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=max_iter,
            args=inputs,
        )
        self.fn = fn

    @property
    def output_type(self) -> DataType:
        return DataType.TEXT

    async def _run(self, puller: ResultPuller) -> AsyncGenerator[Data | None, None]:
        data = [arg.data for arg in self.args]
        inputs = [d.data for d in data]
        out = [self.fn(fn_args) for fn_args in zip(*inputs)]
        yield data[0].into_text(out)


class SliceFnInput(FnInput):
    NAME: str = "slice"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        max_iter: int | None,
        inp: WorkerArg,
        indices: Iterable[int] | Slice,
    ) -> None:
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=max_iter,
            args=[inp],
        )
        self.indices = indices

    @property
    def output_type(self) -> DataType:
        raise ValueError("SliceFnInput has dynamic output type")

    def _get_worker_request(
        self, puller: ResultPuller, src_worker: str, src_task_id: str
    ) -> WorkerRequest:
        return WorkerRequest(
            src=puller.name,
            dst=src_worker,
            requesting=self.task_id,
            requested=src_task_id,
            iteration=self._iteration,
            indices=self.indices,
        )

    async def _run(self, puller: ResultPuller) -> AsyncGenerator[Data | None, None]:
        indices = utils.indices_to_list(self.indices)
        yield self.args[0].data.get_by_indices(indices, uncheck=True)


class ConcatFnInput(FnInput):
    NAME: str = "concat"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        max_iter: int | None,
        inputs: list[WorkerArg],
    ) -> None:
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=max_iter,
            args=inputs,
        )

    @property
    def output_type(self) -> DataType:
        raise ValueError("ConcatFnInput has dynamic output type")

    @asynccontextmanager
    async def input_iterator(
        self, puller: ResultPuller, is_concat: bool = True
    ) -> AsyncGenerator[AsyncIterator[Self], None]:
        async with super().input_iterator(puller, is_concat) as it:
            yield it

    async def _run(self, puller: ResultPuller) -> AsyncGenerator[Data | None, None]:
        if len(self.args) == 0:
            raise ValueError("ConcatFnInput must have at least one input")
        ret = self.args[0].data
        for arg in self.args[1:]:
            ret += arg.data
        yield ret
