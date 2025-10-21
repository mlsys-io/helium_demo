import functools
import re
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager
from typing import Self

from helium.ops.cond_ops import Predicate
from helium.runtime.data import Data, DataType
from helium.runtime.functional.fns import FnInput
from helium.runtime.worker.worker_input import ResultPuller, WorkerArg, WorkerRequest


class SwitchFnInput(FnInput):
    NAME: str = "switch"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        max_iter: int | None,
        inp: WorkerArg,
        cond_args: list[WorkerArg],
        pred: Predicate | None,
        branch: bool,
        dead_on_empty: bool,
    ) -> None:
        if isinstance(pred, list) and len(pred) != len(cond_args):
            raise ValueError("Inconsistent number of predicates")
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=max_iter,
            args=cond_args,
            kwargs=dict(inp=inp),
        )
        self.pred = pred
        self.branch = branch
        self.dead_on_empty = dead_on_empty

    @property
    def output_type(self) -> DataType:
        raise ValueError("SwitchFnInput has dynamic output type")

    async def run(self, puller: ResultPuller) -> AsyncGenerator[Data | None, None]:
        async with self.input_iterator(puller) as input_iterator:
            async for inp in input_iterator:
                if inp.is_dead():
                    yield None
                elif (
                    inp.max_iter is not None
                    and inp.max_iter >= 0
                    and inp._iteration >= inp.max_iter
                ):
                    # Exceeds the maximum number of iterations.
                    # If the args are not dead, this is used as a loop control op.
                    # Send the dead signal to the true branch and all the remaining
                    # inputs to the false branch.
                    if inp.branch:
                        yield None
                    else:
                        yield inp.kwargs["inp"].data
                else:
                    async for data in inp._run(puller):
                        yield data

    async def _run(self, puller: ResultPuller) -> AsyncGenerator[Data | None, None]:
        def apply_regex(patterns: list[str], args: list[str]) -> bool:
            assert len(patterns) == len(args)
            return not all(
                re.match(pattern, arg) for pattern, arg in zip(patterns, args)
            ) and all(len(arg) > 0 for arg in args)

        def apply_not_regex(patterns: list[str], args: list[str]) -> bool:
            return not apply_regex(patterns, args)

        def apply_pred(
            pred: Callable[..., bool],
            iteration: Iterable[int],
            args: list[str],
        ) -> bool:
            return pred(*iteration, *args)

        def apply_not_pred(
            pred: Callable[..., bool],
            iteration: Iterable[int],
            args: list[str],
        ) -> bool:
            return not pred(*iteration, *args)

        inp_arg = self.kwargs["inp"]
        inp_data = inp_arg.data

        if self.pred is None:
            if self.branch:
                # Forward the input if no predicate is provided.
                yield inp_data
            else:
                yield inp_data.into_empty(inp_data.dtype)
            return

        cond_args: list[list[str]] = [
            (
                arg.data.as_text()
                if arg.data.is_text()
                else [
                    msg_data[-1].content for msg_data in arg.data.as_message()
                ]  # Last message in each message list
            )
            for arg in self.args
        ]

        pred: Callable[[list[str]], bool]
        if isinstance(self.pred, str):
            if self.branch:
                pred = functools.partial(apply_regex, [self.pred] * len(cond_args))
            else:
                pred = functools.partial(apply_not_regex, [self.pred] * len(cond_args))
        elif isinstance(self.pred, list):
            if self.branch:
                pred = functools.partial(apply_regex, self.pred)
            else:
                pred = functools.partial(apply_not_regex, self.pred)
        else:
            iteration = (self._iteration - 1,) if self.looping else ()
            if self.branch:
                pred = functools.partial(apply_pred, self.pred, iteration)
            else:
                pred = functools.partial(apply_not_pred, self.pred, iteration)

        pred_list = [pred(cond) for cond in zip(*cond_args)]  # type: ignore
        out = inp_data.filter(pred_list)
        if out.is_empty() and self.dead_on_empty:
            yield None
        else:
            yield out


class MergeFnInput(FnInput):
    NAME: str = "merge"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        max_iter: int | None,
        inputs: list[WorkerArg],
    ):
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
        raise ValueError("MergeFnInput has dynamic output type")

    async def _run(self, puller: ResultPuller) -> AsyncGenerator[Data | None, None]:
        outputs = [arg for arg in self.args if not arg.is_dead()]
        if len(outputs) == 0:
            yield None
        elif len(outputs) > 1:
            raise ValueError("Multiple alive inputs found")
        yield outputs[0].data


class EnterFnInput(FnInput):
    NAME: str = "enter"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        max_iter: int | None,
        init_inp: WorkerArg,
        future_inp: WorkerArg,
    ) -> None:
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=max_iter,
            args=[init_inp, future_inp],
        )

    @property
    def output_type(self) -> DataType:
        raise ValueError("EnterFnInput has dynamic output type")

    def _get_worker_request(
        self, puller: ResultPuller, src_worker: str, src_task_id: str
    ) -> WorkerRequest:
        return WorkerRequest(
            src=puller.name,
            dst=src_worker,
            requesting=self.task_id,
            requested=src_task_id,
            iteration=max(self._iteration - 1, 1),
        )

    def _get_unsub_request(
        self, puller: ResultPuller, src_worker: str, src_task_id: str
    ) -> WorkerRequest:
        request = self._get_worker_request(puller, src_worker, src_task_id)
        # Adjust to the iteration gap between EnterOp and looping tasks
        request.iteration = max(request.iteration - 1, 1)
        return request.to_unsubscribe(0, WorkerRequest.UnsubMode.TASK)

    def _get_arg(self) -> WorkerArg:
        return self.args[0] if self._iteration == 1 else self.args[1]

    def get_all_args(self) -> set[WorkerArg]:
        return {self._get_arg()}

    async def resolve(self, puller: ResultPuller) -> "EnterFnInput":
        self._iteration += 1
        puller.start_profiling_task(puller.name, self.op_id, self.iteration)
        return self

    @asynccontextmanager
    async def input_iterator(
        self, puller: ResultPuller, is_concat: bool = False
    ) -> AsyncGenerator[AsyncIterator[Self], None]:
        async def iterate_async() -> AsyncIterator[Self]:
            puller.start_profiling_task(puller.name, self.op_id, self.iteration)
            yield self

        self._iteration += 1
        yield iterate_async()

    async def _run(self, puller: ResultPuller) -> AsyncGenerator[Data | None, None]:
        arg = self._get_arg()
        arg.mark_resolve()
        if arg.src_worker is None or arg.src_task_id is None:
            yield arg.data
        else:
            async for data in self.resolve_arg_stream(
                puller, arg.src_worker, arg.src_task_id
            ):
                yield data


class ExitFnInput(FnInput):
    NAME: str = "exit"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        inp: WorkerArg,
    ) -> None:
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=None,
            args=[inp],
        )

    @property
    def output_type(self) -> DataType:
        raise ValueError("ExitFnInput has dynamic output type")

    async def run(self, puller: ResultPuller) -> AsyncGenerator[Data | None, None]:
        # Override to ignore dead signals from the input
        async with self.input_iterator(puller) as input_iterator:
            async for inp in input_iterator:
                if not inp.is_dead():
                    yield inp.args[0].data

    def _run(self, puller: ResultPuller) -> AsyncIterator[Data | None]:
        raise NotImplementedError()
