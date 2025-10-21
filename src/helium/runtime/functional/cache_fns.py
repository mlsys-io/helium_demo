from collections.abc import AsyncGenerator
from typing import Any, Hashable, cast

from helium.common import Message
from helium.runtime.data import Data, DataType, MessageList
from helium.runtime.functional.fns import FnInput
from helium.runtime.worker.worker_input import ResultPuller


class CacheResolveFnInput(FnInput):
    NAME: str = "cache_resolve"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        max_iter: int | None,
        worker_input: Any,
    ):
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=max_iter,
            args=[],
            kwargs={},
        )
        self.inp = worker_input
        self.resolved_data: Data | None = None

    @classmethod
    def wrap(cls, inp: Any) -> "CacheResolveFnInput":
        return cls(
            request_id=inp.request_id,
            op_id=inp.op_id,
            is_eager=inp.is_eager,
            ref_count=1,
            max_iter=inp.max_iter,
            worker_input=inp,
        )

    async def _run(self, puller: ResultPuller) -> AsyncGenerator[Data | None, None]:
        inp = self.inp
        cache_keys = inp.get_cache_keys()
        if cache_keys is None:
            yield self.inp.empty_output()
            return

        cached_results = await puller.get_cached_results(inp, list(cache_keys.values()))
        indices: list[int] = inp.data.indices
        cached_indices: list[int] = []
        if inp.output_type == DataType.TEXT:
            data: list[str] = []
            cached_data: list[str] = []
            for i, result in zip(indices, cached_results):
                if result is None:
                    data.append(f"{self.op_id}_{i}")
                else:
                    result_str = cast(str, result)
                    data.append(result_str)
                    cached_indices.append(i)
                    cached_data.append(result_str)
            if len(cached_indices) > 0:
                self.resolved_data = Data.text(cached_data, indices=cached_indices)
            yield Data.text(data, indices=indices)
        else:
            messages: list[list[Message]] = []
            cached_messages: list[list[Message]] = []
            for i, inp_messages, result in zip(
                indices, inp.data.as_message(), cached_results
            ):
                if result is None:
                    messages.append(
                        inp_messages + [Message("assistant", f"{self.op_id}_{i}")]
                    )
                else:
                    result_msgs = cast(list[Message], result)
                    messages.append(result_msgs)
                    cached_indices.append(i)
                    cached_messages.append(result_msgs)
            if len(cached_indices) > 0:
                self.resolved_data = Data.message(
                    MessageList.from_messages(cached_messages), indices=cached_indices
                )
            yield Data.message(MessageList.from_messages(messages), indices=indices)

    @property
    def output_type(self) -> DataType:
        return self.inp.output_type


class CacheFetchFnInput(FnInput):
    NAME: str = "cache_fetch"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        max_iter: int | None,
        cached_data: Data | tuple[Any, dict[int, Hashable]],
    ):
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=max_iter,
            args=[],
            kwargs={},
        )
        self.data = cached_data

    async def _run(self, puller: ResultPuller) -> AsyncGenerator[Data | None, None]:
        if isinstance(self.data, Data):
            yield self.data
            return

        inp, cached_keys = self.data
        cached_results = await puller.get_cached_results(
            inp, list(cached_keys.values())
        )
        if any(result is None for result in cached_results):
            raise ValueError("Cache miss for some keys")
        if inp.output_type == DataType.TEXT:
            yield inp.data.into_text([cast(str, result) for result in cached_results])
        elif inp.output_type == DataType.MESSAGE:
            yield inp.data.into_message(
                MessageList.from_messages(
                    [cast(list[Message], result) for result in cached_results]
                )
            )

    @property
    def output_type(self) -> DataType:
        if isinstance(self.data, Data):
            return self.data.dtype
        return self.data[0].output_type
