import asyncio
import enum
import time
from abc import abstractmethod
from collections import Counter
from collections.abc import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    Sequence,
)
from typing import Any, Hashable, Literal, Self, cast

from helium.common import GenerationConfig, Message, Slice
from helium.runtime.data import Data, DataType, MessageData
from helium.runtime.llm import (
    BatchChatMixin,
    BatchCompleteMixin,
    LLMProfilingInfo,
    LLMRegistry,
    LLMServiceConfig,
    UsageInfo,
    VLLMLocalLLM,
)
from helium.runtime.llm.utils import apply_chat_template
from helium.runtime.utils.logger import (
    Logger,
    LogLevel,
    init_child_logger,
    log_on_exception_async,
    log_on_exception_async_generator,
)
from helium.runtime.utils.loop import AsyncEventLoop, AsyncMPEventLoop, EventLoop
from helium.runtime.utils.pool import AsyncPool
from helium.runtime.utils.queue import AIOQueue
from helium.runtime.worker.worker import (
    TASK_END,
    AIOWorker,
    DefaultAsyncWorker,
    TaskEnd,
    WorkerInput,
    WorkerInputBatch,
    WorkerManager,
    WorkerOutput,
)
from helium.runtime.worker.worker_input import WorkerArg


class LLMWorkerInput(WorkerInput):
    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        max_iter: int | None,
        input_slice: Slice,
        generation_config: GenerationConfig,
        cacheable: bool,
        args: list[WorkerArg],
    ):
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=max_iter,
            args=args,
        )
        self.input_slice = input_slice
        self.generation_config = generation_config
        self.cacheable = cacheable
        self._generation_key: tuple | None = None

    @property
    @abstractmethod
    def indices(self) -> list[int]:
        pass

    @property
    @abstractmethod
    def data(self) -> Data:
        pass

    @property
    def generation_key(self) -> tuple:
        if self._generation_key is None:
            self._generation_key = self.get_generation_key()
        return self._generation_key

    @property
    def is_hashable(self) -> bool:
        temperature = self.generation_config.temperature
        return temperature is not None and temperature == 0

    def get_generation_key(self) -> tuple:
        generation_config = self.generation_config
        return (
            generation_config.model,
            generation_config.base_url,
            generation_config.frequency_penalty,
            generation_config.logit_bias,
            generation_config.max_tokens,
            generation_config.presence_penalty,
            generation_config.stop,
            generation_config.ignore_eos,
        )

    def get_cache_key(self, content: str | list[Message]) -> Hashable | None:
        return None

    def get_cache_keys(self, data: Data | None = None) -> dict[int, Hashable] | None:
        return None


class LLMCompletionWorkerInput(LLMWorkerInput):
    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        max_iter: int | None,
        input_slice: Slice,
        generation_config: GenerationConfig,
        prompts: WorkerArg,
        echo: bool,
        cacheable: bool,
    ):
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=max_iter,
            input_slice=input_slice,
            generation_config=generation_config,
            cacheable=cacheable,
            args=[prompts],
        )
        self.echo = echo

    @property
    def output_type(self) -> DataType:
        return DataType.TEXT

    @property
    def indices(self) -> list[int]:
        return self.args[0].data.indices

    @property
    def data(self) -> Data:
        return self.args[0].data

    def get_cache_key(self, content: str | list[Message]) -> Hashable | None:
        assert isinstance(content, str)
        if self.is_hashable:
            return (self.generation_key, content)
        return None

    def get_cache_keys(self, data: Data | None = None) -> dict[int, Hashable] | None:
        if not self.is_hashable:
            return None
        inp_data = self.data if data is None else data
        return {i: self.get_cache_key(text) for i, text in inp_data.iter_text()}


class LLMChatWorkerInput(LLMWorkerInput):
    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        max_iter: int | None,
        input_slice: Slice,
        generation_config: GenerationConfig,
        messages: WorkerArg,
        return_history: bool,
        cacheable: bool,
    ):
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=max_iter,
            input_slice=input_slice,
            generation_config=generation_config,
            cacheable=cacheable,
            args=[messages],
        )
        self.return_history = return_history

    @property
    def output_type(self) -> DataType:
        return DataType.MESSAGE if self.return_history else DataType.TEXT

    @property
    def indices(self) -> list[int]:
        return self.args[0].data.indices

    @property
    def data(self) -> Data:
        return self.args[0].data

    def get_cache_key(self, content: str | list[Message]) -> Hashable | None:
        assert isinstance(content, list)
        if self.is_hashable:
            return (
                self.generation_key,
                tuple((msg.role, msg.content) for msg in content),
            )
        return None

    def get_cache_keys(self, data: Data | None = None) -> dict[int, Hashable] | None:
        if not self.is_hashable:
            return None
        inp_data = self.data if data is None else data
        return {i: self.get_cache_key(msgs) for i, msgs in inp_data.iter_message()}


class DispatchMode(enum.Enum):
    RUNTIME_ADJUSTMENT = enum.auto()
    NO_RUNTIME_ADJUSTMENT = enum.auto()
    OP_WISE_NO_WAIT = enum.auto()


class LLMWorkerInputBatch(WorkerInputBatch):
    def __init__(
        self,
        inputs: Sequence[LLMWorkerInput],
        schedule: Iterable[list[str]],
        dispatch_mode: DispatchMode = DispatchMode.RUNTIME_ADJUSTMENT,
        profiling: bool = False,
    ):
        super().__init__(inputs)
        self.schedule = schedule
        self.dispatch_mode = dispatch_mode
        self.profiling = profiling

    @property
    def llm_inputs(self) -> Sequence[LLMWorkerInput]:
        return self.inputs  # type: ignore

    def wrap(self, inputs: Sequence[WorkerInput]) -> Self:
        cast_inputs = cast(Sequence[LLMWorkerInput], inputs)
        return self.__class__(
            cast_inputs, self.schedule, self.dispatch_mode, self.profiling
        )

    def __iter__(self) -> Iterator[LLMWorkerInput]:
        return iter(self.llm_inputs)

    @property
    def input_map(self) -> dict[str, LLMWorkerInput]:
        return {inp.op_id: inp for inp in self.llm_inputs}

    @property
    def input_slice_map(self) -> dict[str, Slice]:
        return {inp.op_id: inp.input_slice for inp in self.llm_inputs}


class LLMPrecomputeInput(WorkerInput):
    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        generation_config: GenerationConfig,
        prompts: list[str | list[Message]],
    ):
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=None,
        )
        self.generation_config = generation_config
        self.prompts = prompts

    @property
    def output_type(self) -> DataType:
        return DataType.TEXT


class LLMPrecomputeInputBatch(WorkerInputBatch):
    def __init__(self, inputs: Sequence[LLMPrecomputeInput]):
        super().__init__(inputs)

    @property
    def llm_inputs(self) -> Sequence[LLMPrecomputeInput]:
        return self.inputs  # type: ignore

    @property
    def output_type(self) -> DataType:
        return DataType.TEXT


CacheItemType = tuple[str, int, str | list[Message], GenerationConfig]
"""Type for caching: (service_name, query_index, content, generation_config)"""


class LLMWorker(DefaultAsyncWorker):
    def __init__(
        self,
        manager: WorkerManager,
        service_config: LLMServiceConfig,
        name: str | None = None,
        logger: Logger | None = None,
        log_level: LogLevel | None = None,
        benchmarking: bool = False,
    ):
        super().__init__(manager=manager, name=name, logger=logger, log_level=log_level)
        self.service_name = service_config.name
        self.service_config = service_config
        self._llm = LLMRegistry.get(
            service_config.name, config=service_config, benchmarking=benchmarking
        )
        self._tokenizer = self._llm.get_tokenizer()
        # TODO: Consider when to use batchable
        self._batchable = (
            isinstance(self._llm, BatchCompleteMixin)
            and isinstance(self._llm, BatchChatMixin)
            and not isinstance(self._llm, VLLMLocalLLM)
        )

        self._dispatcher = LLMRequestDispatcher(self)
        self._benchmark_loop: EventLoop = (
            AsyncEventLoop(self._benchmark_func, result_collector=AsyncPool())
            if isinstance(self, AIOWorker)
            else AsyncMPEventLoop(self._benchmark_func, has_result=True)
        )

    @classmethod
    def get_llm_service_name(cls, config: LLMServiceConfig) -> str:
        return cls.service_name_with_model(config.name, config.args.get("model"))

    @classmethod
    def service_name_with_model(cls, llm_service: str, model: str | None) -> str:
        if llm_service in ["vllm-local", "vllm-openai"]:
            if model is None:
                raise ValueError(
                    f"Model name must be provided for {llm_service} service."
                )
            return f"{llm_service}-{model}"
        return llm_service

    async def _start(self) -> bool:
        to_start = await super()._start()
        if to_start:
            assert not self._benchmark_loop.is_started()
            await self._benchmark_loop.start()
            await self._llm.start()
        return to_start

    async def _join(self) -> bool:
        to_join = await super()._join()
        if to_join:
            assert self._benchmark_loop.is_running()
            await self._benchmark_loop.stop()
            await self._llm.stop()
        return to_join

    async def reset_prefix_cache(self) -> None:
        await self._llm.reset_prefix_cache()

    async def clear_kv_cache(self) -> None:
        kv_cache_manager = self.manager.kv_cache_manager
        if kv_cache_manager is None:
            return
        await self._llm.clear_kv_cache(kv_cache_manager.controller_client)

    async def change_kv_role(self, new_role: str) -> None:
        await self._llm.change_kv_role(new_role)

    @log_on_exception_async(ignore=[asyncio.CancelledError])
    async def _benchmark_func(
        self, event: tuple[Literal["start", "end"], str | None, str | None], _
    ) -> dict[str, Any] | None:
        mode, api_key, base_url = event
        if mode == "start":
            await self._llm.start_benchmark(api_key, base_url)
            res = None
        elif mode == "end":
            res = await self._llm.stop_benchmark(api_key, base_url)
        else:
            self.logger.warning("Invalid benchmark mode: %s", mode)
            res = None
        return res

    async def start_benchmark(self, api_key: str | None, base_url: str | None) -> None:
        await self._benchmark_loop.process_event(("start", api_key, base_url))

    async def stop_benchmark(
        self, api_key: str | None, base_url: str | None
    ) -> dict[str, Any]:
        res = await self._benchmark_loop.process_event(("end", api_key, base_url))
        if res is None:
            raise ValueError("Invalid benchmark result")
        return res

    @log_on_exception_async_generator(ignore=[asyncio.CancelledError])
    async def execute(
        self, inputs: WorkerInputBatch
    ) -> AsyncGenerator[Sequence[tuple[WorkerInput, WorkerOutput]], None]:
        self.logger.debug("Executing tasks %s", [inp.task_id for inp in inputs])

        if not isinstance(inputs, (LLMWorkerInputBatch, LLMPrecomputeInputBatch)):
            raise ValueError(
                f"Invalid input type: {type(inputs)} (expected LLMWorkerInputBatch)"
            )

        generator: AsyncGenerator[Sequence[tuple[WorkerInput, WorkerOutput]], None]
        if isinstance(inputs, LLMPrecomputeInputBatch):
            profiling_range = f"llm_precompute:{self.name}"
            generator = self._precompute(inputs)
        elif inputs.profiling:
            profiling_range = f"llm_profile:{self.name}"
            generator = self._profile(inputs)
        else:
            profiling_range = f"llm_execute:{self.name}"
            generator = self._execute(inputs)

        self.start_profiling_range(self.name, profiling_range)
        try:
            async for out in generator:
                yield out
        except Exception as e:
            err_inputs = inputs.inputs
            await self._handle_error(err_inputs, e)
            yield [(inp, None) for inp in err_inputs]
        await self.stop_profiling_range(self.name, profiling_range)

    async def _precompute(
        self, inputs: LLMPrecomputeInputBatch
    ) -> AsyncGenerator[Sequence[tuple[LLMPrecomputeInput, WorkerOutput]], None]:
        tokenizer = self._tokenizer

        for inp in inputs.llm_inputs:
            await inp.resolve(self)
            prompts = inp.prompts
            config = inp.generation_config

            if tokenizer is not None:
                # Apply chat template to chat prompts
                new_prompts: list[str | list[Message]] = [
                    prompt for prompt in prompts if isinstance(prompt, str)
                ]
                chat_prompts = [
                    prompt for prompt in prompts if not isinstance(prompt, str)
                ]
                formatted = apply_chat_template(
                    tokenizer, chat_prompts, strip_begin_of_text=True
                )
                new_prompts.extend(formatted)
                prompts = new_prompts

                # Print long prefixes
                # for prompt in prompts:
                #     if len(tokenizer.encode(prompt)) >= 200:
                #         print(repr(prompt), flush=True)

            client = self.manager.get_kv_cache_client()
            await self._llm.precompute_kv_cache(prompts, config, client)
            yield [(inp, Data.text(["OK"] * len(prompts), None)), (inp, TASK_END)]

    def _profile(
        self, inputs: LLMWorkerInputBatch
    ) -> AsyncGenerator[Sequence[tuple[LLMWorkerInput, WorkerOutput]], None]:
        usage_info_dict: dict[LLMWorkerInput, list[UsageInfo]] = {}

        async def handle_request_func(
            request_tracker: AsyncPool[str, tuple[LLMWorkerInput, Data, int]],
        ):
            outputs = await self._llm.get_available_outputs(with_usage=True)
            results: list[tuple[LLMWorkerInput, Data]] = []
            for request_id, result, usage_info in outputs:
                inp, inp_data, i = await request_tracker.pop(request_id)
                # Collect data
                data = self._wrap_output(inp, inp_data, result, [i])
                results.append((inp, data))
                # Collect usage info
                if inp in usage_info_dict:
                    usage_info_dict[inp].append(usage_info)
                else:
                    usage_info_dict[inp] = [usage_info]
                # Cache results
                await self._cache_results(inp, inp_data, data)
            return results

        async def handle_task_end_func(inp: LLMWorkerInput):
            # Send profiling results via global job state
            usage_info_list = usage_info_dict.pop(inp, [])
            profiling_info = LLMProfilingInfo.aggregate(usage_info_list)
            await self.manager.set_job_state(
                self.name,
                inp.request_id,
                "llm_profiling_info",
                inp.op_id,
                profiling_info,
            )

        return self._execute_with_schedule(
            inputs, handle_request_func, handle_task_end_func, with_usage=True
        )

    def _execute(
        self, inputs: WorkerInputBatch
    ) -> AsyncGenerator[Sequence[tuple[LLMWorkerInput, WorkerOutput]], None]:
        async def handle_request_func(
            request_tracker: AsyncPool[str, tuple[LLMWorkerInput, Data, int]],
        ):
            outputs = await self._llm.get_available_outputs(with_usage=False)
            results: list[tuple[LLMWorkerInput, Data]] = []
            for request_id, result in outputs:
                inp, inp_data, i = await request_tracker.pop(request_id)
                data = self._wrap_output(inp, inp_data, result, [i])
                results.append((inp, data))
                # Cache results
                await self._cache_results(inp, inp_data, data)
            return results

        async def handle_task_end_func(inp: LLMWorkerInput):
            return

        assert isinstance(inputs, LLMWorkerInputBatch)
        return self._execute_with_schedule(
            inputs, handle_request_func, handle_task_end_func, with_usage=False
        )

    async def _execute_with_schedule(
        self,
        inputs: LLMWorkerInputBatch,
        handle_request_func: Callable[
            [AsyncPool[str, tuple[LLMWorkerInput, Data, int]]],
            Awaitable[list[tuple[LLMWorkerInput, Data]]],
        ],
        handle_task_end_func: Callable[[LLMWorkerInput], Awaitable[None]],
        with_usage: bool,
    ) -> AsyncGenerator[Sequence[tuple[LLMWorkerInput, WorkerOutput]], None]:
        result_queue: AIOQueue[Sequence[tuple[LLMWorkerInput, WorkerOutput]]] = (
            AIOQueue()
        )
        request_counter: dict[LLMWorkerInput, int] = {
            inp: inp.input_slice.length for inp in inputs
        }
        request_tracker: AsyncPool[str, tuple[LLMWorkerInput, Data, int]] = AsyncPool()
        caching_queue: AIOQueue[
            Sequence[tuple[LLMWorkerInput, int, str | list[Message]]]
        ] = AIOQueue()

        async def caching_loop() -> None:
            cache_item: CacheItemType
            service_name = self.service_name
            try:
                while True:
                    items = await caching_queue.get()
                    for inp, i, content in items:
                        cache_item = (service_name, i, content, inp.generation_config)
                        await self.manager.set_job_state(
                            self.name,
                            inp.request_id,
                            "llm_cacheable",
                            f"{inp.op_id}_{i}",
                            cache_item,
                        )
            except asyncio.CancelledError:
                pass

        async def dispatching_loop() -> None:
            async for data_list in await self._dispatcher.dispatch(inputs):
                llm_inputs: list[str | list[Message]] = []
                configs: list[GenerationConfig] = []
                request_metas: list[tuple[LLMWorkerInput, Data, int]] = []
                to_die: list[tuple[LLMWorkerInput, None]] = []
                to_cache: list[tuple[LLMWorkerInput, int, str | list[Message]]] = []
                for item in data_list:
                    if isinstance(item, LLMWorkerInput):
                        # Dead input
                        assert item.is_dead()
                        to_die.append((item, None))
                        del request_counter[item]
                    else:
                        # Alive input
                        inp, inp_data, data = item
                        assert inp_data is not None
                        content, i = data
                        llm_inputs.append(content)
                        configs.append(inp.generation_config)
                        request_metas.append((inp, inp_data, i))
                        if inp.cacheable:
                            to_cache.append((inp, i, content))
                if to_die:
                    await result_queue.put(to_die)
                if llm_inputs:
                    await self._llm.start_request_processing()

                    request_ids = await self._send_requests_to_llm(
                        llm_inputs, configs, with_usage
                    )
                    await request_tracker.put_all_no_wait(
                        dict(zip(request_ids, request_metas))
                    )

                    await self._llm.wait_available()
                if to_cache:
                    await caching_queue.put(to_cache)

            if not request_counter:
                # This fixes a race condition where the pulling loop is waiting
                # for new request outputs but there are no more requests.
                result_queue.close()
            # No more inputs, close the caching queue.
            caching_queue.close()

        async def pulling_loop() -> None:
            while request_counter:
                results = await handle_request_func(request_tracker)
                await result_queue.put(results)
                task_ends: list[tuple[LLMWorkerInput, TaskEnd]] = []
                for out_inp, _ in results:
                    request_counter[out_inp] -= 1
                    if request_counter[out_inp] == 0:
                        await handle_task_end_func(out_inp)
                        del request_counter[out_inp]
                        task_ends.append((out_inp, TASK_END))
                await result_queue.put(task_ends)
            result_queue.close()

        # Caching task should not be cancelled.
        caching_task = asyncio.create_task(caching_loop())
        tasks = [
            asyncio.create_task(coro()) for coro in (dispatching_loop, pulling_loop)
        ]
        try:
            while True:
                yield await result_queue.get()
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.wait(tasks)
            caching_task.cancel()
            await caching_task

    async def _send_requests_to_llm(
        self,
        inputs: list[str | list[Message]],
        configs: Sequence[GenerationConfig | None],
        with_usage: bool,
    ) -> list[str]:
        tokenizer = self._tokenizer
        if tokenizer is not None:
            chat_inputs = [inp for inp in inputs if isinstance(inp, list)]
            formatted_chat_inputs = apply_chat_template(
                tokenizer, chat_inputs, strip_begin_of_text=True
            )
            chat_inputs_iter = iter(formatted_chat_inputs)
            inputs = [
                inp if isinstance(inp, str) else next(chat_inputs_iter)
                for inp in inputs
            ]
        return await self._llm.add_requests(inputs, configs, with_usage)

    def _wrap_output(
        self,
        inp: WorkerInput,
        inp_data: Data,
        out: list[str] | list[Message],
        indices: list[int],
    ) -> Data:
        if isinstance(inp, LLMCompletionWorkerInput):
            return self._wrap_completion_output(
                inp, inp_data, cast(list[str], out), indices
            )
        elif isinstance(inp, LLMChatWorkerInput):
            return self._wrap_chat_output(inp, inp_data, out, indices)
        else:
            raise ValueError(f"Invalid input type: {type(inp)}")

    def _wrap_completion_output(
        self,
        inp: LLMCompletionWorkerInput,
        inp_data: Data,
        out: list[str],
        indices: list[int],
    ) -> Data:
        if not inp.echo:
            return Data.text(data=out, indices=indices)

        data = inp_data.get_by_indices(indices).as_text()
        new_text_data = [old + new for old, new in zip(data, out)]
        return Data.text(data=new_text_data, indices=indices)

    def _wrap_chat_output(
        self,
        inp: LLMChatWorkerInput,
        inp_data: Data,
        out: list[str] | list[Message],
        indices: list[int],
    ) -> Data:
        if not inp.return_history:
            if len(out) == 0:
                return Data.text(data=[], indices=indices)
            if isinstance(out[0], str):
                out = cast(list[str], out)
                return Data.text(data=out, indices=indices)
            out = cast(list[Message], out)
            return Data.text(data=[msg.content for msg in out], indices=indices)

        data = inp_data.get_by_indices(indices).as_message()
        if len(out) == 0:
            new_message_data = MessageData.empty()
        elif isinstance(out[0], str):
            out = cast(list[str], out)
            new_message_data = MessageData(self._llm.DEFAULT_LLM_ROLE, out)
        else:
            out = cast(list[Message], out)
            new_message_data = MessageData(
                role=out[0].role, content=[msg.content for msg in out]
            )
        data.append(new_message_data)
        return Data.message(data=data, indices=indices)

    async def _cache_results(
        self, inp: LLMWorkerInput, inp_data: Data, data: Data
    ) -> None:
        if self.manager.prompt_cache_manager is None:
            return

        cache_keys = inp.get_cache_keys(inp_data)
        if cache_keys is None:
            return

        batch_results: dict[Hashable, str] | dict[Hashable, list[Message]]
        if data.is_text():
            batch_results = {cache_keys[i]: out for i, out in data.iter_text()}
        else:
            batch_results = {cache_keys[i]: out for i, out in data.iter_message()}
        await self.cache_results(inp, batch_results)


_DispatchedItemType = (
    LLMWorkerInput
    | tuple[
        LLMWorkerInput,
        Data | None,
        tuple[str | list[Message], int],
    ]
)


class LLMRequestDispatcher:
    def __init__(self, worker: LLMWorker) -> None:
        self.worker = worker
        self.logger = init_child_logger(self.__class__.__name__, worker.logger)

        service_info = worker.service_config.info
        self._accumulation_window: float = service_info.accumulation_window
        self._max_accumulation_time: float = service_info.max_accumulation_time

    async def dispatch(
        self, inputs: LLMWorkerInputBatch
    ) -> AsyncGenerator[Sequence[_DispatchedItemType], None]:
        match inputs.dispatch_mode:
            case DispatchMode.RUNTIME_ADJUSTMENT:
                dispatch_func = self._dispatch_with_runtime_adjustment
            case DispatchMode.NO_RUNTIME_ADJUSTMENT:
                dispatch_func = self._dispatch_no_runtime_adjustment
            case DispatchMode.OP_WISE_NO_WAIT:
                dispatch_func = self._dispatch_op_wise_no_wait
        return dispatch_func(inputs)

    async def _dispatch_op_wise_no_wait(
        self, inputs: LLMWorkerInputBatch
    ) -> AsyncGenerator[Sequence[_DispatchedItemType], None]:
        schedule = inputs.schedule
        input_map = inputs.input_map
        for to_dispatch in schedule:
            input_to_dispatch = [input_map[op_id] for op_id in to_dispatch]
            async for resolved_inputs in self._resolve_inputs(input_to_dispatch):
                new_items: list[_DispatchedItemType] = []
                for inp in resolved_inputs:
                    if inp.is_dead():
                        new_items.append(inp)
                        continue
                    # Dispatch data
                    data = inp.data
                    for i, content in data.iter_content():
                        new_items.append((inp, data, (content, i)))
                yield new_items

    async def _dispatch_no_runtime_adjustment(
        self, inputs: LLMWorkerInputBatch
    ) -> AsyncGenerator[Sequence[_DispatchedItemType], None]:
        schedule = inputs.schedule
        input_map = inputs.input_map
        input_slice_map = inputs.input_slice_map

        # Inputs to dispatch
        dispatch_groups: list[
            tuple[list[LLMWorkerInput], Counter[int], dict[int, set[LLMWorkerInput]]]
        ] = []
        # op_id -> index counter
        index_counter_map: dict[str, Counter[int]] = {}
        # Data pools for each input
        data_pools: dict[str, AsyncPool[int, tuple[Data, str | list[Message]]]] = {}
        for to_dispatch in schedule:
            dispatch_group: list[LLMWorkerInput] = []
            index_counter: Counter[int] = Counter()
            index_input_map: dict[int, set[LLMWorkerInput]] = {}
            dispatch_groups.append((dispatch_group, index_counter, index_input_map))
            index_set: set[int] = set()
            for op_id in to_dispatch:
                inp = input_map[op_id]
                dispatch_group.append(inp)
                index_counter_map[op_id] = index_counter
                data_pools[op_id] = AsyncPool()
                input_indices = list(input_slice_map[op_id])
                index_set.update(input_indices)
                for i in input_indices:
                    if i in index_input_map:
                        index_input_map[i].add(inp)
                    else:
                        index_input_map[i] = {inp}
            # Initialize index counter
            index_counter.update({i: 0 for i in index_set})

        accumulation_tasks: set[asyncio.Task] = set()
        to_yield: list[_DispatchedItemType] = []
        for group_idx, (dispatch_group, index_counter, index_input_map) in enumerate(
            dispatch_groups
        ):
            # Dead inputs
            dead_inputs: list[LLMWorkerInput] = []
            # Condvar for notifying new data
            new_data_cond = asyncio.Condition()
            # Start an accumuation task for all ops
            accumulation_task = asyncio.create_task(
                self._accumulate_input_data(
                    dispatch_group,
                    data_pools,
                    index_counter_map,
                    dead_inputs,
                    new_data_cond,
                )
            )
            accumulation_tasks.add(accumulation_task)
            accumulation_task.add_done_callback(accumulation_tasks.discard)
            await asyncio.sleep(0)  # Allow accumulating task to run

            while index_counter and dispatch_group:
                # Yield if blocking
                if not (index_counter.total() > 0 or dead_inputs) and to_yield:
                    yield to_yield
                    to_yield = []

                # Wait for new index to be available
                async with new_data_cond:
                    await new_data_cond.wait_for(
                        lambda: index_counter.total() > 0 or dead_inputs
                    )

                if dead_inputs:
                    # Collect dead inputs
                    for inp in dead_inputs:
                        to_yield.append(inp)
                        for group, _, _ in dispatch_groups[group_idx:]:
                            if inp in group:
                                group.remove(inp)
                                break
                    dead_inputs.clear()

                if index_counter.total() > 0:
                    # Get the next index to dispatch
                    i = index_counter.most_common(1)[0][0]
                    del index_counter[i]

                    index_inputs = index_input_map[i]
                    for inp in dispatch_group:
                        if inp not in index_inputs:
                            # The input has been dispatched for this index
                            # TODO: Remove the input from the dispatch group when all
                            # of its indices are dispatched.
                            continue

                        data_pool = data_pools.get(inp.op_id)
                        if data_pool is None:
                            continue  # The input has died

                        item = await data_pool.pop_nowait(i)
                        if item is None:
                            # If no data, flush and wait
                            if to_yield:
                                yield to_yield
                                to_yield = []
                            item = await data_pool.pop(i)
                        data, text = item
                        to_yield.append((inp, data, (text, i)))

        # Yield the remaining inputs
        if to_yield:
            yield to_yield

        if accumulation_tasks:
            await asyncio.wait(accumulation_tasks)

    async def _dispatch_with_runtime_adjustment(
        self, inputs: LLMWorkerInputBatch
    ) -> AsyncGenerator[Sequence[_DispatchedItemType], None]:
        schedule = inputs.schedule
        input_map = inputs.input_map
        input_slice_map = inputs.input_slice_map

        # Inputs to dispatch
        dispatch_groups: list[
            tuple[list[LLMWorkerInput], Counter[int], dict[int, set[LLMWorkerInput]]]
        ] = []
        # op_id -> index counter
        index_counter_map: dict[str, Counter[int]] = {}
        # Data pools for each input
        data_pools: dict[str, AsyncPool[int, tuple[Data, str | list[Message]]]] = {}
        for to_dispatch in schedule:
            dispatch_group: list[LLMWorkerInput] = []
            index_counter: Counter[int] = Counter()
            index_input_map: dict[int, set[LLMWorkerInput]] = {}
            dispatch_groups.append((dispatch_group, index_counter, index_input_map))
            index_set: set[int] = set()
            for op_id in to_dispatch:
                inp = input_map[op_id]
                dispatch_group.append(inp)
                index_counter_map[op_id] = index_counter
                data_pools[op_id] = AsyncPool()
                input_indices = list(input_slice_map[op_id])
                index_set.update(input_indices)
                for i in input_indices:
                    if i in index_input_map:
                        index_input_map[i].add(inp)
                    else:
                        index_input_map[i] = {inp}
            # Initialize index counter
            index_counter.update({i: 0 for i in index_set})

        # Dead inputs
        dead_inputs: list[LLMWorkerInput] = []
        # Condvar for notifying new data
        new_data_cond = asyncio.Condition()
        # Start an accumuation task for all ops
        accumulation_task = asyncio.create_task(
            self._accumulate_input_data(
                cast(Sequence[LLMWorkerInput], inputs.inputs),
                data_pools,
                index_counter_map,
                dead_inputs,
                new_data_cond,
            )
        )
        await asyncio.sleep(0)  # Allow accumulating task to run

        def has_available_data(
            dispatch_groups: list[
                tuple[
                    list[LLMWorkerInput], Counter[int], dict[int, set[LLMWorkerInput]]
                ]
            ],
        ) -> bool:
            return any(
                index_counter.total() > 0 for _, index_counter, _ in dispatch_groups
            )

        accumulation_window = self._accumulation_window
        max_accumulation_time = self._max_accumulation_time

        to_yield: list[_DispatchedItemType] = []
        batch_start_time: float = time.monotonic()  # Timestamp
        while dispatch_groups:
            # Yield if blocking
            if (
                not (has_available_data(dispatch_groups) or dead_inputs)
                or (time.monotonic() - batch_start_time >= max_accumulation_time)
            ) and to_yield:
                yield to_yield
                to_yield = []
                batch_start_time = time.monotonic()

            # Wait for new index to be available
            async with new_data_cond:
                await new_data_cond.wait_for(
                    lambda: has_available_data(dispatch_groups) or dead_inputs
                )

            if dead_inputs:
                # Yield dead inputs
                for inp in dead_inputs:
                    to_yield.append(inp)
                    for dispatch_group, _, _ in dispatch_groups:
                        if inp in dispatch_group:
                            dispatch_group.remove(inp)
                dead_inputs.clear()
                # Remove empty dispatch groups
                dispatch_groups = [
                    group_counter
                    for group_counter in dispatch_groups
                    if group_counter[0]
                ]
                if not dispatch_groups:
                    break

            if not has_available_data(dispatch_groups):
                continue

            # Accumulate more data based on the accumulation window
            if accumulation_window > 0 and not dead_inputs:
                new_data_time = time.monotonic()
                cur_total = sum(
                    index_counter.total() for _, index_counter, _ in dispatch_groups
                )
                while True:
                    # Enforce hard accumulation time limit
                    if time.monotonic() - batch_start_time >= max_accumulation_time:
                        break
                    remaining = accumulation_window - (time.monotonic() - new_data_time)
                    if remaining <= 0:
                        break
                    try:
                        async with new_data_cond:
                            await asyncio.wait_for(
                                new_data_cond.wait(), timeout=remaining
                            )
                    except asyncio.TimeoutError:
                        break
                    if dead_inputs:
                        break  # Prioritize yielding dead inputs immediately
                    new_total = sum(
                        index_counter.total() for _, index_counter, _ in dispatch_groups
                    )
                    if new_total <= cur_total:
                        # No increase -> proceed to dispatch
                        break
                    # More data arrived; reset timer to allow further accumulation
                    cur_total = new_total
                    new_data_time = time.monotonic()

            # Try dispatching according to the schedule
            has_dispatched = False
            to_remove: list[int] = []
            for group_idx, (
                dispatch_group,
                index_counter,
                index_input_map,
            ) in enumerate(dispatch_groups):
                if index_counter.total() == 0:
                    if has_dispatched:
                        break  # Dispatch what we have now
                    continue  # Move to the next group

                while index_counter.total() > 0:
                    # Get the next index to dispatch
                    i = index_counter.most_common(1)[0][0]

                    index_inputs = index_input_map[i]
                    for inp in dispatch_group:
                        if inp not in index_inputs:
                            # The input has been dispatched for this index
                            # TODO: Remove the input from the dispatch group when all
                            # of its indices are dispatched.
                            continue

                        data_pool = data_pools.get(inp.op_id)
                        if data_pool is None:
                            continue  # The input has died.

                        item = await data_pool.pop_nowait(i)
                        if item is not None:
                            # Add data to yield
                            data, text = item
                            to_yield.append((inp, data, (text, i)))
                            # Mark the input as dispatched for this index
                            index_inputs.remove(inp)
                            index_counter[i] -= 1
                            # Mark that we have dispatched something
                            has_dispatched = True

                    if not index_inputs:
                        # If all inputs are dispatched for this index, remove it
                        del index_counter[i]
                        del index_input_map[i]

                if not index_counter:
                    # Mark the group for removal if all indices are dispatched
                    to_remove.append(group_idx)

            # Remove empty dispatch groups
            for idx in reversed(to_remove):
                del dispatch_groups[idx]
            if not dispatch_groups:
                break

        # Yield the remaining inputs
        if to_yield:
            yield to_yield

        await accumulation_task

    @log_on_exception_async(ignore=[asyncio.CancelledError])
    async def _accumulate_input_data(
        self,
        inputs: Sequence[LLMWorkerInput],
        data_pools: dict[str, AsyncPool[int, tuple[Data, str | list[Message]]]],
        index_counter_map: dict[str, Counter[int]],
        dead_inputs: list[LLMWorkerInput],
        new_data_cond: asyncio.Condition,
    ) -> None:
        """
        Parameters
        ----------
        inputs: list[LLMWorkerInput]
            List of inputs to resolve and accumulate data for.
        data_pools: dict[str, AsyncPool[int, tuple[Data, str | list[Message]]]]
            Mapping from op IDs to data pools (index -> (data, content)).
        index_counter_map: dict[str, Counter[int]]
            Mapping from op IDs to counters for the number of inputs whose data at
            each index is available.
        dead_inputs: list[LLMWorkerInput]
            List to put dead inputs into.
        new_data_cond: asyncio.Condition
            Conditional variable to notify when a new data is available.
        """
        async for resolved_inputs in self._resolve_inputs(inputs):
            new_data = False
            for inp in resolved_inputs:
                if inp.is_dead():
                    # Remove the input from the data pools
                    data_pools.pop(inp.op_id, None)
                    dead_inputs.append(inp)
                    new_data = True
                    continue

                index_counter = index_counter_map[inp.op_id]
                inp_data = inp.data
                data = inp_data
                # Add data to the data pool
                data_pool = data_pools[inp.op_id]
                new_items = {
                    i: (inp_data, content) for i, content in data.iter_content()
                }
                await data_pool.put_all_no_wait(new_items)
                for i in new_items:
                    if i in index_counter:
                        index_counter[i] += 1
                        new_data = True
            if new_data:
                async with new_data_cond:
                    new_data_cond.notify_all()

    async def _resolve_inputs(
        self, inputs: Sequence[LLMWorkerInput]
    ) -> AsyncGenerator[list[LLMWorkerInput], None]:
        pool: AsyncPool[LLMWorkerInput | None, None] = AsyncPool()
        task_counter: int = len(inputs)

        async def _resolve_input(inp: LLMWorkerInput) -> None:
            nonlocal task_counter

            async with inp.input_iterator(self.worker) as it:
                async for resolved_inp in it:
                    await pool.put(resolved_inp, None)
                    # Block until the input is consumed.
                    await pool.wait(resolved_inp)

            task_counter -= 1
            if task_counter == 0:
                await pool.put(None, None)

        tasks: list[asyncio.Task] = []
        for inp in inputs:
            if all(arg.is_available() for arg in inp.get_all_args()):
                await inp.resolve(self.worker)
                await pool.put(inp, None)
                task_counter -= 1
            else:
                tasks.append(asyncio.create_task(_resolve_input(inp)))
        if task_counter == 0:
            await pool.put(None, None)

        running = True
        while running:
            maybe_inputs = await pool.get_all()
            if None in maybe_inputs:
                running = False
            ready_inputs = [inp for inp in maybe_inputs if inp is not None]
            if len(ready_inputs) > 0:
                yield ready_inputs
            for maybe_input in maybe_inputs:
                await pool.pop_nowait(maybe_input)

        if tasks:
            await asyncio.wait(tasks)
        assert pool.empty()
