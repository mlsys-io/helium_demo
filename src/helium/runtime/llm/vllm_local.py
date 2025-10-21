import asyncio
import os
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from typing import Any, Literal, overload

from helium import envs
from helium.common import GenerationConfig, Message
from helium.runtime.cache_manager.kv_cache import KVCacheClient, read_lmcache_config
from helium.runtime.llm import (
    BaseLLM,
    BatchChatMixin,
    BatchCompleteMixin,
    LLMServiceConfig,
    UsageInfo,
)
from helium.runtime.llm.registry import LLMRegistry
from helium.runtime.llm.utils import AnyTokenizer, apply_chat_template, get_tokenizer
from helium.runtime.utils.logger import log_on_exception_async
from helium.runtime.utils.loop import MPConcurrentEventLoop
from helium.runtime.utils.pool import AsyncPool
from helium.runtime.utils.queue import AIOQueue
from helium.runtime.utils.vllm.config import LocalVLLMServerConfig
from helium.runtime.utils.vllm.engine.local_v0 import (
    OpenAIEngineClient as OpenAIEngineClientV0,
)
from helium.runtime.utils.vllm.engine.local_v1 import (
    OpenAIEngineClient as OpenAIEngineClientV1,
)
from helium.runtime.utils.vllm.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    ErrorResponse,
    TokenizeChatRequest,
    TokenizeCompletionRequest,
    TokenizeResponse,
)
from helium.runtime.utils.vllm.openai.serving_utils import VLLMBenchmarker
from helium.runtime.utils.vllm.server import (
    CompiledServerConfig,
    configure_vllm_logging,
)
from helium.utils import run_coroutine_blocking, unique_id
from vllm.vllm.v1.metrics.stats import SchedulerStats


class StatRequest:
    pass


class StatResponse:
    __slots__ = ("stats",)

    def __init__(self, stats: dict[str, Any]) -> None:
        self.stats = stats


class ResetPrefixCacheRequest:
    pass


class ResetPrefixCacheResponse:
    pass


class ChangeKVRoleRequest:
    __slots__ = ("new_role",)

    def __init__(self, new_role: str) -> None:
        self.new_role = new_role


class ChangeKVRoleResponse:
    pass


class GetSchedulerStatsRequest:
    pass


class GetSchedulerStatsResponse:
    __slots__ = ("stats",)

    def __init__(self, stats: SchedulerStats) -> None:
        self.stats = stats


class ClearSchedulerStatsRequest:
    pass


class ClearSchedulerStatsResponse:
    pass


OpenAIEngineClientType = OpenAIEngineClientV0 | OpenAIEngineClientV1
LLMRequestType = CompletionRequest | ChatCompletionRequest
TokenRequestType = TokenizeCompletionRequest | TokenizeChatRequest
CacheRequestType = ResetPrefixCacheRequest | ChangeKVRoleRequest
StatRequestType = StatRequest | GetSchedulerStatsRequest | ClearSchedulerStatsRequest

NonLLMRequestType = TokenRequestType | CacheRequestType | StatRequestType
RequestType = LLMRequestType | NonLLMRequestType

LLMResponseType = CompletionResponse | ChatCompletionResponse
TokenResponseType = TokenizeResponse
CacheResponseType = ResetPrefixCacheResponse | ChangeKVRoleResponse
StatResponseType = (
    StatResponse | GetSchedulerStatsResponse | ClearSchedulerStatsResponse
)

NonLLMResponseType = TokenResponseType | CacheResponseType | StatResponseType
ResponseType = LLMResponseType | NonLLMResponseType


@asynccontextmanager
async def _engine_context(
    server_config: CompiledServerConfig,
) -> AsyncGenerator[OpenAIEngineClientType, None]:
    # Set device to run the model on.
    cuda_device = server_config.inner.cuda_device
    if cuda_device is not None:
        # Set device to run the model on.
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    configure_vllm_logging(server_config)

    # Set LMCache config file if provided
    if server_config.inner.lmcache_config_file is not None:
        os.environ["LMCACHE_CONFIG_FILE"] = str(server_config.inner.lmcache_config_file)

    OpenAIEngineClient: type[OpenAIEngineClientType]
    if server_config.inner.use_v1:
        os.environ["VLLM_USE_V1"] = "1"
        OpenAIEngineClient = OpenAIEngineClientV1
    else:
        os.environ["VLLM_USE_V1"] = "0"
        OpenAIEngineClient = OpenAIEngineClientV0
        server_config.args.disable_frontend_multiprocessing = True

    async with OpenAIEngineClient.build(
        server_config.args, server_config.inner.mock
    ) as engine_client:
        yield engine_client


async def _handle_engine_request(
    event: tuple[str, RequestType], engine_client: OpenAIEngineClientType | None
) -> ResponseType:
    assert engine_client is not None
    request_id, request = event
    response: ResponseType | ErrorResponse
    match request:
        case CompletionRequest():
            response = await engine_client.create_completion(request_id, request)
        case ChatCompletionRequest():
            response = await engine_client.create_chat_completion(request_id, request)
        case ResetPrefixCacheRequest():
            await engine_client.reset_prefix_cache()
            response = ResetPrefixCacheResponse()
        case ChangeKVRoleRequest():
            await engine_client.change_kv_role(request.new_role)
            response = ChangeKVRoleResponse()
        case TokenizeCompletionRequest() | TokenizeChatRequest():
            response = await engine_client.create_tokenize(request_id, request)
        case StatRequest():
            response = StatResponse(engine_client.get_stats())
        case GetSchedulerStatsRequest():
            stats = await engine_client.get_scheduler_stats()
            response = GetSchedulerStatsResponse(stats)
        case ClearSchedulerStatsRequest():
            await engine_client.clear_scheduler_stats()
            response = ClearSchedulerStatsResponse()
    if isinstance(response, ErrorResponse):
        raise ValueError(f"Engine error: {response}")
    return response


class EngineLoopHandler:
    def __init__(
        self,
        engine_loop: MPConcurrentEventLoop[
            tuple[str, RequestType], ResponseType, OpenAIEngineClientType
        ],
    ) -> None:
        self._engine_loop = engine_loop

        self._running = False
        self._pulling_loop_task: asyncio.Task[None] | None = None

        self._llm_pool: AsyncPool[str, LLMResponseType] = AsyncPool()
        self._non_llm_pool: AsyncPool[str, NonLLMResponseType] = AsyncPool()

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        await self._engine_loop.start()
        self._pulling_loop_task = asyncio.create_task(self._pulling_loop())

    async def stop(self) -> None:
        if not self._running:
            return
        assert self._pulling_loop_task is not None
        self._running = False
        await self._engine_loop.stop()
        await self._pulling_loop_task
        self._pulling_loop_task = None

    def is_running(self) -> bool:
        return self._running

    @log_on_exception_async(ignore=[asyncio.CancelledError])
    async def _pulling_loop(self) -> None:
        while self._running:
            for key, response in await self._engine_loop.pop_all_results():
                match response:
                    case CompletionResponse() | ChatCompletionResponse():
                        await self._llm_pool.put_nowait(key, response)
                    case _:
                        await self._non_llm_pool.put_nowait(key, response)

    async def add_llm_requests(
        self, requests: list[tuple[str, LLMRequestType]]
    ) -> list[str]:
        keys = await self._engine_loop.add_event_batch(requests)  # type: ignore
        assert keys is not None
        return keys

    async def process_llm_request(
        self, request_id: str, request: LLMRequestType
    ) -> LLMResponseType:
        key = await self._engine_loop.add_event((request_id, request))
        assert key is not None
        response = await self._llm_pool.pop(key)
        return response

    @overload
    async def process_non_llm_request(
        self, request_id: str, request: StatRequestType
    ) -> StatResponseType: ...

    @overload
    async def process_non_llm_request(
        self, request_id: str, request: TokenRequestType
    ) -> TokenResponseType: ...

    @overload
    async def process_non_llm_request(
        self, request_id: str, request: ResetPrefixCacheRequest
    ) -> ResetPrefixCacheResponse: ...

    @overload
    async def process_non_llm_request(
        self, request_id: str, request: ChangeKVRoleRequest
    ) -> ChangeKVRoleResponse: ...

    async def process_non_llm_request(
        self, request_id: str, request: NonLLMRequestType
    ) -> NonLLMResponseType:
        key = await self._engine_loop.add_event((request_id, request))
        assert key is not None
        response = await self._non_llm_pool.pop(key)
        return response

    async def get_all_llm(
        self,
    ) -> list[tuple[str, LLMResponseType]]:
        responses = await self._llm_pool.pop_all()
        return list(responses.items())

    async def get_llm(self, key: str) -> LLMResponseType:
        return await self._llm_pool.pop(key)

    async def pop_llm(self) -> tuple[str, LLMResponseType]:
        return await self._llm_pool.pop_first()


@LLMRegistry.register("vllm-local")
class VLLMLocalLLM(BaseLLM, BatchCompleteMixin, BatchChatMixin):
    def __init__(self, *, config: LLMServiceConfig, benchmarking: bool) -> None:
        super().__init__(config=config)
        service_args = config.args.copy()
        service_args["benchmarking"] = benchmarking
        self._server_config = LocalVLLMServerConfig(**service_args)
        self._engine_loop: EngineLoopHandler | None = None
        self._benchmarker = VLLMBenchmarker()
        self._tokenizer = get_tokenizer(self._server_config.model)
        self._busy_threshold = config.info.busy_threshold

        lmcache_config = read_lmcache_config(self._server_config.lmcache_config_file)
        self._cache_instance_id = lmcache_config.lmcache_instance_id  # type: ignore

    async def start(self) -> None:
        if self._engine_loop is not None:
            await self._engine_loop.start()
            return

        server_config = self._server_config.compile([])
        event_loop = MPConcurrentEventLoop(
            _handle_engine_request,
            result_collector=AIOQueue(),
            context_manager=_engine_context(server_config),
        )
        engine_loop = EngineLoopHandler(event_loop)
        await engine_loop.start()
        self._engine_loop = engine_loop

    async def stop(self) -> None:
        if self._engine_loop is None:
            return
        await self._engine_loop.stop()
        self._engine_loop = None

    def _get_request_id(self, request: RequestType) -> str:
        uid = unique_id()
        match request:
            case CompletionRequest():
                request_id = f"cmpl-{uid}"
            case ChatCompletionRequest():
                request_id = f"chatcmpl-{uid}"
            case TokenizeCompletionRequest() | TokenizeChatRequest():
                request_id = f"tokn-{uid}"
            case ResetPrefixCacheRequest():
                request_id = f"rset-{uid}"
            case ChangeKVRoleRequest():
                request_id = f"chkv-{uid}"
            case StatRequest():
                request_id = f"stat-{uid}"
            case GetSchedulerStatsRequest():
                request_id = f"getschd-{uid}"
            case ClearSchedulerStatsRequest():
                request_id = f"clrschd-{uid}"
        return request_id

    async def _process_llm_request(self, request: LLMRequestType) -> LLMResponseType:
        if self._engine_loop is None or not self._engine_loop.is_running():
            raise ValueError("Engine is not running")

        request_id = self._get_request_id(request)

        response = await self._engine_loop.process_llm_request(request_id, request)
        if response.metrics is not None:
            if isinstance(response.metrics, list):
                for m in response.metrics:
                    self._benchmarker.add_metrics(m)
            else:
                self._benchmarker.add_metrics(response.metrics)
        return response

    @overload
    async def _process_non_llm_request(
        self, request: StatRequestType
    ) -> StatResponseType: ...

    @overload
    async def _process_non_llm_request(
        self, request: TokenRequestType
    ) -> TokenResponseType: ...

    @overload
    async def _process_non_llm_request(
        self, request: ResetPrefixCacheRequest
    ) -> ResetPrefixCacheResponse: ...

    @overload
    async def _process_non_llm_request(
        self, request: ChangeKVRoleRequest
    ) -> ChangeKVRoleResponse: ...

    async def _process_non_llm_request(
        self, request: NonLLMRequestType
    ) -> NonLLMResponseType:
        if self._engine_loop is None or not self._engine_loop.is_running():
            raise ValueError("Engine is not running")

        request_id = self._get_request_id(request)

        response = await self._engine_loop.process_non_llm_request(request_id, request)
        return response

    async def start_benchmark(self, api_key: str | None, base_url: str | None) -> None:
        if self._engine_loop is None or not self._engine_loop.is_running():
            raise ValueError("Engine is not running")
        self._benchmarker.start_benchmark()

    async def stop_benchmark(
        self, api_key: str | None, base_url: str | None
    ) -> dict[str, Any]:
        if self._engine_loop is None or not self._engine_loop.is_running():
            raise ValueError("Engine is not running")
        bench = self._benchmarker.stop_benchmark()
        stats = await self._get_stats()
        return bench | stats

    def get_tokenizer(self) -> AnyTokenizer | None:
        return self._tokenizer

    async def add_requests(
        self,
        inputs: Sequence[str | list[Message]],
        configs: Sequence[GenerationConfig | None],
        with_usage: bool,
        is_precompute: bool = False,
    ) -> list[str]:
        if self._engine_loop is None or not self._engine_loop.is_running():
            raise ValueError("Engine is not running")
        if len(inputs) != len(configs):
            raise ValueError("Inputs and configs must have the same length")
        if len(inputs) == 0:
            return []
        events: list[tuple[str, LLMRequestType]] = []
        priority = 1 if is_precompute else 0  # For KV cache precomputation
        request: LLMRequestType
        for inp, config in zip(inputs, configs):
            if config is None:
                config = GenerationConfig.from_env()
            if isinstance(inp, str):
                request = CompletionRequest(
                    prompt=inp, **config.vllm_kwargs(), priority=priority
                )
            else:
                request = ChatCompletionRequest(
                    messages=[message.to_dict() for message in inp],
                    **config.vllm_kwargs(),
                    priority=priority,
                )
            request_id = self._get_request_id(request)
            events.append((request_id, request))
        keys = await self._engine_loop.add_llm_requests(events)
        assert keys is not None
        return keys

    @overload
    async def get_request_output(
        self, with_usage: Literal[False]
    ) -> tuple[str, list[str] | list[Message]]: ...

    @overload
    async def get_request_output(
        self, with_usage: Literal[True]
    ) -> tuple[str, list[str] | list[Message], UsageInfo]: ...

    async def get_request_output(
        self, with_usage: bool
    ) -> (
        tuple[str, list[str] | list[Message]]
        | tuple[str, list[str] | list[Message], UsageInfo]
    ):
        if self._engine_loop is None or not self._engine_loop.is_running():
            raise ValueError("Engine is not running")

        request_id, response = await self._engine_loop.pop_llm()
        if not isinstance(response, (CompletionResponse, ChatCompletionResponse)):
            raise ValueError("Invalid response type")

        if response.metrics is not None:
            if isinstance(response.metrics, list):
                for m in response.metrics:
                    self._benchmarker.add_metrics(m)
            else:
                self._benchmarker.add_metrics(response.metrics)

        outputs = (
            self._format_completion_response(response)
            if isinstance(response, CompletionResponse)
            else self._format_chat_response(response)
        )

        if with_usage:
            usage_info = self._get_usage_info(response)
            return request_id, outputs, usage_info
        return request_id, outputs

    @overload
    async def get_available_outputs(
        self, with_usage: Literal[False]
    ) -> list[tuple[str, list[str] | list[Message]]]: ...

    @overload
    async def get_available_outputs(
        self, with_usage: Literal[True]
    ) -> list[tuple[str, list[str] | list[Message], UsageInfo]]: ...

    async def get_available_outputs(
        self, with_usage: bool
    ) -> (
        list[tuple[str, list[str] | list[Message]]]
        | list[tuple[str, list[str] | list[Message], UsageInfo]]
    ):
        if self._engine_loop is None or not self._engine_loop.is_running():
            raise ValueError("Engine is not running")

        results = await self._engine_loop.get_all_llm()
        if with_usage:
            ret_with_usage: list[tuple[str, list[str] | list[Message], UsageInfo]] = []
            for request_id, response in results:
                if response.metrics is not None:
                    if isinstance(response.metrics, list):
                        for m in response.metrics:
                            self._benchmarker.add_metrics(m)
                    else:
                        self._benchmarker.add_metrics(response.metrics)
                outputs = (
                    self._format_completion_response(response)
                    if isinstance(response, CompletionResponse)
                    else self._format_chat_response(response)
                )
                usage_info = self._get_usage_info(response)
                ret_with_usage.append((request_id, outputs, usage_info))
            return ret_with_usage
        ret: list[tuple[str, list[str] | list[Message]]] = []
        for request_id, response in results:
            if response.metrics is not None:
                if isinstance(response.metrics, list):
                    for m in response.metrics:
                        self._benchmarker.add_metrics(m)
                else:
                    self._benchmarker.add_metrics(response.metrics)
            outputs = (
                self._format_completion_response(response)
                if isinstance(response, CompletionResponse)
                else self._format_chat_response(response)
            )
            ret.append((request_id, outputs))
        return ret

    def complete(
        self, prompt: str, config: GenerationConfig | None = None
    ) -> list[str]:
        return run_coroutine_blocking(self.complete_async(prompt, config))

    async def _complete_async(
        self, prompt: str, config: GenerationConfig | None = None
    ) -> tuple[list[str], CompletionResponse]:
        if config is None:
            config = GenerationConfig.from_env()

        request = CompletionRequest(prompt=prompt, **config.vllm_kwargs())
        response = await self._process_llm_request(request)
        assert isinstance(response, CompletionResponse)
        outputs = self._format_completion_response(response)
        return outputs, response

    async def complete_async(
        self, prompt: str, config: GenerationConfig | None = None
    ) -> list[str]:
        outputs, _ = await self._complete_async(prompt, config)
        return outputs

    async def complete_async_with_usage(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> tuple[list[str], UsageInfo]:
        outputs, response = await self._complete_async(prompt, config)
        usage = self._get_usage_info(response)
        return outputs, usage

    def chat(
        self, messages: list[Message], config: GenerationConfig | None = None
    ) -> list[Message]:
        return run_coroutine_blocking(self.chat_async(messages, config))

    async def _chat_async(
        self, messages: list[Message], config: GenerationConfig | None = None
    ) -> tuple[list[Message], ChatCompletionResponse]:
        if config is None:
            config = GenerationConfig.from_env()
        formated_messages = [message.to_dict() for message in messages]
        request = ChatCompletionRequest(messages=formated_messages, **config.vllm_kwargs())  # type: ignore
        response = await self._process_llm_request(request)
        assert isinstance(response, ChatCompletionResponse)
        outputs = self._format_chat_response(response)
        return outputs, response

    async def chat_async(
        self, messages: list[Message], config: GenerationConfig | None = None
    ) -> list[Message]:
        outputs, _ = await self._chat_async(messages, config)
        return outputs

    async def chat_async_with_usage(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> tuple[list[Message], UsageInfo]:
        outputs, response = await self._chat_async(messages, config)
        usage = self._get_usage_info(response)
        return outputs, usage

    async def batch_complete_async(
        self, prompts: list[str], config: GenerationConfig | None = None
    ) -> list[str]:
        if config is None:
            config = GenerationConfig.from_env()

        request = CompletionRequest(prompt=prompts, **config.vllm_kwargs())
        response = await self._process_llm_request(request)
        assert isinstance(response, CompletionResponse)
        outputs = self._format_completion_response(response)
        return outputs

    def batch_chat(
        self,
        messages_list: list[list[Message]],
        config: GenerationConfig | None = None,
    ) -> list[Message]:
        return run_coroutine_blocking(self.batch_chat_async(messages_list, config))

    async def batch_chat_async(
        self,
        messages_list: list[list[Message]],
        config: GenerationConfig | None = None,
    ) -> list[Message]:
        if config is None:
            config = GenerationConfig.from_env()

        formatted_messages = apply_chat_template(config.model, messages_list)
        request = CompletionRequest(prompt=formatted_messages, **config.vllm_kwargs())  # type: ignore
        response = await self._process_llm_request(request)
        assert isinstance(response, CompletionResponse)
        outputs = [
            Message(role=self.DEFAULT_LLM_ROLE, content=choice.text)
            for choice in response.to_openai().choices
        ]
        return outputs

    async def tokenize(self, prompt: str | list[Message]) -> list[int]:
        request: TokenRequestType
        if isinstance(prompt, str):
            request = TokenizeCompletionRequest(prompt=prompt)
        else:
            request = TokenizeChatRequest(
                messages=[msg.to_dict() for msg in prompt],
                add_generation_prompt=True,
                add_special_tokens=False,
                chat_template_kwargs={
                    "enable_thinking": envs.HELIUM_VLLM_ENABLE_THINKING
                },
            )

        response = await self._process_non_llm_request(request)
        return response.tokens

    async def precompute_kv_cache(
        self,
        prompts: list[str | list[Message]],
        config: GenerationConfig,
        client: KVCacheClient,
    ) -> None:
        # GPU precomputation
        keys = await self.add_requests(
            inputs=prompts,
            configs=[config] * len(prompts),
            with_usage=False,
            is_precompute=True,
        )
        assert self._engine_loop is not None
        for key in keys:
            await self._engine_loop.get_llm(key)

        # LMCache pinning
        # for prompt in prompts:
        #     if isinstance(prompt, str):
        #         await self.complete_async(prompt, config)
        #     else:
        #         await self.chat_async(prompt, config)
        #     token_ids = await self.tokenize(prompt)
        #     await client.pin(self._cache_instance_id, token_ids)

    async def reset_prefix_cache(self) -> None:
        await self._process_non_llm_request(ResetPrefixCacheRequest())

    async def clear_kv_cache(self, client: KVCacheClient) -> None:
        await client.clear(self._cache_instance_id)

    async def change_kv_role(self, new_role: str) -> None:
        await self._process_non_llm_request(ChangeKVRoleRequest(new_role))

    async def start_request_processing(self) -> None:
        await self._process_non_llm_request(ClearSchedulerStatsRequest())

    async def wait_available(self) -> None:
        while True:
            response = await self._process_non_llm_request(GetSchedulerStatsRequest())
            assert isinstance(response, GetSchedulerStatsResponse)
            stats = response.stats
            if stats.num_waiting_reqs <= self._busy_threshold:
                break

    async def _get_stats(self) -> dict[str, Any]:
        resp = await self._process_non_llm_request(StatRequest())
        if not isinstance(resp, StatResponse):
            raise ValueError("Invalid response type")
        return resp.stats

    def _get_usage_info(self, response: LLMResponseType) -> UsageInfo:
        res_usage = response.usage
        return (
            UsageInfo()
            if res_usage is None
            else UsageInfo(
                prompt_tokens=res_usage.prompt_tokens,
                output_tokens=res_usage.completion_tokens,
                total_tokens=res_usage.total_tokens,
            )
        )

    def _format_completion_response(self, response: CompletionResponse) -> list[str]:
        return [choice.text for choice in response.to_openai().choices]

    def _format_chat_response(self, response: ChatCompletionResponse) -> list[Message]:
        outputs = []
        for choice in response.to_openai().choices:
            message = choice.message
            if message.content is None:
                raise Exception("Invalid server response")
            outputs.append(Message(role=message.role, content=message.content))
        return outputs
