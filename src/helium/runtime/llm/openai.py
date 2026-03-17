import asyncio
import os
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, cast, overload

import backoff
import openai
from openai import AsyncOpenAI, OpenAI, RateLimitError
from openai.types.chat import ChatCompletion
from openai.types.completion import Completion

from helium.common import GenerationConfig, Message
from helium.runtime.llm import BaseLLM, BatchCompleteMixin, LLMServiceConfig, UsageInfo
from helium.runtime.llm.registry import LLMRegistry
from helium.runtime.llm.utils import AnyTokenizer, apply_chat_template, get_tokenizer
from helium.runtime.utils.loop import MPConcurrentEventLoop
from helium.runtime.utils.queue import AIOQueue
from helium.runtime.utils.vllm.utils import (
    async_request_metrics_openai,
    async_request_start_benchmark_openai,
    async_request_stop_benchmark_openai,
    get_metric_values,
    strip_v1_suffix,
)


class _SyncHttpxClientWrapper(openai.DefaultHttpxClient):
    """Borrowed from langchain_openai.chat_models._client_utils"""

    def __del__(self) -> None:
        if self.is_closed:
            return

        try:
            self.close()
        except Exception:
            pass


class _AsyncHttpxClientWrapper(openai.DefaultAsyncHttpxClient):
    """Borrowed from langchain_openai.chat_models._client_utils"""

    def __del__(self) -> None:
        if self.is_closed:
            return

        try:
            asyncio.get_running_loop().create_task(self.aclose())
        except Exception:
            pass


@lru_cache
def _get_default_httpx_client(
    base_url: str | None, timeout: Any
) -> _SyncHttpxClientWrapper:
    """Borrowed from langchain_openai.chat_models._client_utils"""
    return _SyncHttpxClientWrapper(
        base_url=base_url
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1",
        timeout=timeout,
    )


@lru_cache
def _get_default_async_httpx_client(
    base_url: str | None, timeout: Any
) -> _AsyncHttpxClientWrapper:
    """Borrowed from langchain_openai.chat_models._client_utils"""
    return _AsyncHttpxClientWrapper(
        base_url=base_url
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1",
        timeout=timeout,
    )


def _get_usage_info(response: Completion | ChatCompletion) -> UsageInfo:
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


@dataclass(slots=True)
class CompletionRequest:
    prompts: str | list[str]
    config: GenerationConfig | None
    with_usage: bool


@dataclass(slots=True)
class ChatRequest:
    messages: list[Message]
    config: GenerationConfig | None
    with_usage: bool


@dataclass(slots=True)
class BatchChatRequest:
    messages_list: list[list[Message]]
    config: GenerationConfig | None
    with_usage: bool


@dataclass(slots=True)
class BenchmarkRequest:
    is_start: bool
    api_key: str | None
    base_url: str | None

    @classmethod
    def start(cls, api_key: str | None, base_url: str | None) -> "BenchmarkRequest":
        return cls(is_start=True, api_key=api_key, base_url=base_url)

    @classmethod
    def stop(cls, api_key: str | None, base_url: str | None) -> "BenchmarkRequest":
        return cls(is_start=False, api_key=api_key, base_url=base_url)


@dataclass(slots=True)
class CompletionResponse:
    outputs: list[str]
    usage: UsageInfo | None

    @classmethod
    def from_response(
        cls, response: Completion, with_usage: bool
    ) -> "CompletionResponse":
        outputs = [choice.text for choice in response.choices]
        usage = _get_usage_info(response) if with_usage else None
        return cls(outputs=outputs, usage=usage)


@dataclass(slots=True)
class ChatResponse:
    outputs: list[Message]
    usage: UsageInfo | None

    @classmethod
    def from_response(
        cls, response: ChatCompletion, with_usage: bool
    ) -> "ChatResponse":
        outputs = []
        for choice in response.choices:
            message = choice.message
            if message.content is None:
                raise Exception("Invalid server response")
            outputs.append(Message(role=message.role, content=message.content))
        usage = _get_usage_info(response) if with_usage else None
        return cls(outputs=outputs, usage=usage)

    @classmethod
    def from_completion_response(
        cls, response: Completion, llm_role: str, with_usage: bool
    ) -> "ChatResponse":
        outputs = [
            Message(role=llm_role, content=choice.text) for choice in response.choices
        ]
        usage = _get_usage_info(response) if with_usage else None
        return cls(outputs=outputs, usage=usage)


@dataclass(slots=True)
class BenchmarkResponse:
    err: Exception | None
    metrics: dict[str, Any] | None

    @classmethod
    def ok(cls, metrics: dict[str, Any] | None = None) -> "BenchmarkResponse":
        return cls(err=None, metrics=metrics)

    @classmethod
    def error(cls, err: Exception) -> "BenchmarkResponse":
        return cls(err=err, metrics=None)

    def raise_for_error(self) -> None:
        if self.err is not None:
            raise self.err


RequestType = CompletionRequest | ChatRequest | BatchChatRequest | BenchmarkRequest
ResponseType = CompletionResponse | ChatResponse | BenchmarkResponse


class _AsyncOpenAIClient:
    def __init__(
        self,
        default_llm_role: str,
        get_config_func: Callable[[GenerationConfig | None], GenerationConfig],
    ) -> None:
        self.default_llm_role = default_llm_role
        self.get_config_func = get_config_func

    async def start_benchmark(self, request: BenchmarkRequest) -> BenchmarkResponse:
        try:
            api_key = request.api_key
            base_url = request.base_url
            if base_url is not None:
                base_url = strip_v1_suffix(base_url)
            client = self._get_async_client(api_key, base_url)
            await async_request_start_benchmark_openai(client)
            return BenchmarkResponse.ok()
        except Exception as e:
            return BenchmarkResponse.error(err=e)

    async def stop_benchmark(self, request: BenchmarkRequest) -> BenchmarkResponse:
        try:
            api_key = request.api_key
            base_url = request.base_url
            if base_url is not None:
                base_url = strip_v1_suffix(base_url)
            client = self._get_async_client(api_key, base_url)
            # Get inference benchmark metrics
            bench = await async_request_stop_benchmark_openai(client)

            # Get vLLM engine metrics
            engine_metrics = await async_request_metrics_openai(client)
            metrics = get_metric_values(engine_metrics)
            return BenchmarkResponse.ok(bench | metrics)
        except Exception as e:
            return BenchmarkResponse.error(err=e)

    @backoff.on_exception(
        backoff.expo, exception=RateLimitError, max_tries=10, max_time=60
    )
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        prompts = request.prompts
        config = self.get_config_func(request.config)

        client = self._get_async_client(config.api_key, config.base_url)

        response: Completion = await client.completions.create(
            prompt=prompts, **config.openai_kwargs()
        )
        return CompletionResponse.from_response(response, request.with_usage)

    @backoff.on_exception(
        backoff.expo, exception=RateLimitError, max_tries=10, max_time=60
    )
    async def chat(self, request: ChatRequest) -> ChatResponse:
        messages = request.messages
        config = self.get_config_func(request.config)

        client = self._get_async_client(config.api_key, config.base_url)

        formated_messages = [message.to_dict() for message in messages]
        response: ChatCompletion = await client.chat.completions.create(
            messages=formated_messages,  # type: ignore
            **config.openai_kwargs(),
        )
        return ChatResponse.from_response(response, request.with_usage)

    @backoff.on_exception(
        backoff.expo, exception=RateLimitError, max_tries=10, max_time=60
    )
    async def batch_chat(self, request: BatchChatRequest) -> ChatResponse:
        messages_list = request.messages_list
        config = self.get_config_func(request.config)

        client = self._get_async_client(config.api_key, config.base_url)
        if config.model is None:
            raise ValueError("Unknown model")

        formatted_messages = apply_chat_template(config.model, messages_list)
        response: Completion = await client.completions.create(
            prompt=formatted_messages, **config.openai_kwargs()
        )
        return ChatResponse.from_completion_response(
            response, self.default_llm_role, request.with_usage
        )

    @lru_cache
    def _get_async_client(self, api_key: str, base_url: str) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=_get_default_async_httpx_client(base_url, None),
        )


@asynccontextmanager
async def _client_context(
    default_llm_role: str,
    get_config_func: Callable[[GenerationConfig | None], GenerationConfig],
) -> AsyncGenerator[_AsyncOpenAIClient, None]:
    yield _AsyncOpenAIClient(default_llm_role, get_config_func)


async def _handle_client_request(
    event: RequestType, client: _AsyncOpenAIClient | None
) -> ResponseType:
    assert client is not None
    match event:
        case CompletionRequest():
            return await client.complete(event)
        case ChatRequest():
            return await client.chat(event)
        case BatchChatRequest():
            return await client.batch_chat(event)
        case BenchmarkRequest():
            if event.is_start:
                return await client.start_benchmark(event)
            return await client.stop_benchmark(event)


@LLMRegistry.register("openai")
class OpenAILLM(BaseLLM, BatchCompleteMixin):
    def __init__(self, *, config: LLMServiceConfig, **_) -> None:
        super().__init__(config=config)
        self._client_loop: (
            MPConcurrentEventLoop[RequestType, ResponseType, _AsyncOpenAIClient] | None
        ) = None

    @lru_cache
    def _get_sync_client(self, api_key: str, base_url: str) -> OpenAI:
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=_get_default_httpx_client(base_url, None),
        )

    def _get_config(self, config: GenerationConfig | None) -> GenerationConfig:
        if config is None:
            config = GenerationConfig.from_env()
        return config

    async def start(self) -> None:
        if self._client_loop is not None:
            await self._client_loop.start()
            return

        client_loop = MPConcurrentEventLoop(
            _handle_client_request,
            result_collector=AIOQueue(),
            context_manager=_client_context(self.DEFAULT_LLM_ROLE, self._get_config),
        )
        await client_loop.start()
        self._client_loop = client_loop

    async def stop(self) -> None:
        if self._client_loop is None:
            return
        await self._client_loop.stop()
        self._client_loop = None

    async def _send_to_client(self, request: RequestType) -> ResponseType:
        if self._client_loop is None or not self._client_loop.is_running():
            raise RuntimeError("Client loop is not running")
        return await self._client_loop.process_event(request)

    async def start_benchmark(self, api_key: str | None, base_url: str | None) -> None:
        pass

    async def stop_benchmark(
        self, api_key: str | None, base_url: str | None
    ) -> dict[str, Any]:
        return {}

    def get_tokenizer(self) -> AnyTokenizer | None:
        model = self.config.args.get("model")
        return None if model is None else get_tokenizer(model)

    async def add_requests(
        self,
        inputs: Sequence[str | list[Message]],
        configs: Sequence[GenerationConfig | None],
        with_usage: bool,
    ) -> list[str]:
        if len(inputs) != len(configs):
            raise ValueError("Inputs and configs must have the same length")
        if self._client_loop is None or not self._client_loop.is_running():
            raise ValueError("Client loop is not running")
        keys = []
        request: CompletionRequest | ChatRequest
        for inp, config in zip(inputs, configs):
            if isinstance(inp, str):
                request = CompletionRequest(inp, config, with_usage=with_usage)
            else:
                request = ChatRequest(inp, config, with_usage=with_usage)
            key = await self._client_loop.add_event(request)
            assert key is not None
            keys.append(key)
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
        if self._client_loop is None or not self._client_loop.is_running():
            raise ValueError("Client loop is not running")

        request_id, response = await self._client_loop.pop_result()
        if not isinstance(response, (CompletionResponse, ChatResponse)):
            raise ValueError("Invalid response type")

        if with_usage:
            if response.usage is None:
                raise ValueError("Usage info is not available")
            return request_id, response.outputs, response.usage
        return request_id, response.outputs

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
        if self._client_loop is None or not self._client_loop.is_running():
            raise ValueError("Client loop is not running")

        results = cast(
            list[tuple[str, CompletionResponse | ChatResponse]],
            await self._client_loop.pop_all_results(),
        )

        if with_usage:
            return [
                (request_id, response.outputs, response.usage)
                for request_id, response in results
                if response.usage is not None
            ]
        return [(request_id, response.outputs) for request_id, response in results]

    @backoff.on_exception(
        backoff.expo, exception=RateLimitError, max_tries=10, max_time=60
    )
    def complete(
        self, prompt: str, config: GenerationConfig | None = None
    ) -> list[str]:
        config = self._get_config(config)
        client = self._get_sync_client(config.api_key, config.base_url)

        response: Completion = client.completions.create(
            prompt=prompt, **config.openai_kwargs()
        )
        res_messages = [choice.text for choice in response.choices]
        return res_messages

    async def _complete_async(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        with_usage: bool = False,
    ) -> CompletionResponse:
        request = CompletionRequest(
            prompts=prompt, config=config, with_usage=with_usage
        )
        response = await self._send_to_client(request)
        assert isinstance(response, CompletionResponse)
        return response

    async def complete_async(
        self, prompt: str, config: GenerationConfig | None = None
    ) -> list[str]:
        response = await self._complete_async(prompt, config, with_usage=False)
        return response.outputs

    async def complete_async_with_usage(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> tuple[list[str], UsageInfo]:
        response = await self._complete_async(prompt, config, with_usage=True)
        assert response.usage is not None
        return response.outputs, response.usage

    @backoff.on_exception(
        backoff.expo, exception=RateLimitError, max_tries=10, max_time=60
    )
    def chat(
        self, messages: list[Message], config: GenerationConfig | None = None
    ) -> list[Message]:
        config = self._get_config(config)
        client = self._get_sync_client(config.api_key, config.base_url)

        formated_messages = [message.to_dict() for message in messages]
        response: ChatCompletion = client.chat.completions.create(
            messages=formated_messages,  # type: ignore
            **config.openai_kwargs(),
        )

        res_messages = []
        for choice in response.choices:
            message = choice.message
            if message.content is None:
                raise Exception("Invalid server response")
            res_messages.append(Message(role=message.role, content=message.content))

        return res_messages

    async def _chat_async(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
        with_usage: bool = False,
    ) -> ChatResponse:
        request = ChatRequest(messages=messages, config=config, with_usage=with_usage)
        response = await self._send_to_client(request)
        assert isinstance(response, ChatResponse)
        return response

    async def chat_async(
        self, messages: list[Message], config: GenerationConfig | None = None
    ) -> list[Message]:
        response = await self._chat_async(messages, config, with_usage=False)
        return response.outputs

    async def chat_async_with_usage(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> tuple[list[Message], UsageInfo]:
        response = await self._chat_async(messages, config, with_usage=True)
        assert response.usage is not None
        return response.outputs, response.usage

    def batch_complete(
        self, prompts: list[str], config: GenerationConfig | None = None
    ) -> list[str]:
        config = self._get_config(config)
        client = self._get_sync_client(config.api_key, config.base_url)

        response: Completion = client.completions.create(
            prompt=prompts, **config.openai_kwargs()
        )
        res_messages = [choice.text for choice in response.choices]
        return res_messages

    async def batch_complete_async(
        self, prompts: list[str], config: GenerationConfig | None = None
    ) -> list[str]:
        request = CompletionRequest(prompts=prompts, config=config, with_usage=False)
        response = await self._send_to_client(request)
        assert isinstance(response, CompletionResponse)
        return response.outputs
