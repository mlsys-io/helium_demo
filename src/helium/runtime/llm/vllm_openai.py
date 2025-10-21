from typing import Any

from openai.types.completion import Completion

from helium.common import GenerationConfig, Message
from helium.runtime.llm import BatchChatMixin, LLMServiceConfig
from helium.runtime.llm.openai import (
    BatchChatRequest,
    BenchmarkRequest,
    BenchmarkResponse,
    ChatResponse,
    OpenAILLM,
)
from helium.runtime.llm.registry import LLMRegistry
from helium.runtime.llm.utils import apply_chat_template
from helium.runtime.utils.logger import get_default_logger

logger = get_default_logger()


@LLMRegistry.register("vllm-openai")
class VLLMOpenAILLM(OpenAILLM, BatchChatMixin):
    def __init__(self, *, config: LLMServiceConfig, **_) -> None:
        super().__init__(config=config)
        self._api_key: str | None = config.args.get("api_key")
        self._base_url: str | None
        if (host := config.args.get("host")) and (port := config.args.get("port")):
            self._base_url = f"http://{host}:{port}/v1"
        else:
            self._base_url = None

    async def start_benchmark(self, api_key: str | None, base_url: str | None) -> None:
        api_key = self._api_key or api_key
        base_url = self._base_url or base_url
        request = BenchmarkRequest.start(api_key, base_url)
        response = await self._send_to_client(request)
        assert isinstance(response, BenchmarkResponse)
        response.raise_for_error()

    async def stop_benchmark(
        self, api_key: str | None, base_url: str | None
    ) -> dict[str, Any]:
        api_key = self._api_key or api_key
        base_url = self._base_url or base_url
        request = BenchmarkRequest.stop(api_key, base_url)
        response = await self._send_to_client(request)
        assert isinstance(response, BenchmarkResponse)
        response.raise_for_error()
        metrics = response.metrics or {}
        return metrics

    def batch_chat(
        self,
        messages_list: list[list[Message]],
        config: GenerationConfig | None = None,
    ) -> list[Message]:
        config = self._get_config(config)
        client = self._get_sync_client(config.api_key, config.base_url)

        formatted_messages = apply_chat_template(config.model, messages_list)
        response: Completion = client.completions.create(
            prompt=formatted_messages, **config.openai_kwargs()
        )
        res_messages = [
            Message(role=self.DEFAULT_LLM_ROLE, content=choice.text)
            for choice in response.choices
        ]
        return res_messages

    async def batch_chat_async(
        self,
        messages_list: list[list[Message]],
        config: GenerationConfig | None = None,
    ) -> list[Message]:
        request = BatchChatRequest(
            messages_list=messages_list, config=config, with_usage=False
        )
        response = await self._send_to_client(request)
        assert isinstance(response, ChatResponse)
        return response.outputs

    def _get_config(self, config: GenerationConfig | None) -> GenerationConfig:
        if config is None:
            config = GenerationConfig.from_env(
                api_key=self._api_key, base_url=self._base_url
            )
        else:
            if self._api_key is not None:
                config.api_key = self._api_key
            if self._base_url is not None:
                config.base_url = self._base_url
        return config
