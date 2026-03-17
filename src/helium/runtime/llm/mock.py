from collections.abc import Sequence
from typing import Any, Literal, overload

from helium import envs
from helium.common import GenerationConfig, Message
from helium.runtime.cache_manager.kv_cache import KVCacheClient
from helium.runtime.llm import LLMServiceConfig, UsageInfo
from helium.runtime.llm.registry import LLMRegistry
from helium.runtime.llm.utils import AnyTokenizer
from helium.runtime.utils.logger import get_debug_logger
from helium.runtime.utils.queue import AIOQueue, AsyncQueue


@LLMRegistry.register("mock")
class MockLLM:
    DEFAULT_LLM_ROLE: str = "assistant"

    def __init__(self, *, config: LLMServiceConfig, **_) -> None:
        self.config = config
        self._verbose = config.args.get("verbose", envs.DEBUG_MOCK_LLM_VERBOSE)
        self._counter = 0
        self._results: AsyncQueue[
            tuple[str, list[str] | list[Message], UsageInfo | None]
        ] = AIOQueue()

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def start_benchmark(self, api_key: str, base_url: str) -> None:
        pass

    async def stop_benchmark(self, api_key: str, base_url: str) -> dict[str, Any]:
        return {}

    def get_tokenizer(self) -> AnyTokenizer | None:
        return None

    async def add_requests(
        self,
        inputs: Sequence[str | list[Message]],
        configs: Sequence[GenerationConfig | None],
        with_usage: bool,
    ) -> list[str]:
        if len(inputs) != len(configs):
            raise ValueError("Inputs and configs must have the same length")

        if self._verbose:
            for inp in inputs:
                get_debug_logger().debug("Received input: %r", inp)

        request_ids = []
        for inp in inputs:
            self._counter += 1
            request_id = f"request_{self._counter}"
            request_ids.append(request_id)
            usage_info = UsageInfo() if with_usage else None
            output: list[str] | list[Message]
            if isinstance(inp, str):
                output = ["MOCK"]
            else:
                output = [Message(role="assistant", content="MOCK")]
            await self._results.put((request_id, output, usage_info))

        return request_ids

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
        request_id, outputs, usage_info = await self._results.get()
        if with_usage:
            if usage_info is None:
                raise ValueError("Usage info is not available")
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
        results = await self._results.get_all()
        if with_usage:
            return results  # type: ignore
        return [(request_id, outputs) for request_id, outputs, _ in results]

    def complete(
        self, prompt: str, config: GenerationConfig | None = None
    ) -> list[str]:
        return ["MOCK"]

    async def complete_async(
        self, prompt: str, config: GenerationConfig | None = None
    ) -> list[str]:
        return self.complete(prompt, config)

    async def complete_async_with_usage(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> tuple[list[str], UsageInfo]:
        return self.complete(prompt, config), UsageInfo()

    def chat(
        self, messages: list[Message], config: GenerationConfig | None = None
    ) -> list[Message]:
        return [Message(role="assistant", content="MOCK")]

    async def chat_async(
        self, messages: list[Message], config: GenerationConfig | None = None
    ) -> list[Message]:
        return self.chat(messages, config)

    async def chat_async_with_usage(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> tuple[list[Message], UsageInfo]:
        return self.chat(messages, config), UsageInfo()

    async def tokenize(self, prompt: str | list[Message]) -> list[int]:
        return [0]

    async def precompute_kv_cache(
        self,
        prompts: list[str | list[Message]],
        config: GenerationConfig,
        client: KVCacheClient,
    ) -> None:
        pass

    async def reset_prefix_cache(self) -> None:
        pass

    async def clear_kv_cache(self, client: KVCacheClient) -> None:
        pass

    async def change_kv_role(self, new_role: str) -> None:
        pass

    async def start_request_processing(self) -> None:
        return

    async def wait_available(self) -> None:
        return
