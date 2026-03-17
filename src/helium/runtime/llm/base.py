from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Self, overload

import numpy as np

from helium import envs
from helium.common import GenerationConfig, Message
from helium.runtime.cache_manager.kv_cache import KVCacheClient
from helium.runtime.llm.utils import AnyTokenizer


@dataclass(slots=True)
class LLMServiceInfo:
    cache_capacity: int = 0
    max_num_reqs: int = 1
    max_num_batched_tokens: int = 0
    alpha: float = 1
    is_memory_limited: bool = False
    prefix_caching_enabled: bool = False
    busy_threshold: int = 10
    # busy_threshold: int = 999999999   # Set this when disabling CAS

    accumulation_window: float = 0
    """Accumulation window in seconds.
    Allows short delay to accumulate more indices before dispatching, improving 
    batch size. Set to 0 to disable.
    """
    max_accumulation_time: float = 0
    """Maximum accumulation time before forcing a yield"""

    def __post_init__(self) -> None:
        self.is_memory_limited = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            cache_capacity=data.get("cache_capacity", 0),
            max_num_reqs=data.get("max_num_reqs", 1),
            max_num_batched_tokens=data.get("max_num_batched_tokens", 0),
            alpha=data.get("alpha", 1),
            is_memory_limited=data.get("is_memory_limited", False),
            prefix_caching_enabled=data.get("prefix_caching_enabled", False),
        )

    @property
    def token_budget(self) -> int:
        if self.is_memory_limited:
            return self.cache_capacity
        return self.max_num_batched_tokens


@dataclass
class LLMServiceConfig:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    info: LLMServiceInfo = field(default_factory=LLMServiceInfo)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        name = data["name"]
        args = data.get("args", {})
        info = LLMServiceInfo.from_dict(data.get("info", {}))
        return cls(name=name, args=args, info=info)


@dataclass(slots=True)
class UsageInfo:
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class LLMProfilingInfo:
    request_count: int
    prompt_tokens_avg: float
    prompt_tokens_std: float
    output_tokens_avg: float
    output_tokens_std: float
    total_tokens_avg: float
    total_tokens_std: float

    @classmethod
    def aggregate(cls, usage_infos: list[UsageInfo]) -> "LLMProfilingInfo":
        def stats(attr: str) -> tuple[float, float]:
            values = [
                getattr(info, attr)
                for info in usage_infos
                if getattr(info, attr) is not None
            ]
            if not values:
                return 0.0, 0.0
            return float(np.mean(values)), float(np.std(values))

        prompt_avg, prompt_std = stats("prompt_tokens")
        output_avg, output_std = stats("output_tokens")
        total_avg, total_std = stats("total_tokens")

        return cls(
            request_count=len(usage_infos),
            prompt_tokens_avg=prompt_avg,
            prompt_tokens_std=prompt_std,
            output_tokens_avg=output_avg,
            output_tokens_std=output_std,
            total_tokens_avg=total_avg,
            total_tokens_std=total_std,
        )

    @classmethod
    def merge(cls, profiling_info_list: list["LLMProfilingInfo"]) -> "LLMProfilingInfo":
        if not profiling_info_list:
            return cls.default()

        request_count = 0
        prompt_tokens_avg = 0.0
        prompt_tokens_std = 0.0
        output_tokens_avg = 0.0
        output_tokens_std = 0.0
        total_tokens_avg = 0.0
        total_tokens_std = 0.0

        for info in profiling_info_list:
            request_count += info.request_count
            prompt_tokens_avg += info.prompt_tokens_avg * info.request_count
            if prompt_tokens_std < info.prompt_tokens_std:
                prompt_tokens_std = info.prompt_tokens_std
            output_tokens_avg += info.output_tokens_avg * info.request_count
            if output_tokens_std < info.output_tokens_std:
                output_tokens_std = info.output_tokens_std
            total_tokens_avg += info.total_tokens_avg * info.request_count
            if total_tokens_std < info.total_tokens_std:
                total_tokens_std = info.total_tokens_std

        request_count = request_count
        prompt_tokens_avg /= request_count
        output_tokens_avg /= request_count
        total_tokens_avg /= request_count

        return cls(
            request_count=request_count,
            prompt_tokens_avg=prompt_tokens_avg,
            prompt_tokens_std=prompt_tokens_std,
            output_tokens_avg=output_tokens_avg,
            output_tokens_std=output_tokens_std,
            total_tokens_avg=total_tokens_avg,
            total_tokens_std=total_tokens_std,
        )

    @classmethod
    def default(cls) -> "LLMProfilingInfo":
        return cls(
            request_count=0,
            prompt_tokens_avg=0.0,
            prompt_tokens_std=0.0,
            output_tokens_avg=0.0,
            output_tokens_std=0.0,
            total_tokens_avg=0.0,
            total_tokens_std=0.0,
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, json_obj: dict[str, Any]) -> "LLMProfilingInfo":
        return cls(**json_obj)


class BaseLLM:
    DEFAULT_LLM_ROLE: str = "assistant"

    def __init__(self, *, config: LLMServiceConfig, **_) -> None:
        self.config = config

    @abstractmethod
    async def start(self) -> None:
        pass

    @abstractmethod
    async def stop(self) -> None:
        pass

    @abstractmethod
    async def start_benchmark(self, api_key: str | None, base_url: str | None) -> None:
        pass

    @abstractmethod
    async def stop_benchmark(
        self, api_key: str | None, base_url: str | None
    ) -> dict[str, Any]:
        pass

    @abstractmethod
    def get_tokenizer(self) -> AnyTokenizer | None:
        pass

    async def add_requests(
        self,
        inputs: Sequence[str | list[Message]],
        configs: Sequence[GenerationConfig | None],
        with_usage: bool,
    ) -> list[str]:
        raise NotImplementedError()

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
        raise NotImplementedError()

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
        raise NotImplementedError()

    @abstractmethod
    def complete(
        self, prompt: str, config: GenerationConfig | None = None
    ) -> list[str]:
        pass

    @abstractmethod
    async def complete_async(
        self, prompt: str, config: GenerationConfig | None = None
    ) -> list[str]:
        pass

    @abstractmethod
    async def complete_async_with_usage(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> tuple[list[str], UsageInfo]:
        pass

    @abstractmethod
    def chat(
        self, messages: list[Message], config: GenerationConfig | None = None
    ) -> list[Message]:
        pass

    @abstractmethod
    async def chat_async(
        self, messages: list[Message], config: GenerationConfig | None = None
    ) -> list[Message]:
        pass

    @abstractmethod
    async def chat_async_with_usage(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> tuple[list[Message], UsageInfo]:
        pass

    async def tokenize(self, prompt: str | list[Message]) -> list[int]:
        tokenizer = self.get_tokenizer()
        if tokenizer is None:
            raise ValueError("Tokenizer is not available")

        if isinstance(prompt, str):
            return tokenizer.encode(prompt)
        else:
            return tokenizer.apply_chat_template(  # type: ignore
                [message.to_dict() for message in prompt],
                tokenize=True,
                add_generation_prompt=True,
                add_special_tokens=False,
                enable_thinking=envs.HELIUM_VLLM_ENABLE_THINKING,
            )

    async def precompute_kv_cache(
        self,
        prompts: list[str | list[Message]],
        config: GenerationConfig,
        client: KVCacheClient,
    ) -> None:
        """Precomputes the KV cache for the given prompts and stores them in the KV cache."""
        pass

    async def reset_prefix_cache(self) -> None:
        """Resets the prefix cache in the LLM, if supported."""
        pass

    async def clear_kv_cache(self, client: KVCacheClient) -> None:
        """Clears the KV cache in the LLM, if supported."""
        pass

    async def change_kv_role(self, new_role: str) -> None:
        """Changes the KV role of the LLM, if supported."""
        pass

    async def start_request_processing(self) -> None:
        """Marks the start of request processing

        This is used to clear scheduler stats before processing requests.
        """
        return

    async def wait_available(self) -> None:
        """Waits until the LLM is available (not busy processing requests)."""
        return


class BatchCompleteMixin:
    @abstractmethod
    def batch_complete(
        self, prompts: list[str], config: GenerationConfig | None = None
    ) -> list[str]:
        pass

    @abstractmethod
    async def batch_complete_async(
        self, prompts: list[str], config: GenerationConfig | None = None
    ) -> list[str]:
        pass


class BatchChatMixin:
    @abstractmethod
    def batch_chat(
        self,
        messages_list: list[list[Message]],
        config: GenerationConfig | None = None,
    ) -> list[Message]:
        pass

    @abstractmethod
    async def batch_chat_async(
        self,
        messages_list: list[list[Message]],
        config: GenerationConfig | None = None,
    ) -> list[Message]:
        pass
