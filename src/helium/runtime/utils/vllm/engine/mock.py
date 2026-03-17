from argparse import Namespace
from collections.abc import AsyncGenerator, Iterable, Mapping
from typing import Any

from helium.runtime.utils.vllm.vllm_logger import init_child_logger
from vllm.vllm.config import VllmConfig
from vllm.vllm.engine.arg_utils import AsyncEngineArgs
from vllm.vllm.engine.protocol import EngineClient
from vllm.vllm.inputs.data import PromptType, StreamingInput
from vllm.vllm.inputs.preprocess import InputPreprocessor
from vllm.vllm.lora.request import LoRARequest
from vllm.vllm.outputs import CompletionOutput, PoolingRequestOutput, RequestOutput
from vllm.vllm.pooling_params import PoolingParams
from vllm.vllm.renderers.inputs import DictPrompt, TokPrompt
from vllm.vllm.renderers.protocol import BaseRenderer
from vllm.vllm.sampling_params import SamplingParams
from vllm.vllm.tokenizers.protocol import TokenizerLike
from vllm.vllm.usage.usage_lib import UsageContext
from vllm.vllm.v1.engine import EngineCoreRequest
from vllm.vllm.v1.engine.input_processor import InputProcessor
from vllm.vllm.v1.metrics.stats import SchedulerStats

logger = init_child_logger("mock")


class MockLLMEngine(EngineClient):
    def __init__(self, vllm_config: VllmConfig) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.io_processor = None

        self.input_processor = InputProcessor(vllm_config=vllm_config)
        self.processor = self.input_processor

        logger.warning("Initialized MockLLMEngine")

    @classmethod
    def from_cli_args(cls, args: Namespace) -> "MockLLMEngine":
        engine_args = AsyncEngineArgs.from_cli_args(args)
        vllm_config = engine_args.create_engine_config(
            usage_context=UsageContext.OPENAI_API_SERVER
        )
        return cls(vllm_config)

    async def generate(
        self,
        prompt: (
            EngineCoreRequest
            | PromptType
            | DictPrompt
            | TokPrompt
            | AsyncGenerator[StreamingInput, None]
        ),
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        if isinstance(prompt, dict):
            maybe_prompt = prompt.get("prompt")
            if maybe_prompt is None:
                token_ids = prompt.get("prompt_token_ids")
                if token_ids is None:
                    raise ValueError(
                        "Unsupported prompt format: missing 'prompt' or 'prompt_token_ids'"
                    )
                tokenizer = self.get_tokenizer()
                prompt = tokenizer.decode(token_ids, skip_special_tokens=False)  # type: ignore
                assert isinstance(prompt, str)
            elif isinstance(maybe_prompt, str):
                prompt = maybe_prompt
            else:
                raise ValueError(f"Unsupported prompt type: {type(maybe_prompt)}")

        assert isinstance(prompt, str)
        logger.warning("request_id: %s, prompt: %r", request_id, prompt)

        yield RequestOutput(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=[],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="MOCK",
                    token_ids=[0],
                    cumulative_logprob=None,
                    logprobs=None,
                    finish_reason="stop",
                )
            ],
            finished=True,
        )

    async def get_scheduler_stats(self) -> SchedulerStats:
        """Wait until the next scheduler stats are available and return them."""
        return SchedulerStats()

    async def clear_scheduler_stats(self) -> None:
        """Clear any pending scheduler stats without returning them."""
        return

    @property
    def is_running(self) -> bool:
        return True

    @property
    def is_stopped(self) -> bool:
        return False

    @property
    def errored(self) -> bool:
        return False

    @property
    def dead_error(self) -> BaseException:
        return Exception()

    @property
    def renderer(self) -> BaseRenderer:
        raise NotImplementedError("Not supported.")

    def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """Generate outputs for a request from a pooling model."""
        raise ValueError("Not supported.")

    async def abort(self, request_id: str | Iterable[str]) -> None:
        """Abort a request.

        Args:
            request_id: The unique id of the request.
        """
        return

    async def get_vllm_config(self) -> VllmConfig:
        """Get the vLLM configuration of the vLLM engine."""
        return self.vllm_config

    def get_input_preprocessor(self) -> InputPreprocessor:
        """Get the input processor of the vLLM engine."""
        return self.input_processor.input_preprocessor

    def get_tokenizer(self) -> TokenizerLike:
        """Get the tokenizer for the engine."""
        return self.input_processor.get_tokenizer()

    async def is_tracing_enabled(self) -> bool:
        return False

    async def do_log_stats(self) -> None:
        pass

    async def check_health(self) -> None:
        """Raise if unhealthy"""
        logger.debug("Called check_health()")

    async def start_profile(self) -> None:
        """Start profiling the engine"""
        pass

    async def stop_profile(self) -> None:
        """Start profiling the engine"""
        pass

    async def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the prefix cache"""
        return True

    async def reset_mm_cache(self) -> None:
        """Reset the multi-modal cache"""
        pass

    async def reset_encoder_cache(self) -> None:
        """Reset the encoder cache"""
        pass

    async def sleep(self, level: int = 1) -> None:
        """Sleep the engine"""
        pass

    async def wake_up(self, tags: list[str] | None = None) -> None:
        """Wake up the engine"""
        pass

    async def is_sleeping(self) -> bool:
        """Check whether the engine is sleeping"""
        return False

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        return True

    async def pause_generation(self, **kwargs: Any) -> None:
        """Pause new generation/encoding requests."""
        pass

    async def resume_generation(self) -> None:
        """Resume accepting generation/encoding requests."""
        pass

    async def is_paused(self) -> bool:
        """Return whether the engine is currently paused."""
        return False

    async def change_kv_role(self, new_role: str) -> None:
        """Change the role used in key-value caching."""
        pass
