from argparse import Namespace
from collections.abc import AsyncGenerator, Mapping

from helium.runtime.utils.vllm.vllm_logger import init_child_logger
from vllm.vllm.config import DecodingConfig, ModelConfig, VllmConfig
from vllm.vllm.core.scheduler import SchedulerOutputs
from vllm.vllm.engine.arg_utils import AsyncEngineArgs
from vllm.vllm.engine.protocol import EngineClient
from vllm.vllm.inputs.data import PromptType
from vllm.vllm.inputs.preprocess import InputPreprocessor
from vllm.vllm.lora.request import LoRARequest
from vllm.vllm.model_executor.layers.sampler import SamplerOutput
from vllm.vllm.outputs import CompletionOutput, PoolingRequestOutput, RequestOutput
from vllm.vllm.pooling_params import PoolingParams
from vllm.vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.vllm.sampling_params import SamplingParams
from vllm.vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.vllm.usage.usage_lib import UsageContext
from vllm.vllm.utils import Device
from vllm.vllm.v1.engine.processor import Processor
from vllm.vllm.v1.metrics.stats import SchedulerStats

logger = init_child_logger("mock")


class MockLLMEngine(EngineClient):
    def __init__(self, vllm_config: VllmConfig) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config

        self.tokenizer = init_tokenizer_from_configs(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            lora_config=vllm_config.lora_config,  # type: ignore[arg-type]
        )

        self.processor = Processor(vllm_config=vllm_config, tokenizer=self.tokenizer)

        logger.warning("Initialized MockLLMEngine")

    @classmethod
    def from_cli_args(cls, args: Namespace) -> "MockLLMEngine":
        engine_args = AsyncEngineArgs.from_cli_args(args)
        vllm_config = engine_args.create_engine_config(
            usage_context=UsageContext.OPENAI_API_SERVER
        )
        return cls(vllm_config)

    async def generate(  # type: ignore[override]
        self,
        prompt: PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        prompt_adapter_request: PromptAdapterRequest | None = None,
        priority: int = 0,
    ) -> AsyncGenerator[RequestOutput, None]:
        if isinstance(prompt, dict):
            maybe_prompt = prompt.get("prompt")
            if maybe_prompt is None:
                token_ids = prompt.get("prompt_token_ids")
                if token_ids is None:
                    raise ValueError(
                        "Unsupported prompt format: missing 'prompt' or 'prompt_token_ids'"
                    )
                tokenizer = await self.get_tokenizer()
                prompt = tokenizer.decode(token_ids, skip_special_tokens=False)  # type: ignore
                assert isinstance(prompt, str)
            elif isinstance(maybe_prompt, str):
                prompt = maybe_prompt
            else:
                raise ValueError(f"Unsupported prompt type: {type(maybe_prompt)}")

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

    def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """Generate outputs for a request from a pooling model."""
        raise ValueError("Not supported.")

    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Args:
            request_id: The unique id of the request.
        """
        return

    async def get_vllm_config(self) -> VllmConfig:
        """Get the vLLM configuration of the vLLM engine."""
        return self.vllm_config

    async def get_model_config(self) -> ModelConfig:
        """Get the model configuration of the vLLM engine."""
        return self.model_config

    async def get_decoding_config(self) -> DecodingConfig:
        """Get the decoding configuration of the vLLM engine."""
        raise ValueError("Not supported.")

    async def get_input_preprocessor(self) -> InputPreprocessor:
        """Get the input processor of the vLLM engine."""
        return self.processor.input_preprocessor

    async def get_tokenizer(
        self,
        lora_request: LoRARequest | None = None,
    ) -> AnyTokenizer:
        """Get the appropriate tokenizer for the request"""
        return self.tokenizer.get_lora_tokenizer(lora_request)

    async def is_tracing_enabled(self) -> bool:
        return False

    async def do_log_stats(
        self,
        scheduler_outputs: SchedulerOutputs | None = None,
        model_output: list[SamplerOutput] | None = None,
    ) -> None:
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

    async def reset_prefix_cache(self, device: Device | None = None) -> None:
        """Reset the prefix cache"""
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

    async def add_lora(self, lora_request: LoRARequest) -> None:
        """Load a new LoRA adapter into the engine for future requests."""
        pass

    async def change_kv_role(self, new_role: str) -> None:
        """Change the role used in key-value caching."""
        pass
