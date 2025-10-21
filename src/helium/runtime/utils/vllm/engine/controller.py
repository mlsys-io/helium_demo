import asyncio
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from helium.runtime.utils.vllm.vllm_logger import init_child_logger
from vllm.vllm.config import DecodingConfig, ModelConfig, VllmConfig
from vllm.vllm.core.scheduler import SchedulerOutputs
from vllm.vllm.engine.protocol import EngineClient
from vllm.vllm.inputs.data import PromptType
from vllm.vllm.inputs.preprocess import InputPreprocessor
from vllm.vllm.lora.request import LoRARequest
from vllm.vllm.model_executor.layers.sampler import SamplerOutput
from vllm.vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.vllm.pooling_params import PoolingParams
from vllm.vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.vllm.sampling_params import SamplingParams
from vllm.vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.vllm.utils import Device

logger = init_child_logger("controller")


class DispatchMethod(Enum):
    LOTTERY = auto()
    SHORTEST_QUEUE = auto()

    @classmethod
    def from_str(cls, name: str) -> "DispatchMethod":
        if name == "lottery":
            return cls.LOTTERY
        elif name == "shortest_queue":
            return cls.SHORTEST_QUEUE
        else:
            raise ValueError("Invalid dispatch method")


@dataclass
class EngineClientInfo:
    cache_capacity: int | None = None


class EngineClientController(EngineClient):
    def __init__(
        self,
        engine_clients: Sequence[EngineClient],
        engine_infos: Sequence[EngineClientInfo],
        dispatch_method: DispatchMethod = DispatchMethod.SHORTEST_QUEUE,
    ):
        if len(engine_clients) == 0:
            raise ValueError("At least one engine client is required.")
        if len(engine_clients) != len(engine_infos):
            raise ValueError("The numbers of engine clients and infos mismatch.")

        self.dispatch_method = dispatch_method
        self._workers = engine_clients
        self._worker_infos = engine_infos
        self._request_counts: list[int] = [0] * len(engine_clients)
        """List of request counts for each worker."""
        self._request_trackers: dict[str, int] = {}
        """Mapping from request_id to worker index."""

    @property
    def num_workers(self) -> int:
        return len(self._workers)

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
        i = self._get_next_worker_index()
        logger.debug("Dispatching request %s to worker %d", request_id, i)

        self._request_counts[i] += 1
        self._request_trackers[request_id] = i

        try:
            generator = self._workers[i].generate(
                prompt,
                sampling_params,
                request_id,
                lora_request,
                trace_headers,
                prompt_adapter_request,
                priority,
            )
            async for output in generator:
                yield output
        finally:
            self._request_counts[i] -= 1
            del self._request_trackers[request_id]

    def _get_next_worker_index(self) -> int:
        """
        Adapted from FastChat's controller
        (https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/controller.py)
        """
        match self.dispatch_method:
            case DispatchMethod.LOTTERY:
                num_workers = self.num_workers
                cache_capacities = [info.cache_capacity for info in self._worker_infos]
                if any(cache_capacities):
                    weights = np.ones(num_workers, dtype=np.float32) / num_workers
                else:
                    weights = np.array(cache_capacities, dtype=np.float32)
                    weights /= np.sum(weights)
                i = int(np.random.choice(np.arange(num_workers), p=weights))
            case DispatchMethod.SHORTEST_QUEUE:
                weights = np.array(self._request_counts, dtype=np.float32)
                cache_capacities = [info.cache_capacity for info in self._worker_infos]
                if all(cache_capacities):
                    weights /= np.array(cache_capacities, dtype=np.float32)
                i = int(np.argmin(weights))
        return i

    @property
    def is_running(self) -> bool:
        return all(worker.is_running for worker in self._workers)

    @property
    def is_stopped(self) -> bool:
        return any(worker.is_stopped for worker in self._workers)

    @property
    def errored(self) -> bool:
        return any(worker.errored for worker in self._workers)

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
        i = self._request_trackers.get(request_id)
        if i is not None:
            await self._workers[i].abort(request_id)

    async def get_vllm_config(self) -> VllmConfig:
        """Get the vLLM configuration of the vLLM engine."""
        return await self._workers[0].get_vllm_config()

    async def get_model_config(self) -> ModelConfig:
        """Get the model configuration of the vLLM engine."""
        return await self._workers[0].get_model_config()

    async def get_decoding_config(self) -> DecodingConfig:
        """Get the decoding configuration of the vLLM engine."""
        raise ValueError("Not supported.")

    async def get_input_preprocessor(self) -> InputPreprocessor:
        """Get the input processor of the vLLM engine."""
        return await self._workers[0].get_input_preprocessor()

    async def get_tokenizer(
        self,
        lora_request: LoRARequest | None = None,
    ) -> AnyTokenizer:
        """Get the appropriate tokenizer for the request"""
        return await self._workers[0].get_tokenizer(lora_request=lora_request)

    async def is_tracing_enabled(self) -> bool:
        return all(
            await asyncio.gather(
                *[worker.is_tracing_enabled() for worker in self._workers]
            )
        )

    async def do_log_stats(
        self,
        scheduler_outputs: SchedulerOutputs | None = None,
        model_output: list[SamplerOutput] | None = None,
    ) -> None:
        await asyncio.gather(
            *[
                worker.do_log_stats(scheduler_outputs, model_output)
                for worker in self._workers
            ]
        )

    async def check_health(self) -> None:
        """Raise if unhealthy"""
        await asyncio.gather(*[worker.check_health() for worker in self._workers])

    async def start_profile(self) -> None:
        """Start profiling the engine"""
        await asyncio.gather(*[worker.start_profile() for worker in self._workers])

    async def stop_profile(self) -> None:
        """Start profiling the engine"""
        await asyncio.gather(*[worker.stop_profile() for worker in self._workers])

    async def reset_prefix_cache(self, device: Device | None = None) -> None:
        """Reset the prefix cache"""
        await asyncio.gather(
            *[worker.reset_prefix_cache(device) for worker in self._workers]
        )

    async def sleep(self, level: int = 1) -> None:
        """Sleep the engine"""
        await asyncio.gather(*[worker.sleep(level) for worker in self._workers])

    async def wake_up(self, tags: list[str] | None = None) -> None:
        """Wake up the engine"""
        await asyncio.gather(*[worker.wake_up(tags) for worker in self._workers])

    async def is_sleeping(self) -> bool:
        """Check whether the engine is sleeping"""
        return any(
            await asyncio.gather(*[worker.is_sleeping() for worker in self._workers])
        )

    async def add_lora(self, lora_request: LoRARequest) -> None:
        """Load a new LoRA adapter into the engine for future requests."""
        await asyncio.gather(
            *[worker.add_lora(lora_request) for worker in self._workers]
        )
