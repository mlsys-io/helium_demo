import asyncio
from collections.abc import AsyncGenerator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np

from helium.runtime.utils.vllm.vllm_logger import init_child_logger
from vllm.vllm.config import ModelConfig, VllmConfig
from vllm.vllm.engine.protocol import EngineClient
from vllm.vllm.inputs.data import PromptType, StreamingInput
from vllm.vllm.inputs.preprocess import InputPreprocessor
from vllm.vllm.lora.request import LoRARequest
from vllm.vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.vllm.pooling_params import PoolingParams
from vllm.vllm.renderers.inputs import DictPrompt, TokPrompt
from vllm.vllm.renderers.protocol import BaseRenderer
from vllm.vllm.sampling_params import SamplingParams
from vllm.vllm.tasks import SupportedTask
from vllm.vllm.tokenizers.protocol import TokenizerLike
from vllm.vllm.v1.engine import EngineCoreRequest

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

        # Expose required EngineClient attributes from the primary worker
        self.vllm_config = engine_clients[0].vllm_config
        self.model_config = engine_clients[0].model_config
        self.input_processor = engine_clients[0].input_processor
        self.io_processor = getattr(engine_clients[0], "io_processor", None)

    @property
    def num_workers(self) -> int:
        return len(self._workers)

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
        i = self._get_next_worker_index()
        logger.debug("Dispatching request %s to worker %d", request_id, i)

        self._request_counts[i] += 1
        self._request_trackers[request_id] = i

        try:
            generator = self._workers[i].generate(
                prompt,
                sampling_params,
                request_id,
                lora_request=lora_request,
                trace_headers=trace_headers,
                priority=priority,
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

    @property
    def renderer(self) -> BaseRenderer:
        return self._workers[0].renderer

    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return await self._workers[0].get_supported_tasks()

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
        if isinstance(request_id, str):
            i = self._request_trackers.get(request_id)
            if i is not None:
                await self._workers[i].abort(request_id)
        else:
            for rid in request_id:
                i = self._request_trackers.get(rid)
                if i is not None:
                    await self._workers[i].abort(rid)

    async def get_vllm_config(self) -> VllmConfig:
        """Get the vLLM configuration of the vLLM engine."""
        return self._workers[0].vllm_config

    async def get_model_config(self) -> ModelConfig:
        """Get the model configuration of the vLLM engine."""
        return self._workers[0].model_config

    def get_input_preprocessor(self) -> InputPreprocessor:
        """Get the input processor of the vLLM engine."""
        return self._workers[0].input_processor.input_preprocessor

    def get_tokenizer(self) -> TokenizerLike:
        """Get the appropriate tokenizer"""
        return self._workers[0].renderer.get_tokenizer()

    async def is_tracing_enabled(self) -> bool:
        return all(
            await asyncio.gather(
                *[worker.is_tracing_enabled() for worker in self._workers]
            )
        )

    async def do_log_stats(self) -> None:
        await asyncio.gather(*[worker.do_log_stats() for worker in self._workers])

    async def check_health(self) -> None:
        """Raise if unhealthy"""
        await asyncio.gather(*[worker.check_health() for worker in self._workers])

    async def start_profile(self) -> None:
        """Start profiling the engine"""
        await asyncio.gather(*[worker.start_profile() for worker in self._workers])

    async def stop_profile(self) -> None:
        """Start profiling the engine"""
        await asyncio.gather(*[worker.stop_profile() for worker in self._workers])

    async def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the prefix cache"""
        results = await asyncio.gather(
            *[
                worker.reset_prefix_cache(reset_running_requests, reset_connector)
                for worker in self._workers
            ]
        )
        return all(results)

    async def reset_mm_cache(self) -> None:
        """Reset the multi-modal cache"""
        await asyncio.gather(*[worker.reset_mm_cache() for worker in self._workers])

    async def reset_encoder_cache(self) -> None:
        """Reset the encoder cache"""
        await asyncio.gather(
            *[worker.reset_encoder_cache() for worker in self._workers]
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

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        results = await asyncio.gather(
            *[worker.add_lora(lora_request) for worker in self._workers]
        )
        return all(results)

    async def pause_generation(self, **kwargs) -> None:
        """Pause new generation/encoding requests."""
        await asyncio.gather(
            *[worker.pause_generation(**kwargs) for worker in self._workers]
        )

    async def resume_generation(self) -> None:
        """Resume accepting generation/encoding requests."""
        await asyncio.gather(*[worker.resume_generation() for worker in self._workers])

    async def is_paused(self) -> bool:
        """Return whether the engine is currently paused."""
        return any(
            await asyncio.gather(*[worker.is_paused() for worker in self._workers])
        )
