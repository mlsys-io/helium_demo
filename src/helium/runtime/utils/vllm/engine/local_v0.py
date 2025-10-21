import asyncio
import time
from argparse import Namespace
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from typing import Any, NamedTuple, cast

import jinja2
from fastapi import Request
from prometheus_client import REGISTRY as METRIC_REGISTRY

from helium.runtime.utils.loop import AsyncEventLoop
from helium.runtime.utils.pool import AsyncPool
from helium.runtime.utils.vllm.engine.mock import MockLLMEngine
from helium.runtime.utils.vllm.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    DetokenizeRequest,
    DetokenizeResponse,
    ErrorResponse,
    RequestResponseMetadata,
    TokenizeRequest,
    TokenizeResponse,
)
from helium.runtime.utils.vllm.openai.serving_chat import OpenAIServingChat
from helium.runtime.utils.vllm.openai.serving_completion import OpenAIServingCompletion
from helium.runtime.utils.vllm.openai.serving_tokenize import OpenAIServingTokenization
from helium.runtime.utils.vllm.openai.serving_utils import (
    BenchmarkRequestTracker,
    add_request_output_delta,
)
from helium.runtime.utils.vllm.utils import get_metric_values
from helium.runtime.utils.vllm.vllm_logger import init_child_logger
from vllm.vllm.config import ModelConfig
from vllm.vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.vllm.engine.protocol import EngineClient
from vllm.vllm.entrypoints.chat_utils import load_chat_template
from vllm.vllm.entrypoints.openai.api_server import build_async_engine_client
from vllm.vllm.entrypoints.openai.serving_models import (
    BaseModelPath,
    OpenAIServingModels,
)
from vllm.vllm.inputs import PromptType
from vllm.vllm.lora.request import LoRARequest
from vllm.vllm.outputs import RequestOutput
from vllm.vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.vllm.transformers_utils.tokenizers import (
    MistralTokenizer,
    maybe_serialize_tool_calls,
    truncate_tool_call_ids,
)
from vllm.vllm.utils import merge_async_iterators
from vllm.vllm.v1.metrics.stats import SchedulerStats

logger = init_child_logger("local")


class SerializedLLMEngine(AsyncLLMEngine):
    """
    A wrapper around an AsyncLLMEngine that serializes the inputs to the engine.
    """

    class Input(NamedTuple):
        prompt: PromptType
        sampling_params: SamplingParams
        request_id: str
        lora_request: LoRARequest | None
        trace_headers: Mapping[str, str] | None
        prompt_adapter_request: PromptAdapterRequest | None
        priority: int

    def __init__(self, engine: EngineClient) -> None:
        self.engine = engine  # type: ignore[assignment]
        self._engine_loop: AsyncEventLoop[
            SerializedLLMEngine.Input,
            AsyncGenerator[RequestOutput, None],
            None,
        ] = AsyncEventLoop(self._engine_func, result_collector=AsyncPool())

    def __getattr__(self, attr: str) -> Any:
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.engine, attr)

    async def get_model_config(self) -> ModelConfig:
        return await self.engine.get_model_config()

    async def _start(self) -> None:
        await self._engine_loop.start()

    async def _stop(self) -> None:
        await self._engine_loop.stop()

    @classmethod
    @asynccontextmanager
    async def from_cli_args(
        cls, args: Namespace
    ) -> AsyncGenerator["SerializedLLMEngine", None]:
        async with build_async_engine_client(args) as engine:
            async with cls.from_engine(engine) as serialized_engine:
                yield serialized_engine

    @classmethod
    @asynccontextmanager
    async def from_engine(
        cls, engine: EngineClient
    ) -> AsyncGenerator["SerializedLLMEngine", None]:
        serialized_engine = cls(engine)
        await serialized_engine._start()
        try:
            yield serialized_engine
        finally:
            await serialized_engine._stop()

    async def generate(
        self,
        prompt: PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        prompt_adapter_request: PromptAdapterRequest | None = None,
        priority: int = 0,
    ) -> AsyncGenerator[RequestOutput, None]:
        input = SerializedLLMEngine.Input(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
            prompt_adapter_request=prompt_adapter_request,
            priority=priority,
        )
        generator = await self._engine_loop.process_event(input)

        async for output in generator:
            yield output

    async def _engine_func(
        self, inp: "SerializedLLMEngine.Input", _
    ) -> AsyncGenerator[RequestOutput, None]:
        return self.engine.generate(
            prompt=inp.prompt,
            sampling_params=inp.sampling_params,
            request_id=inp.request_id,
            lora_request=inp.lora_request,
            trace_headers=inp.trace_headers,
            prompt_adapter_request=inp.prompt_adapter_request,
            priority=inp.priority,
        )

    def get_stats(self) -> dict[str, Any]:
        engine_metrics = list(METRIC_REGISTRY.collect())
        stats = get_metric_values(engine_metrics)
        return stats

    async def reset_prefix_cache(self, device: Any | None = None) -> None:
        return await self.engine.reset_prefix_cache(device)


class OpenAIEngineClient:
    def __init__(
        self, engine: SerializedLLMEngine, model_config: ModelConfig, args: Namespace
    ) -> None:
        self.engine = engine

        if args.served_model_name is not None:
            served_model_names = args.served_model_name
        else:
            served_model_names = [args.model]

        base_model_paths = [
            BaseModelPath(name=name, model_path=args.model)
            for name in served_model_names
        ]
        serving_models = OpenAIServingModels(
            engine_client=engine,
            model_config=model_config,
            base_model_paths=base_model_paths,
            lora_modules=args.lora_modules,
            prompt_adapters=args.prompt_adapters,
        )

        chat_template = load_chat_template(args.chat_template)
        self.benchmarking: bool = args.benchmarking
        self.serving_completion = OpenAIServingCompletion(
            self.engine,
            model_config,
            serving_models,
            request_logger=None,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            benchmarking=self.benchmarking,
        )
        self.serving_chat = OpenAIServingChat(
            self.engine,
            model_config,
            serving_models,
            args.response_role,
            request_logger=None,
            chat_template=chat_template,
            chat_template_content_format=args.chat_template_content_format,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_auto_tools=args.enable_auto_tool_choice,
            tool_parser=args.tool_call_parser,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            benchmarking=self.benchmarking,
        )
        self.serving_tokenize = OpenAIServingTokenization(
            self.engine,
            model_config,
            serving_models,
            request_logger=None,
            chat_template=chat_template,
            chat_template_content_format=args.chat_template_content_format,
        )
        self.chat_template = chat_template

    @classmethod
    @asynccontextmanager
    async def build(
        cls, args: Namespace, mock: bool
    ) -> AsyncGenerator["OpenAIEngineClient", None]:
        if mock:
            mock_engine = MockLLMEngine.from_cli_args(args)
            engine_context = SerializedLLMEngine.from_engine(mock_engine)
        else:
            engine_context = SerializedLLMEngine.from_cli_args(args)
        async with engine_context as engine:
            engine_client = await cls.from_serialized_engine(engine, args)
            yield engine_client

    @classmethod
    async def from_serialized_engine(
        cls, engine: SerializedLLMEngine, args: Namespace
    ) -> "OpenAIEngineClient":
        model_config = await engine.get_model_config()
        return cls(engine, model_config, args)

    async def create_completion(
        self,
        request_id: str,
        request: CompletionRequest,
        raw_request: Request | None = None,
    ) -> CompletionResponse | ErrorResponse:
        """Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/completions/create
        for the API specification. This API mimics the OpenAI Completion API.

        NOTE: Currently we do not support the following feature:
            - suffix (the language models we currently support do not support
            suffix)
        """
        serving_completion = self.serving_completion
        error_check_ret = await serving_completion._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if serving_completion.engine_client.errored:
            raise serving_completion.engine_client.dead_error

        if request.stream:
            raise NotImplementedError("Streaming is not supported in this engine")

        if self.benchmarking:
            if request.stream:
                raise NotImplementedError(
                    "Streaming is not supported when benchmarking"
                )
            request.stream = True

        # Return error for unsupported features.
        if request.suffix is not None:
            return serving_completion.create_error_response(
                "suffix is not currently supported"
            )

        created_time = int(time.time())

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        try:
            (
                lora_request,
                prompt_adapter_request,
            ) = serving_completion._maybe_get_adapters(request)

            tokenizer = await self.engine.engine.get_tokenizer(lora_request)

            request_prompts, engine_prompts = (
                await serving_completion._preprocess_completion(
                    request,
                    tokenizer,
                    request.prompt,
                    truncate_prompt_tokens=request.truncate_prompt_tokens,
                    add_special_tokens=request.add_special_tokens,
                )
            )
        except ValueError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return serving_completion.create_error_response(str(e))
        except TypeError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return serving_completion.create_error_response(str(e))
        except RuntimeError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return serving_completion.create_error_response(str(e))
        except jinja2.TemplateError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return serving_completion.create_error_response(str(e))

        num_prompts = len(engine_prompts)
        benchmark_trackers = (
            [BenchmarkRequestTracker() for _ in range(num_prompts)]
            if self.benchmarking
            else None
        )

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                sampling_params: SamplingParams | BeamSearchParams
                default_max_tokens = serving_completion.max_model_len - len(
                    engine_prompt["prompt_token_ids"]
                )
                if request.use_beam_search:
                    sampling_params = request.to_beam_search_params(
                        default_max_tokens, serving_completion.default_sampling_params
                    )
                else:
                    sampling_params = request.to_sampling_params(
                        default_max_tokens,
                        serving_completion.model_config.logits_processor_pattern,
                        serving_completion.default_sampling_params,
                    )

                request_id_item = f"{request_id}-{i}"

                serving_completion._log_inputs(
                    request_id_item,
                    request_prompts[i],
                    params=sampling_params,
                    lora_request=lora_request,
                    prompt_adapter_request=prompt_adapter_request,
                )

                trace_headers = (
                    None
                    if raw_request is None
                    else await serving_completion._get_trace_headers(
                        raw_request.headers
                    )
                )

                if isinstance(sampling_params, BeamSearchParams):
                    generator = serving_completion.engine_client.beam_search(
                        prompt=engine_prompt,
                        request_id=request_id,
                        params=sampling_params,
                    )
                else:
                    generator = serving_completion.engine_client.generate(
                        engine_prompt,
                        sampling_params,
                        request_id_item,
                        lora_request=lora_request,
                        prompt_adapter_request=prompt_adapter_request,
                        trace_headers=trace_headers,
                        priority=request.priority,
                    )

                generators.append(generator)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return serving_completion.create_error_response(str(e))

        result_generator = merge_async_iterators(*generators)

        model_name = serving_completion._get_model_name(request.model, lora_request)
        num_prompts = len(engine_prompts)

        # Non-streaming response
        final_res_batch: list[RequestOutput | None] = [None] * num_prompts
        try:
            async for i, res in result_generator:
                prev_res = final_res_batch[i]
                if benchmark_trackers is not None:
                    benchmark_trackers[i].update(res)
                if prev_res is None:
                    final_res_batch[i] = res
                else:
                    final_res_batch[i] = add_request_output_delta(prev_res, res)

            for i, final_res in enumerate(final_res_batch):
                assert final_res is not None

                # The output should contain the input text
                # We did not pass it into vLLM engine to avoid being redundant
                # with the inputs token IDs
                if final_res.prompt is None:
                    final_res.prompt = request_prompts[i]["prompt"]

            final_res_batch_checked = cast(list[RequestOutput], final_res_batch)

            response = serving_completion.request_output_to_completion_response(
                final_res_batch_checked,
                benchmark_trackers,
                request,
                request_id,
                created_time,
                model_name,
                tokenizer,
                request_metadata,
            )
        except asyncio.CancelledError:
            return serving_completion.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return serving_completion.create_error_response(str(e))

        return response

    async def create_chat_completion(
        self,
        request_id: str,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> ChatCompletionResponse | ErrorResponse:
        """
        Chat Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI
        Chat Completion API.
        """
        serving_chat = self.serving_chat

        error_check_ret = await serving_chat._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if serving_chat.engine_client.errored:
            raise serving_chat.engine_client.dead_error

        if self.benchmarking:
            if request.stream:
                raise NotImplementedError(
                    "Streaming is not supported when benchmarking"
                )
            request.stream = True  # To enable benchmarking

        try:
            (
                lora_request,
                prompt_adapter_request,
            ) = serving_chat._maybe_get_adapters(request)

            model_name = serving_chat._get_model_name(request.model, lora_request)

            tokenizer = await self.engine.engine.get_tokenizer(lora_request)

            tool_parser = serving_chat.tool_parser

            # validation for OpenAI tools
            # tool_choice = "required" is not supported
            if request.tool_choice == "required":
                return serving_chat.create_error_response(
                    'tool_choice = "required" is not supported!'
                )

            if isinstance(tokenizer, MistralTokenizer):
                # because of issues with pydantic we need to potentially
                # re-serialize the tool_calls field of the request
                # for more info: see comment in `maybe_serialize_tool_calls`
                maybe_serialize_tool_calls(request)  # type: ignore
                truncate_tool_call_ids(request)  # type: ignore

            if (
                request.tool_choice == "auto"
                and not (serving_chat.enable_auto_tools and tool_parser is not None)
                and not isinstance(tokenizer, MistralTokenizer)
            ):
                # for hf tokenizers, "auto" tools requires
                # --enable-auto-tool-choice and --tool-call-parser
                return serving_chat.create_error_response(
                    '"auto" tool choice requires '
                    "--enable-auto-tool-choice and --tool-call-parser to be set"
                )

            tool_dicts = (
                None
                if request.tools is None
                else [tool.model_dump() for tool in request.tools]
            )

            (
                conversation,
                request_prompts,
                engine_prompts,
            ) = await serving_chat._preprocess_chat(
                request,
                tokenizer,
                request.messages,
                chat_template=request.chat_template or self.chat_template,
                chat_template_content_format=serving_chat.chat_template_content_format,
                add_generation_prompt=request.add_generation_prompt,
                continue_final_message=request.continue_final_message,
                tool_dicts=tool_dicts,
                documents=request.documents,
                chat_template_kwargs=request.chat_template_kwargs,
                tool_parser=tool_parser,
                truncate_prompt_tokens=request.truncate_prompt_tokens,
                add_special_tokens=request.add_special_tokens,
            )
        except ValueError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return serving_chat.create_error_response(str(e))
        except TypeError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return serving_chat.create_error_response(str(e))
        except RuntimeError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return serving_chat.create_error_response(str(e))
        except jinja2.TemplateError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return serving_chat.create_error_response(str(e))

        benchmark_tracker = BenchmarkRequestTracker() if self.benchmarking else None

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                sampling_params: SamplingParams | BeamSearchParams
                default_max_tokens = serving_chat.max_model_len - len(
                    engine_prompt["prompt_token_ids"]
                )
                if request.use_beam_search:
                    sampling_params = request.to_beam_search_params(
                        default_max_tokens, serving_chat.default_sampling_params
                    )
                else:
                    sampling_params = request.to_sampling_params(
                        default_max_tokens,
                        serving_chat.model_config.logits_processor_pattern,
                        serving_chat.default_sampling_params,
                    )

                serving_chat._log_inputs(
                    request_id,
                    request_prompts[i],
                    params=sampling_params,
                    lora_request=lora_request,
                    prompt_adapter_request=prompt_adapter_request,
                )

                trace_headers = (
                    None
                    if raw_request is None
                    else await serving_chat._get_trace_headers(raw_request.headers)
                )

                if isinstance(sampling_params, BeamSearchParams):
                    generator = serving_chat.engine_client.beam_search(
                        prompt=engine_prompt,
                        request_id=request_id,
                        params=sampling_params,
                    )
                else:
                    generator = serving_chat.engine_client.generate(
                        engine_prompt,
                        sampling_params,
                        request_id,
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                        prompt_adapter_request=prompt_adapter_request,
                        priority=request.priority,
                    )

                generators.append(generator)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return serving_chat.create_error_response(str(e))

        assert len(generators) == 1
        (result_generator,) = generators

        try:
            return await serving_chat.chat_completion_full_generator(
                request,
                result_generator,
                benchmark_tracker,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
            )
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return serving_chat.create_error_response(str(e))

    async def create_tokenize(
        self, request_id: str, request: TokenizeRequest
    ) -> TokenizeResponse | ErrorResponse:
        return await self.serving_tokenize.create_tokenize(request_id, request)

    async def create_detokenize(
        self, request_id: str, request: DetokenizeRequest
    ) -> DetokenizeResponse | ErrorResponse:
        return await self.serving_tokenize.create_detokenize(request_id, request)

    async def reset_prefix_cache(self, device: Any | None = None) -> None:
        return await self.engine.reset_prefix_cache(device)

    async def change_kv_role(self, new_role: str) -> None:
        raise NotImplementedError()

    def get_stats(self) -> dict[str, Any]:
        return self.engine.get_stats()

    async def get_scheduler_stats(self) -> SchedulerStats:
        raise NotImplementedError()

    async def clear_scheduler_stats(self) -> None:
        raise NotImplementedError()
