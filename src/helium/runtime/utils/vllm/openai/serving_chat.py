# pyright: reportPossiblyUnboundVariable=false

import asyncio
import json
import re
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from collections.abc import Sequence as GenericSequence
from typing import Any, Final

import jinja2
import partial_json_parser
from fastapi import Request
from partial_json_parser import MalformedJSON
from pydantic import TypeAdapter

from helium import envs
from helium.runtime.utils.vllm.openai.protocol import (
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ErrorResponse,
    FunctionCall,
    FunctionDefinition,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    ToolCall,
    UsageInfo,
)
from helium.runtime.utils.vllm.openai.serving_utils import (
    BenchmarkRequestTracker,
    add_request_output_delta,
)
from helium.runtime.utils.vllm.vllm_logger import init_child_logger
from vllm.vllm.config import ModelConfig
from vllm.vllm.engine.protocol import EngineClient
from vllm.vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption,
    ConversationMessage,
)
from vllm.vllm.entrypoints.logger import RequestLogger
from vllm.vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.vllm.entrypoints.openai.serving_engine import (
    OpenAIServing,
    clamp_prompt_logprobs,
)
from vllm.vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.vllm.entrypoints.openai.tool_parsers import ToolParser, ToolParserManager
from vllm.vllm.entrypoints.openai.tool_parsers.mistral_tool_parser import (
    MistralToolCall,
)
from vllm.vllm.outputs import CompletionOutput, RequestOutput
from vllm.vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.vllm.sequence import Logprob
from vllm.vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.vllm.transformers_utils.tokenizers import (
    MistralTokenizer,
    maybe_serialize_tool_calls,
    truncate_tool_call_ids,
    validate_request_params,
)

logger = init_child_logger("serving_chat")


class OpenAIServingChat(OpenAIServing):

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        response_role: str,
        *,
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        return_tokens_as_token_ids: bool = False,
        enable_reasoning: bool = False,
        reasoning_parser: str | None = None,
        enable_auto_tools: bool = False,
        tool_parser: str | None = None,
        enable_prompt_tokens_details: bool = False,
        benchmarking: bool = False,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            model_config=model_config,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
        )

        self.benchmarking = benchmarking
        self.response_role = response_role
        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format

        # set up tool use
        self.enable_auto_tools: bool = enable_auto_tools
        if self.enable_auto_tools:
            logger.info(
                '"auto" tool choice has been enabled please note that while'
                " the parallel_tool_calls client option is preset for "
                "compatibility reasons, it will be ignored."
            )

        self.enable_reasoning: bool = enable_reasoning
        self.reasoning_parser: Callable[[AnyTokenizer], ReasoningParser] | None = None
        if self.enable_reasoning:
            try:
                self.reasoning_parser = ReasoningParserManager.get_reasoning_parser(
                    reasoning_parser
                )
            except Exception as e:
                raise TypeError(
                    "Error: --enable-reasoning requires "
                    f"reasoning_parser:'{reasoning_parser}' "
                    "which has not been registered"
                ) from e
        self.tool_parser: Callable[[AnyTokenizer], ToolParser] | None = None
        if self.enable_auto_tools:
            try:
                if tool_parser == "pythonic" and model_config.model.startswith(
                    "meta-llama/Llama-3.2"
                ):
                    logger.warning(
                        "Llama3.2 models may struggle to emit valid pythonic"
                        " tool calls"
                    )
                self.tool_parser = ToolParserManager.get_tool_parser(tool_parser)
            except Exception as e:
                raise TypeError(
                    "Error: --enable-auto-tool-choice requires "
                    f"tool_parser:'{tool_parser}' which has not "
                    "been registered"
                ) from e

        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.default_sampling_params = self.model_config.get_diff_sampling_param()
        if self.default_sampling_params:
            source = self.model_config.generation_config
            source = "model" if source == "auto" else source
            logger.info(
                "Using default chat sampling params from %s: %s",
                source,
                self.default_sampling_params,
            )

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
        """
        Chat Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI
        Chat Completion API.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        stream = request.stream
        if self.benchmarking:
            if request.stream:
                raise NotImplementedError(
                    "Streaming is not supported when benchmarking"
                )
            request.stream = True  # To enable benchmarking

        if not envs.HELIUM_VLLM_ENABLE_THINKING:
            # Disable thinking
            if request.chat_template_kwargs is None:
                request.chat_template_kwargs = {"enable_thinking": False}
            else:
                request.chat_template_kwargs["enable_thinking"] = False

        try:
            (
                lora_request,
                prompt_adapter_request,
            ) = self._maybe_get_adapters(request)

            model_name = self._get_model_name(request.model, lora_request)

            tokenizer = await self.engine_client.get_tokenizer(lora_request)

            tool_parser = self.tool_parser

            if isinstance(tokenizer, MistralTokenizer):
                # because of issues with pydantic we need to potentially
                # re-serialize the tool_calls field of the request
                # for more info: see comment in `maybe_serialize_tool_calls`
                maybe_serialize_tool_calls(request)  # type: ignore
                truncate_tool_call_ids(request)  # type: ignore
                validate_request_params(request)  # type: ignore

            if (
                request.tool_choice == "auto"
                and not (self.enable_auto_tools and tool_parser is not None)
                and not isinstance(tokenizer, MistralTokenizer)
            ):
                # for hf tokenizers, "auto" tools requires
                # --enable-auto-tool-choice and --tool-call-parser
                return self.create_error_response(
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
            ) = await self._preprocess_chat(
                request,
                tokenizer,
                request.messages,
                chat_template=request.chat_template or self.chat_template,
                chat_template_content_format=self.chat_template_content_format,
                add_generation_prompt=request.add_generation_prompt,
                continue_final_message=request.continue_final_message,
                tool_dicts=tool_dicts,
                documents=request.documents,
                chat_template_kwargs=request.chat_template_kwargs,
                tool_parser=tool_parser,
                truncate_prompt_tokens=request.truncate_prompt_tokens,
                add_special_tokens=request.add_special_tokens,
            )
        except (ValueError, TypeError, RuntimeError, jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        benchmark_tracker = BenchmarkRequestTracker() if self.benchmarking else None

        request_id = (
            "chatcmpl-" f"{self._base_request_id(raw_request, request.request_id)}"
        )

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                sampling_params: SamplingParams | BeamSearchParams
                default_max_tokens = self.max_model_len - len(
                    engine_prompt["prompt_token_ids"]
                )
                if request.use_beam_search:
                    sampling_params = request.to_beam_search_params(
                        default_max_tokens, self.default_sampling_params
                    )
                else:
                    sampling_params = request.to_sampling_params(
                        default_max_tokens,
                        self.model_config.logits_processor_pattern,
                        self.default_sampling_params,
                    )

                self._log_inputs(
                    request_id,
                    request_prompts[i],
                    params=sampling_params,
                    lora_request=lora_request,
                    prompt_adapter_request=prompt_adapter_request,
                )

                trace_headers = (
                    None
                    if raw_request is None
                    else await self._get_trace_headers(raw_request.headers)
                )

                if isinstance(sampling_params, BeamSearchParams):
                    generator = self.engine_client.beam_search(
                        prompt=engine_prompt,
                        request_id=request_id,
                        params=sampling_params,
                    )
                else:
                    generator = self.engine_client.generate(
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
            return self.create_error_response(str(e))

        assert len(generators) == 1
        (result_generator,) = generators

        # Streaming response
        if not self.benchmarking and stream:
            return self.chat_completion_stream_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
            )

        try:
            return await self.chat_completion_full_generator(
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
            return self.create_error_response(str(e))

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        return request.messages[-1]["role"]

    @staticmethod
    def _bracket_level(s: str, opening="{", closing="}") -> int:
        """
        Calculate the current level of nested brackets in a given string.
        """
        level = 0
        for char in s:
            if char == opening:
                level += 1
            elif char == closing:
                level -= 1
        return level

    @staticmethod
    def _filter_delta_text(delta_text: str, previous_text: str) -> tuple[str, bool]:
        # remove last '},' of the tool definition stemming from the
        # "name"/"parameters" outer object or closing ']' of the tool list
        # count occurrences of opening and closing curly braces and
        # once level 0 is reached stop outputting text
        # if 0 is reached while parsing the delta_text we know the current
        # tool will finish in this current iteration
        bracket_level = OpenAIServingChat._bracket_level(previous_text)
        updated_delta, passed_zero = "", False
        for c in delta_text:
            if c == "{":
                bracket_level += 1
                passed_zero = bracket_level == 0
            elif c == "}":
                bracket_level -= 1
                passed_zero = bracket_level == 0

            if bracket_level != 0:
                updated_delta += c
            else:
                # if a comma is reached at level 0 we can stop
                if c == ",":
                    break
        return updated_delta, passed_zero

    def extract_tool_call_required_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        function_name_returned: bool,
    ) -> tuple[DeltaMessage | None, bool]:
        try:
            obj = partial_json_parser.loads(current_text)
        except MalformedJSON:
            logger.debug("not enough tokens to parse into JSON yet")
            obj = None

        # check if the current text is a valid array
        # containing a partial tool calling object
        # if not repeat
        if obj is None or not isinstance(obj, list) or not len(obj) > 0:
            function_name_returned = False
            delta_message = None
        else:
            _, finishes_previous_tool = OpenAIServingChat._filter_delta_text(
                delta_text, previous_text
            )
            # take the last tool call from the generated list
            current_tool_call: Any = obj[-1]

            # once parameters have been generated the name is complete as well
            if not finishes_previous_tool and (
                "name" not in current_tool_call or "parameters" not in current_tool_call
            ):
                function_name_returned = False
                delta_message = None
            else:
                if not function_name_returned:
                    # get partly generated arguments from the latest tool call
                    param_match = re.search(r'.*"parameters":\s*(.*)', current_text)
                    arguments = param_match.group(1) if param_match else ""
                    arguments, _ = OpenAIServingChat._filter_delta_text(
                        arguments, previous_text
                    )

                    # if this iteration finishes a previous tool call but a
                    # new incomplete tool is already generated, take the
                    # previous from the list
                    if finishes_previous_tool and "parameters" not in current_tool_call:
                        current_tool_call = obj[-2]

                    function_name_returned = True
                    delta_message = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                function=DeltaFunctionCall(
                                    name=current_tool_call["name"], arguments=arguments
                                ),
                                index=len(obj) - 1,
                                type="function",
                            )
                        ]
                    )

                else:
                    delta_text, _ = OpenAIServingChat._filter_delta_text(
                        delta_text, previous_text
                    )

                    if delta_text != "":
                        delta_message = DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    function=DeltaFunctionCall(
                                        # OpenAI API returns None
                                        # instead of name every time
                                        name=None,
                                        arguments=delta_text,
                                    ),
                                    index=len(obj) - 1,
                                    type="function",
                                )
                            ]
                        )
                    else:
                        delta_message = None

        return delta_message, function_name_returned

    async def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
    ) -> AsyncGenerator[str, None]:
        created_time = int(time.time())
        chunk_object_type: Final = "chat.completion.chunk"
        first_iteration = True

        # Send response for each token for each request.n (index)
        num_choices = 1 if request.n is None else request.n
        previous_num_tokens = [0] * num_choices
        finish_reason_sent = [False] * num_choices
        num_prompt_tokens = 0
        num_cached_tokens = None

        if isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam):
            tool_choice_function_name = request.tool_choice.function.name
        else:
            tool_choice_function_name = None

        # Determine whether tools are in use with "auto" tool choice
        tool_choice_auto = (
            not tool_choice_function_name
            and self._should_stream_with_auto_tool_parsing(request)
        )

        should_stream_with_reasoning_parsing = (
            self._should_stream_with_reasoning_parsing(request)
        )

        all_previous_token_ids: list[list[int]] | None
        function_name_returned: list[bool] | None = None

        # Only one of these will be used, thus previous_texts and
        # all_previous_token_ids will not be used twice in the same iteration.
        if tool_choice_auto or should_stream_with_reasoning_parsing:
            # These are only required in "auto" tool choice case
            previous_texts = [""] * num_choices
            all_previous_token_ids = [[]] * num_choices
            # For reasoning parser and tool call all enabled
            added_content_delta_arr = [False] * num_choices
            reasoning_end_arr = [False] * num_choices
        elif request.tool_choice == "required":
            previous_texts = [""] * num_choices
            function_name_returned = [False] * num_choices
            all_previous_token_ids = None
        else:
            previous_texts, all_previous_token_ids = None, None

        try:
            # There is no need to check if the reasoning_parser is None
            # because the should_stream_with_reasoning_parsing check
            # already ensures that the reasoning_parser is not None.
            # but the pre-commit hook requires it.
            if (
                should_stream_with_reasoning_parsing
                and self.reasoning_parser is not None
            ):
                reasoning_parser = self.reasoning_parser(tokenizer)
            else:
                reasoning_parser = None
        except RuntimeError as e:
            logger.exception("Error in reasoning parser creation.")
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Prepare the tool parser if it's needed
        try:
            if tool_choice_auto and self.tool_parser:
                tool_parsers: list[ToolParser | None] = [
                    self.tool_parser(tokenizer)
                ] * num_choices
            else:
                tool_parsers = [None] * num_choices
        except Exception as e:
            logger.exception("Error in tool parser creation.")
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
            return

        stream_options = request.stream_options
        if stream_options:
            include_usage = stream_options.include_usage
            include_continuous_usage = (
                include_usage and stream_options.continuous_usage_stats
            )
        else:
            include_usage, include_continuous_usage = False, False

        try:
            async for res in result_generator:
                if res.prompt_token_ids is not None:
                    num_prompt_tokens = len(res.prompt_token_ids)
                    if res.encoder_prompt_token_ids is not None:
                        num_prompt_tokens += len(res.encoder_prompt_token_ids)

                # We need to do it here, because if there are exceptions in
                # the result_generator, it needs to be sent as the FIRST
                # response (by the try...catch).
                if first_iteration:
                    num_cached_tokens = res.num_cached_tokens
                    # Send first response for each request.n (index) with
                    # the role
                    role = self.get_chat_request_role(request)

                    # NOTE num_choices defaults to 1 so this usually executes
                    # once per request
                    for i in range(num_choices):
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(
                                role=role,
                                content="",
                            ),
                            logprobs=None,
                            finish_reason=None,
                        )
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name,
                        )

                        # if continuous usage stats are requested, add it
                        if include_continuous_usage:
                            chunk.usage = UsageInfo(
                                prompt_tokens=num_prompt_tokens,
                                completion_tokens=0,
                                total_tokens=num_prompt_tokens,
                            )

                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"

                    # Send response to echo the input portion of the
                    # last message
                    if request.echo:
                        last_msg_content: str | list[dict[str, str]] = ""
                        if (
                            conversation
                            and "content" in conversation[-1]
                            and conversation[-1].get("role") == role
                        ):
                            last_msg_content = conversation[-1]["content"] or ""

                        if last_msg_content:
                            for i in range(num_choices):
                                choice_data = ChatCompletionResponseStreamChoice(
                                    index=i,
                                    delta=DeltaMessage(content=last_msg_content),  # type: ignore
                                    logprobs=None,
                                    finish_reason=None,
                                )
                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    object=chunk_object_type,
                                    created=created_time,
                                    choices=[choice_data],
                                    model=model_name,
                                )
                                if include_continuous_usage:
                                    chunk.usage = UsageInfo(
                                        prompt_tokens=num_prompt_tokens,
                                        completion_tokens=0,
                                        total_tokens=num_prompt_tokens,
                                    )

                                data = chunk.model_dump_json(exclude_unset=True)
                                yield f"data: {data}\n\n"
                    first_iteration = False

                for output in res.outputs:
                    i = output.index
                    tool_parser = tool_parsers[i]

                    if finish_reason_sent[i]:
                        continue

                    if request.logprobs and request.top_logprobs is not None:
                        assert output.logprobs is not None, "Did not output logprobs"
                        logprobs = self._create_chat_logprobs(
                            token_ids=output.token_ids,
                            top_logprobs=output.logprobs,
                            tokenizer=tokenizer,
                            num_output_top_logprobs=request.top_logprobs,
                            return_as_token_id=request.return_tokens_as_token_ids,
                        )
                    else:
                        logprobs = None

                    delta_text = output.text

                    if (
                        not delta_text
                        and not output.token_ids
                        and not previous_num_tokens[i]
                    ):
                        # Chunked prefill case, don't return empty chunks
                        continue

                    delta_message: DeltaMessage | None

                    # just update previous_texts and previous_token_ids
                    if tool_choice_auto or should_stream_with_reasoning_parsing:
                        assert previous_texts is not None
                        assert all_previous_token_ids is not None
                        previous_text = previous_texts[i]
                        previous_token_ids = all_previous_token_ids[i]
                        current_text = previous_text + delta_text
                        current_token_ids = previous_token_ids + list(output.token_ids)

                    # handle streaming deltas for tools with named tool_choice
                    if tool_choice_function_name:
                        assert reasoning_parser is not None
                        if (
                            self.enable_reasoning
                            and not reasoning_parser.is_reasoning_end(
                                previous_token_ids
                            )
                        ):
                            delta_message = (
                                reasoning_parser.extract_reasoning_content_streaming(
                                    previous_text,
                                    current_text,
                                    delta_text,
                                    previous_token_ids,
                                    current_token_ids,
                                    output.token_ids,
                                )
                            )
                            # When encountering think end id in delta_token_ids,
                            # process the `content`. Only keep 'content',
                            # remove 'reasoning_content'
                            if reasoning_parser.is_reasoning_end(
                                list(output.token_ids)
                            ):
                                if delta_message and delta_message.content:
                                    # This need to be added to next `delta_text`
                                    current_text = delta_message.content
                                    delta_message.content = None
                                else:
                                    current_text = ""
                        else:
                            # Just to add remaining `content`
                            if self.enable_reasoning:
                                delta_text = previous_text + delta_text  # type: ignore
                                current_text = ""

                            delta_message = DeltaMessage(
                                tool_calls=[
                                    DeltaToolCall(
                                        function=DeltaFunctionCall(
                                            name=tool_choice_function_name,
                                            arguments=delta_text,
                                        ),
                                        index=i,
                                    )
                                ]
                            )

                    elif request.tool_choice == "required":
                        assert previous_texts is not None
                        assert function_name_returned is not None
                        previous_text = previous_texts[i]
                        current_text = previous_text + delta_text
                        fn_name_returned = function_name_returned[i]

                        delta_message, function_name_returned[i] = (
                            self.extract_tool_call_required_streaming(
                                previous_text=previous_text,
                                current_text=current_text,
                                delta_text=delta_text,
                                function_name_returned=fn_name_returned,
                            )
                        )

                        # update the previous values for the next iteration
                        previous_texts[i] = current_text

                    # handle streaming deltas for tools with "auto" tool choice
                    # and reasoning parser
                    elif tool_choice_auto and self.enable_reasoning:
                        assert tool_parser is not None
                        assert reasoning_parser is not None
                        assert added_content_delta_arr is not None
                        assert reasoning_end_arr is not None
                        if not reasoning_end_arr[i]:
                            delta_message = (
                                reasoning_parser.extract_reasoning_content_streaming(
                                    previous_text,
                                    current_text,
                                    delta_text,
                                    previous_token_ids,
                                    current_token_ids,
                                    output.token_ids,
                                )
                            )

                            # When encountering think end id in delta_token_ids,
                            # set reasoning status to end.
                            # Remove the text and token ids related
                            # to 'reasoning_content'.
                            if reasoning_parser.is_reasoning_end(
                                list(output.token_ids)
                            ):
                                reasoning_end_arr[i] = True
                                current_token_ids = (
                                    reasoning_parser.extract_content_ids(
                                        list(output.token_ids)
                                    )
                                )
                                if delta_message and delta_message.content:
                                    current_text = delta_message.content
                                    delta_message.content = None
                                else:
                                    current_text = ""

                        # handle tool calls only after reasoning is done,
                        else:
                            delta_token_ids = list(output.token_ids)
                            # First time to tool call,
                            # add the remaining text and token ids
                            # to delta from previous
                            if not added_content_delta_arr[i]:
                                added_content_delta_arr[i] = True
                                previous_text = ""
                                previous_token_ids = []
                                delta_text = current_text
                                delta_token_ids = current_token_ids

                            delta_message = tool_parser.extract_tool_calls_streaming(
                                previous_text=previous_text,
                                current_text=current_text,
                                delta_text=delta_text,
                                previous_token_ids=previous_token_ids,
                                current_token_ids=current_token_ids,
                                delta_token_ids=delta_token_ids,
                                request=request,
                            )
                    # when only tool calls
                    elif tool_choice_auto:
                        assert tool_parser is not None
                        delta_message = tool_parser.extract_tool_calls_streaming(
                            previous_text=previous_text,
                            current_text=current_text,
                            delta_text=delta_text,
                            previous_token_ids=previous_token_ids,
                            current_token_ids=current_token_ids,
                            delta_token_ids=output.token_ids,
                            request=request,
                        )
                    # when only reasoning
                    elif self.enable_reasoning:
                        assert reasoning_parser is not None
                        delta_message = (
                            reasoning_parser.extract_reasoning_content_streaming(
                                previous_text,
                                current_text,
                                delta_text,
                                previous_token_ids,
                                current_token_ids,
                                output.token_ids,
                            )
                        )
                    # handle streaming just a content delta
                    else:
                        delta_message = DeltaMessage(content=delta_text)

                    # update the previous values for the next iteration
                    if tool_choice_auto or should_stream_with_reasoning_parsing:
                        assert previous_texts is not None
                        assert all_previous_token_ids is not None
                        previous_texts[i] = current_text
                        all_previous_token_ids[i] = current_token_ids

                    # set the previous values for the next iteration
                    previous_num_tokens[i] += len(output.token_ids)

                    # if the message delta is None (e.g. because it was a
                    # "control token" for tool calls or the parser otherwise
                    # wasn't ready to send a token, then
                    #   get the next token without streaming a chunk
                    if delta_message is None:
                        continue

                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=None,
                        )

                    # if the model is finished generating
                    else:
                        # check to make sure we haven't "forgotten" to stream
                        #   any tokens that were generated but previously
                        #   matched by partial json parsing
                        # only happens if we are NOT using guided decoding
                        auto_tools_called = False
                        if tool_parser:
                            auto_tools_called = len(tool_parser.prev_tool_call_arr) > 0
                            index = (
                                len(tool_parser.prev_tool_call_arr) - 1
                                if auto_tools_called
                                else 0
                            )
                        else:
                            index = 0

                        if (
                            self._should_check_for_unstreamed_tool_arg_tokens(
                                delta_message, output
                            )
                            and tool_parser
                        ):
                            latest_delta_len = 0
                            if (
                                isinstance(
                                    delta_message.tool_calls[0].function,
                                    DeltaFunctionCall,
                                )
                            ) and isinstance(
                                delta_message.tool_calls[0].function.arguments, str
                            ):
                                latest_delta_len = len(
                                    delta_message.tool_calls[0].function.arguments
                                )

                            # get the expected call based on partial JSON
                            # parsing which "autocompletes" the JSON
                            expected_call = json.dumps(
                                tool_parser.prev_tool_call_arr[index].get(
                                    "arguments", {}
                                ),
                                ensure_ascii=False,
                            )

                            # get what we've streamed so far for arguments
                            # for the current tool
                            actual_call = tool_parser.streamed_args_for_tool[index]
                            if latest_delta_len > 0:
                                actual_call = actual_call[:-latest_delta_len]

                            # check to see if there's anything left to stream
                            remaining_call = expected_call.replace(actual_call, "", 1)
                            # set that as a delta message
                            delta_message = DeltaMessage(
                                tool_calls=[
                                    DeltaToolCall(
                                        index=index,
                                        function=DeltaFunctionCall(
                                            arguments=remaining_call
                                        ).model_dump(
                                            exclude_none=True
                                        ),  # type: ignore
                                    )
                                ]
                            )

                        # Send the finish response for each request.n only once
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=(
                                output.finish_reason
                                if not auto_tools_called
                                else "tool_calls"
                            ),
                            stop_reason=output.stop_reason,
                        )

                        finish_reason_sent[i] = True

                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name,
                    )

                    # handle usage stats if requested & if continuous
                    if include_continuous_usage:
                        completion_tokens = previous_num_tokens[i]
                        chunk.usage = UsageInfo(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=num_prompt_tokens + completion_tokens,
                        )

                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

            # once the final token is handled, if stream_options.include_usage
            # is sent, send the usage
            if include_usage:
                completion_tokens = sum(previous_num_tokens)
                final_usage = UsageInfo(
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=num_prompt_tokens + completion_tokens,
                )
                if self.enable_prompt_tokens_details and num_cached_tokens:
                    final_usage.prompt_tokens_details = PromptTokenUsageInfo(
                        cached_tokens=num_cached_tokens
                    )

                final_usage_chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage,
                )
                final_usage_data = final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True
                )
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            num_completion_tokens = sum(previous_num_tokens)
            request_metadata.final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_completion_tokens,
                total_tokens=num_prompt_tokens + num_completion_tokens,
            )

        except Exception as e:
            # TODO: Use a vllm-specific Validation Error
            logger.exception("Error in chat completion stream generator.")
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        benchmark_tracker: BenchmarkRequestTracker | None,
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
    ) -> ErrorResponse | ChatCompletionResponse:

        created_time = int(time.time())
        final_res: RequestOutput | None = None

        try:
            async for res in result_generator:
                if benchmark_tracker is not None:
                    benchmark_tracker.update(res)
                if final_res is None:
                    final_res = res
                else:
                    final_res = add_request_output_delta(final_res, res)
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        assert final_res is not None

        choices: list[ChatCompletionResponseChoice] = []

        role = self.get_chat_request_role(request)
        for output in final_res.outputs:
            token_ids = output.token_ids
            out_logprobs = output.logprobs

            if request.logprobs and request.top_logprobs is not None:
                assert out_logprobs is not None, "Did not output logprobs"
                logprobs = self._create_chat_logprobs(
                    token_ids=token_ids,
                    top_logprobs=out_logprobs,
                    num_output_top_logprobs=request.top_logprobs,
                    tokenizer=tokenizer,
                    return_as_token_id=request.return_tokens_as_token_ids,
                )
            else:
                logprobs = None

            should_stream_with_reasoning_parsing = (
                self._should_stream_with_reasoning_parsing(request)
            )

            # In the OpenAI API the finish_reason is "tools_called"
            # if the tool choice is auto and the model produced a tool
            # call. The same is not true for named function calls
            auto_tools_called = False

            if (
                should_stream_with_reasoning_parsing
                and self.reasoning_parser is not None
            ):
                try:
                    reasoning_parser = self.reasoning_parser(tokenizer)
                except RuntimeError as e:
                    logger.exception("Error in reasoning parser creation.")
                    return self.create_error_response(str(e))
                # If the reasoning parser is enabled,
                # tool calls are extracted exclusively from the content.
                reasoning_content, content = reasoning_parser.extract_reasoning_content(
                    output.text, request=request
                )
            else:
                reasoning_content = None
                content = output.text

            # if auto tools are not enabled, and a named tool choice using
            #   outlines is not being used
            if (not self.enable_auto_tools or not self.tool_parser) and (
                not isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam)
                and request.tool_choice != "required"
            ):
                message = ChatMessage(
                    role=role, reasoning_content=reasoning_content, content=content
                )

            # if the request uses tools and specified a tool choice
            elif (
                request.tool_choice
                and type(request.tool_choice) is ChatCompletionNamedToolChoiceParam
            ):

                tool_call_class = (
                    MistralToolCall
                    if isinstance(tokenizer, MistralTokenizer)
                    else ToolCall
                )
                message = ChatMessage(
                    role=role,
                    reasoning_content=reasoning_content,
                    content="",
                    tool_calls=[
                        tool_call_class(
                            function=FunctionCall(
                                name=request.tool_choice.function.name,
                                arguments=content,  # type:ignore
                            )
                        )
                    ],
                )

            elif request.tool_choice and request.tool_choice == "required":
                tool_call_class = (
                    MistralToolCall
                    if isinstance(tokenizer, MistralTokenizer)
                    else ToolCall
                )

                # the fields of FunctionDefinition are a superset of the
                # tool call outputs and can be used for parsing
                tool_calls = TypeAdapter(list[FunctionDefinition]).validate_json(
                    output.text
                )
                message = ChatMessage(
                    role=role,
                    content="",
                    tool_calls=[
                        tool_call_class(
                            function=FunctionCall(
                                name=tool_call.name,
                                arguments=json.dumps(tool_call.parameters),
                            )
                        )
                        for tool_call in tool_calls
                    ],
                )

            # if the request doesn't use tool choice
            # OR specifies to not use a tool
            elif not request.tool_choice or request.tool_choice == "none":

                message = ChatMessage(
                    role=role, reasoning_content=reasoning_content, content=content
                )

            # handle when there are tools and tool choice is auto
            elif (
                request.tools
                and (request.tool_choice == "auto" or request.tool_choice is None)
                and self.enable_auto_tools
                and self.tool_parser
            ):

                try:
                    tool_parser = self.tool_parser(tokenizer)
                except RuntimeError as e:
                    logger.exception("Error in tool parser creation.")
                    return self.create_error_response(str(e))

                tool_call_info = tool_parser.extract_tool_calls(
                    content if content is not None else "", request=request
                )
                # In the OpenAI API the finish_reason is "tools_called"
                # if the tool choice is auto and the model produced a tool
                # call. The same is not true for named function calls
                auto_tools_called = tool_call_info.tools_called
                if tool_call_info.tools_called:
                    message = ChatMessage(
                        role=role,
                        reasoning_content=reasoning_content,
                        content=tool_call_info.content,
                        tool_calls=tool_call_info.tool_calls,
                    )

                else:
                    # FOR NOW make it a chat message; we will have to detect
                    # the type to make it later.
                    message = ChatMessage(
                        role=role, reasoning_content=reasoning_content, content=content
                    )

            # undetermined case that is still important to handle
            else:
                logger.error(
                    "Error in chat_completion_full_generator - cannot determine"
                    " if tools should be extracted. Returning a standard chat "
                    "completion."
                )
                message = ChatMessage(
                    role=role, reasoning_content=reasoning_content, content=content
                )

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                logprobs=logprobs,
                finish_reason=(
                    "tool_calls"
                    if auto_tools_called
                    else output.finish_reason if output.finish_reason else "stop"
                ),
                stop_reason=output.stop_reason,
            )
            choices.append(choice_data)

        if request.echo:
            last_msg_content: str | list[dict[str, str]] = ""
            if (
                conversation
                and "content" in conversation[-1]
                and conversation[-1].get("role") == role
            ):
                last_msg_content = conversation[-1]["content"] or ""
            if isinstance(last_msg_content, list):
                last_msg_content = "\n".join(msg["text"] for msg in last_msg_content)

            for choice in choices:
                full_message = last_msg_content + (choice.message.content or "")
                choice.message.content = full_message

        assert final_res.prompt_token_ids is not None
        num_prompt_tokens = len(final_res.prompt_token_ids)
        if final_res.encoder_prompt_token_ids is not None:
            num_prompt_tokens += len(final_res.encoder_prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs
        )
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        if self.enable_prompt_tokens_details and final_res.num_cached_tokens:
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=final_res.num_cached_tokens
            )

        request_metadata.final_usage_info = usage

        metrics = None if benchmark_tracker is None else benchmark_tracker.finish()

        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
            prompt_logprobs=clamp_prompt_logprobs(final_res.prompt_logprobs),
            metrics=metrics,
        )

        return response

    def _get_top_logprobs(
        self,
        logprobs: dict[int, Logprob],
        top_logprobs: int | None,
        tokenizer: AnyTokenizer,
        should_return_as_token_id: bool,
    ) -> list[ChatCompletionLogProb]:
        return [
            ChatCompletionLogProb(
                token=(
                    token := self._get_decoded_token(
                        p[1],
                        p[0],
                        tokenizer,
                        return_as_token_id=should_return_as_token_id,
                    )
                ),
                logprob=max(p[1].logprob, -9999.0),
                bytes=list(token.encode("utf-8", errors="replace")),
            )
            for i, p in enumerate(logprobs.items())
            if top_logprobs and i < top_logprobs
        ]

    def _create_chat_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[dict[int, Logprob] | None],
        tokenizer: AnyTokenizer,
        num_output_top_logprobs: int | None = None,
        return_as_token_id: bool | None = None,
    ) -> ChatCompletionLogProbs:
        """Create OpenAI-style logprobs."""
        logprobs_content: list[ChatCompletionLogProbsContent] = []

        should_return_as_token_id = (
            return_as_token_id
            if return_as_token_id is not None
            else self.return_tokens_as_token_ids
        )
        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None:
                token = tokenizer.decode(token_id)
                if should_return_as_token_id:
                    token = f"token_id:{token_id}"

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=token,
                        bytes=list(token.encode("utf-8", errors="replace")),
                    )
                )
            else:
                step_token = step_top_logprobs[token_id]
                step_decoded = step_token.decoded_token

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=self._get_decoded_token(
                            step_token,
                            token_id,
                            tokenizer,
                            should_return_as_token_id,
                        ),
                        logprob=max(step_token.logprob, -9999.0),
                        bytes=(
                            None
                            if step_decoded is None
                            else list(step_decoded.encode("utf-8", errors="replace"))
                        ),
                        top_logprobs=self._get_top_logprobs(
                            step_top_logprobs,
                            num_output_top_logprobs,
                            tokenizer,
                            should_return_as_token_id,
                        ),
                    )
                )

        return ChatCompletionLogProbs(content=logprobs_content)

    def _should_stream_with_auto_tool_parsing(self, request: ChatCompletionRequest):
        """
        Utility function to check if streamed tokens should go through the tool
        call parser that was configured.

        We only want to do this IF user-provided tools are set, a tool parser
        is configured, "auto" tool choice is enabled, and the request's tool
        choice field indicates that "auto" tool choice should be used.
        """
        return (
            request.tools
            and self.tool_parser
            and self.enable_auto_tools
            and request.tool_choice in ["auto", None]
        )

    def _should_stream_with_reasoning_parsing(self, request: ChatCompletionRequest):
        """
        Utility function to check if streamed tokens should go through the
        reasoning parser that was configured.

        We only want to do this IF reasoning is enabled and a reasoning
        parser is configured.
        """
        return self.enable_reasoning and self.reasoning_parser is not None

    def _should_check_for_unstreamed_tool_arg_tokens(
        self,
        delta_message: DeltaMessage | None,
        output: CompletionOutput,
    ) -> bool:
        """
        Check to see if we should check for unstreamed tool arguments tokens.
        This is only applicable when auto tool parsing is enabled, the delta
        is a tool call with arguments.
        """

        # yapf: disable
        return bool(
            # if there is a delta message that includes tool calls which
            # include a function that has arguments
            output.finish_reason is not None
            and self.enable_auto_tools and self.tool_parser and delta_message
            and delta_message.tool_calls and delta_message.tool_calls[0]
            and delta_message.tool_calls[0].function
            and delta_message.tool_calls[0].function.arguments is not None
        )
