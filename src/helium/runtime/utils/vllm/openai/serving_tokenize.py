# SPDX-License-Identifier: Apache-2.0

from typing import Final

import jinja2

from helium.runtime.utils.vllm.openai.protocol import (
    DetokenizeRequest,
    DetokenizeResponse,
    ErrorResponse,
    TokenizeChatRequest,
    TokenizeRequest,
    TokenizeResponse,
)
from helium.runtime.utils.vllm.vllm_logger import init_child_logger
from vllm.vllm.config import ModelConfig
from vllm.vllm.engine.protocol import EngineClient
from vllm.vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.vllm.entrypoints.logger import RequestLogger
from vllm.vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.vllm.entrypoints.openai.serving_models import OpenAIServingModels

logger = init_child_logger("serving_tokenization")


class OpenAIServingTokenization(OpenAIServing):
    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            model_config=model_config,
            models=models,
            request_logger=request_logger,
        )

        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format

    async def create_tokenize(
        self, request_id: str, request: TokenizeRequest
    ) -> TokenizeResponse | ErrorResponse:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        try:
            (
                lora_request,
                prompt_adapter_request,
            ) = self._maybe_get_adapters(request)

            tokenizer = await self.engine_client.get_tokenizer(lora_request)

            if isinstance(request, TokenizeChatRequest):
                (
                    _,
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
                    chat_template_kwargs=request.chat_template_kwargs,
                    add_special_tokens=request.add_special_tokens,
                )
            else:
                (request_prompts, engine_prompts) = await self._preprocess_completion(
                    request,
                    tokenizer,
                    request.prompt,
                    add_special_tokens=request.add_special_tokens,
                )
        except (ValueError, TypeError, jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        input_ids: list[int] = []
        for i, engine_prompt in enumerate(engine_prompts):
            self._log_inputs(
                request_id,
                request_prompts[i],
                params=None,
                lora_request=lora_request,
                prompt_adapter_request=prompt_adapter_request,
            )

            # Silently ignore prompt adapter since it does not affect
            # tokenization (Unlike in Embeddings API where an error is raised)

            input_ids.extend(engine_prompt["prompt_token_ids"])

        return TokenizeResponse(
            tokens=input_ids, count=len(input_ids), max_model_len=self.max_model_len
        )

    async def create_detokenize(
        self, request_id: str, request: DetokenizeRequest
    ) -> DetokenizeResponse | ErrorResponse:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        (
            lora_request,
            prompt_adapter_request,
        ) = self._maybe_get_adapters(request)

        tokenizer = await self.engine_client.get_tokenizer(lora_request)

        self._log_inputs(
            request_id,
            request.tokens,
            params=None,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
        )

        # Silently ignore prompt adapter since it does not affect tokenization
        # (Unlike in Embeddings API where an error is raised)

        prompt_input = await self._tokenize_prompt_input_async(
            request,
            tokenizer,
            request.tokens,
        )
        input_text = prompt_input["prompt"]

        return DetokenizeResponse(prompt=input_text)
