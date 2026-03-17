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
from vllm.vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.vllm.inputs import TokensPrompt

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
            lora_request = self._maybe_get_adapters(request)

            if isinstance(request, TokenizeChatRequest):
                (
                    _,
                    engine_prompts,
                ) = await self._preprocess_chat(
                    request,
                    request.messages,
                    request.chat_template or self.chat_template,
                    self.chat_template_content_format,
                    request.chat_template_kwargs,
                )
            else:
                engine_prompts = await self._preprocess_completion(
                    request,
                    request.prompt,
                    None,
                )
        except (ValueError, TypeError, jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        input_ids: list[int] = []
        for engine_prompt in engine_prompts:
            self._log_inputs(
                request_id,
                engine_prompt,
                params=None,
                lora_request=lora_request,
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

        lora_request = self._maybe_get_adapters(request)

        self._log_inputs(
            request_id,
            TokensPrompt(prompt_token_ids=request.tokens),
            params=None,
            lora_request=lora_request,
        )

        # Silently ignore prompt adapter since it does not affect tokenization
        # (Unlike in Embeddings API where an error is raised)

        prompt_input = await self.renderer.tokenize_prompt_async(
            TokensPrompt(prompt_token_ids=request.tokens),
            request.build_tok_params(self.model_config),
        )
        input_text = prompt_input["prompt"]

        return DetokenizeResponse(prompt=input_text)
