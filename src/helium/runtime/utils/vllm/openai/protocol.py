import time
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field

from helium.runtime.utils.vllm.openai.serving_utils import BenchmarkMetrics
from vllm.vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionNamedFunction,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponse as OpenAIChatCompletionResponse,
)
from vllm.vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatCompletionToolsParam,
    ChatMessage,
)
from vllm.vllm.entrypoints.openai.completion.protocol import (
    CompletionLogProbs,
    CompletionRequest,
)
from vllm.vllm.entrypoints.openai.completion.protocol import (
    CompletionResponse as OpenAICompletionResponse,
)
from vllm.vllm.entrypoints.openai.completion.protocol import (
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
)
from vllm.vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ErrorResponse,
    ExtractedToolCallInformation,
    FunctionCall,
    FunctionDefinition,
    JsonSchemaResponseFormat,
    LogitsProcessorConstructor,
    LogitsProcessors,
    ModelCard,
    ModelList,
    ModelPermission,
    OpenAIBaseModel,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    ResponseFormat,
    StreamOptions,
    ToolCall,
    UsageInfo,
    get_logits_processors,
)
from vllm.vllm.entrypoints.openai.run_batch import (
    BatchRequestInput,
    BatchRequestOutput,
    BatchResponseData,
)
from vllm.vllm.entrypoints.openai.speech_to_text.protocol import (
    TranscriptionRequest,
    TranscriptionResponse,
)
from vllm.vllm.entrypoints.pooling.embed.protocol import (
    EmbeddingChatRequest,
    EmbeddingCompletionRequest,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingResponseData,
)
from vllm.vllm.entrypoints.pooling.pooling.protocol import (
    PoolingChatRequest,
    PoolingCompletionRequest,
    PoolingRequest,
    PoolingResponse,
    PoolingResponseData,
)
from vllm.vllm.entrypoints.pooling.score.protocol import (
    RerankRequest,
    RerankResponse,
    ScoreRequest,
    ScoreResponse,
    ScoreResponseData,
)
from vllm.vllm.entrypoints.serve.lora.protocol import (
    LoadLoRAAdapterRequest,
    UnloadLoRAAdapterRequest,
)
from vllm.vllm.entrypoints.serve.tokenize.protocol import (
    DetokenizeRequest,
    DetokenizeResponse,
    TokenizeChatRequest,
    TokenizeCompletionRequest,
    TokenizeRequest,
    TokenizeResponse,
)
from vllm.vllm.logprobs import Logprob
from vllm.vllm.utils import random_uuid

unused_imports = [
    OpenAIBaseModel,
    ErrorResponse,
    ModelPermission,
    ModelCard,
    ModelList,
    PromptTokenUsageInfo,
    UsageInfo,
    RequestResponseMetadata,
    JsonSchemaResponseFormat,
    ResponseFormat,
    StreamOptions,
    FunctionDefinition,
    ChatCompletionToolsParam,
    ChatCompletionNamedFunction,
    ChatCompletionNamedToolChoiceParam,
    LogitsProcessorConstructor,
    LogitsProcessors,
    get_logits_processors,
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingCompletionRequest,
    EmbeddingChatRequest,
    EmbeddingRequest,
    PoolingCompletionRequest,
    PoolingChatRequest,
    PoolingRequest,
    ScoreRequest,
    CompletionLogProbs,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    EmbeddingResponseData,
    EmbeddingResponse,
    PoolingResponseData,
    PoolingResponse,
    ScoreResponseData,
    ScoreResponse,
    FunctionCall,
    ToolCall,
    DeltaFunctionCall,
    DeltaToolCall,
    ExtractedToolCallInformation,
    ChatMessage,
    ChatCompletionLogProb,
    ChatCompletionLogProbsContent,
    ChatCompletionLogProbs,
    ChatCompletionResponseChoice,
    DeltaMessage,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    BatchRequestInput,
    BatchResponseData,
    BatchRequestOutput,
    TokenizeCompletionRequest,
    TokenizeChatRequest,
    TokenizeRequest,
    TokenizeResponse,
    DetokenizeRequest,
    DetokenizeResponse,
    LoadLoRAAdapterRequest,
    UnloadLoRAAdapterRequest,
    OpenAICompletionResponse,
    OpenAIChatCompletionResponse,
    TranscriptionRequest,
    TranscriptionResponse,
    RerankRequest,
    RerankResponse,
]


@dataclass
class CompletionResponse:
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str | None = None
    choices: list[CompletionResponseChoice] | None = None
    usage: UsageInfo | None = None
    metrics: list[BenchmarkMetrics] | None = None

    def to_openai(self) -> OpenAICompletionResponse:
        if self.model is None:
            raise ValueError("Model is not set")
        if self.choices is None:
            raise ValueError("Choices are not set")
        if self.usage is None:
            raise ValueError("Usage is not set")
        return OpenAICompletionResponse(
            id=self.id,
            object=self.object,
            created=self.created,
            model=self.model,
            choices=self.choices,
            usage=self.usage.model_dump(),
        )

    def model_dump(self) -> dict[str, Any]:
        return self.to_openai().model_dump()

    def model_dump_json(self) -> str:
        return self.to_openai().model_dump_json()


@dataclass
class ChatCompletionResponse:
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str | None = None
    choices: list[ChatCompletionResponseChoice] | None = None
    usage: UsageInfo | None = None
    prompt_logprobs: list[dict[int, Logprob] | None] | None = None
    metrics: BenchmarkMetrics | None = None

    def to_openai(self) -> OpenAIChatCompletionResponse:
        if self.model is None:
            raise ValueError("Model is not set")
        if self.choices is None:
            raise ValueError("Choices are not set")
        if self.usage is None:
            raise ValueError("Usage is not set")
        return OpenAIChatCompletionResponse(
            id=self.id,
            object=self.object,
            created=self.created,
            model=self.model,
            choices=self.choices,
            usage=self.usage.model_dump(),
            prompt_logprobs=self.prompt_logprobs,
        )

    def model_dump(self) -> dict[str, Any]:
        return self.to_openai().model_dump()

    def model_dump_json(self) -> str:
        return self.to_openai().model_dump_json()
