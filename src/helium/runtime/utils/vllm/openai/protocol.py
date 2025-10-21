import time
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field

from helium.runtime.utils.vllm.openai.serving_utils import BenchmarkMetrics
from vllm.vllm.entrypoints.openai.protocol import (
    BatchRequestInput,
    BatchRequestOutput,
    BatchResponseData,
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionNamedFunction,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.vllm.entrypoints.openai.protocol import (
    ChatCompletionResponse as OpenAIChatCompletionResponse,
)
from vllm.vllm.entrypoints.openai.protocol import (
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatCompletionToolsParam,
    ChatMessage,
    CompletionLogProbs,
    CompletionRequest,
)
from vllm.vllm.entrypoints.openai.protocol import (
    CompletionResponse as OpenAICompletionResponse,
)
from vllm.vllm.entrypoints.openai.protocol import (
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    DetokenizeRequest,
    DetokenizeResponse,
    EmbeddingChatRequest,
    EmbeddingCompletionRequest,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingResponseData,
    ErrorResponse,
    ExtractedToolCallInformation,
    FunctionCall,
    FunctionDefinition,
    JsonSchemaResponseFormat,
    LoadLoRAAdapterRequest,
    LogitsProcessorConstructor,
    LogitsProcessors,
    ModelCard,
    ModelList,
    ModelPermission,
    OpenAIBaseModel,
    PoolingChatRequest,
    PoolingCompletionRequest,
    PoolingRequest,
    PoolingResponse,
    PoolingResponseData,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    RerankRequest,
    RerankResponse,
    ResponseFormat,
    ScoreRequest,
    ScoreResponse,
    ScoreResponseData,
    StreamOptions,
    TokenizeChatRequest,
    TokenizeCompletionRequest,
    TokenizeRequest,
    TokenizeResponse,
    ToolCall,
    TranscriptionRequest,
    TranscriptionResponse,
    UnloadLoRAAdapterRequest,
    UsageInfo,
    get_logits_processors,
)
from vllm.vllm.sequence import Logprob
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
            usage=self.usage,
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
            usage=self.usage,
            prompt_logprobs=self.prompt_logprobs,
        )

    def model_dump(self) -> dict[str, Any]:
        return self.to_openai().model_dump()

    def model_dump_json(self) -> str:
        return self.to_openai().model_dump_json()
