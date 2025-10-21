import os
import re
from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Mount
from typing_extensions import assert_never

import vllm.vllm.envs as envs
from helium.runtime.utils.vllm.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    DetokenizeRequest,
    DetokenizeResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingResponseData,
    ErrorResponse,
    LoadLoRAAdapterRequest,
    PoolingRequest,
    PoolingResponse,
    RerankRequest,
    RerankResponse,
    ScoreRequest,
    ScoreResponse,
    TokenizeRequest,
    TokenizeResponse,
    TranscriptionRequest,
    TranscriptionResponse,
    UnloadLoRAAdapterRequest,
)
from helium.runtime.utils.vllm.openai.serving_chat import OpenAIServingChat
from helium.runtime.utils.vllm.openai.serving_completion import OpenAIServingCompletion
from helium.runtime.utils.vllm.openai.serving_utils import benchmarker
from vllm.vllm.engine.protocol import EngineClient
from vllm.vllm.entrypoints.openai.api_server import logger
from vllm.vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.vllm.entrypoints.openai.serving_pooling import OpenAIServingPooling
from vllm.vllm.entrypoints.openai.serving_score import ServingScores
from vllm.vllm.entrypoints.openai.serving_tokenization import OpenAIServingTokenization
from vllm.vllm.entrypoints.openai.serving_transcription import (
    OpenAIServingTranscription,
)
from vllm.vllm.entrypoints.utils import load_aware_call, with_cancellation
from vllm.vllm.utils import Device
from vllm.vllm.version import __version__ as VLLM_VERSION  # type: ignore

router = APIRouter()


async def validate_json_request(raw_request: Request):
    content_type = raw_request.headers.get("content-type", "").lower()
    media_type = content_type.split(";", maxsplit=1)[0]
    if media_type != "application/json":
        raise HTTPException(
            status_code=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported Media Type: Only 'application/json' is allowed",
        )


def mount_metrics(app: FastAPI):
    # Lazy import for prometheus multiprocessing.
    # We need to set PROMETHEUS_MULTIPROC_DIR environment variable
    # before prometheus_client is imported.
    # See https://prometheus.github.io/client_python/multiprocess/
    from prometheus_client import (
        REGISTRY,
        CollectorRegistry,
        make_asgi_app,
        multiprocess,
    )
    from prometheus_fastapi_instrumentator import Instrumentator

    registry = REGISTRY

    prometheus_multiproc_dir_path = os.getenv("PROMETHEUS_MULTIPROC_DIR", None)
    if prometheus_multiproc_dir_path is not None:
        logger.debug(
            "vLLM to use %s as PROMETHEUS_MULTIPROC_DIR", prometheus_multiproc_dir_path
        )
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)

    Instrumentator(
        excluded_handlers=[
            "/metrics",
            "/health",
            "/load",
            "/ping",
            "/version",
            "/server_info",
        ],
        registry=registry,
    ).add().instrument(app).expose(app)

    # Add prometheus asgi middleware to route /metrics requests
    metrics_route = Mount("/metrics", make_asgi_app(registry=registry))

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)


def mount_benchmark(router: APIRouter) -> None:
    @router.get("/start_benchmark")
    def start_benchmark(request: Request) -> None:
        benchmarker.start_benchmark()

    @router.get("/stop_benchmark")
    def stop_benchmark(request: Request) -> JSONResponse:
        results = benchmarker.stop_benchmark()
        return JSONResponse(results)

    @router.post("/reset_prefix_cache")
    async def reset_prefix_cache(raw_request: Request):
        """
        Reset the prefix cache. Note that we currently do not check if the
        prefix cache is successfully reset in the API server.
        """
        device = None
        device_str = raw_request.query_params.get("device")
        if device_str is not None:
            device = Device[device_str.upper()]
        logger.info("Resetting prefix cache with specific %s...", str(device))
        await engine_client(raw_request).reset_prefix_cache(device)
        return Response(status_code=200)


def base(request: Request) -> OpenAIServing:
    # Reuse the existing instance
    return tokenization(request)


def models(request: Request) -> OpenAIServingModels:
    return request.app.state.openai_serving_models


def chat(request: Request) -> OpenAIServingChat | None:
    return request.app.state.openai_serving_chat


def completion(request: Request) -> OpenAIServingCompletion | None:
    return request.app.state.openai_serving_completion


def pooling(request: Request) -> OpenAIServingPooling | None:
    return request.app.state.openai_serving_pooling


def embedding(request: Request) -> OpenAIServingEmbedding | None:
    return request.app.state.openai_serving_embedding


def score(request: Request) -> ServingScores | None:
    return request.app.state.openai_serving_scores


def rerank(request: Request) -> ServingScores | None:
    return request.app.state.openai_serving_scores


def tokenization(request: Request) -> OpenAIServingTokenization:
    return request.app.state.openai_serving_tokenization


def transcription(request: Request) -> OpenAIServingTranscription:
    return request.app.state.openai_serving_transcription


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.get("/health")
async def health(raw_request: Request) -> Response:
    """Health check."""
    await engine_client(raw_request).check_health()
    return Response(status_code=200)


@router.get("/load")
async def get_server_load_metrics(request: Request):
    # This endpoint returns the current server load metrics.
    # It tracks requests utilizing the GPU from the following routes:
    # - /v1/chat/completions
    # - /v1/completions
    # - /v1/audio/transcriptions
    # - /v1/embeddings
    # - /pooling
    # - /score
    # - /v1/score
    # - /rerank
    # - /v1/rerank
    # - /v2/rerank
    return JSONResponse(content={"server_load": request.app.state.server_load_metrics})


@router.api_route("/ping", methods=["GET", "POST"])
async def ping(raw_request: Request) -> Response:
    """Ping check. Endpoint required for SageMaker"""
    return await health(raw_request)


@router.post("/tokenize", dependencies=[Depends(validate_json_request)])
@with_cancellation
async def tokenize(request: TokenizeRequest, raw_request: Request):
    handler = tokenization(raw_request)

    generator = await handler.create_tokenize(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    elif isinstance(generator, TokenizeResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post("/detokenize", dependencies=[Depends(validate_json_request)])
@with_cancellation
async def detokenize(request: DetokenizeRequest, raw_request: Request):
    handler = tokenization(raw_request)

    generator = await handler.create_detokenize(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    elif isinstance(generator, DetokenizeResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.get("/v1/models")
async def show_available_models(raw_request: Request):
    handler = models(raw_request)

    models_ = await handler.show_available_models()
    return JSONResponse(content=models_.model_dump())


@router.get("/version")
async def show_version():
    ver = {"version": VLLM_VERSION}
    return JSONResponse(content=ver)


@router.post("/v1/chat/completions", dependencies=[Depends(validate_json_request)])
@with_cancellation
@load_aware_call
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    handler = chat(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Chat Completions API"
        )

    generator = await handler.create_chat_completion(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)

    elif isinstance(generator, ChatCompletionResponse):
        benchmarker.add_metrics(generator.metrics)
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post("/v1/completions", dependencies=[Depends(validate_json_request)])
@with_cancellation
@load_aware_call
async def create_completion(request: CompletionRequest, raw_request: Request):
    handler = completion(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Completions API"
        )

    generator = await handler.create_completion(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    elif isinstance(generator, CompletionResponse):
        if generator.metrics is not None:
            for m in generator.metrics:
                benchmarker.add_metrics(m)
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post("/v1/embeddings", dependencies=[Depends(validate_json_request)])
@with_cancellation
@load_aware_call
async def create_embedding(request: EmbeddingRequest, raw_request: Request):
    handler = embedding(raw_request)
    if handler is None:
        fallback_handler = pooling(raw_request)
        if fallback_handler is None:
            return base(raw_request).create_error_response(
                message="The model does not support Embeddings API"
            )

        logger.warning(
            "Embeddings API will become exclusive to embedding models "
            "in a future release. To return the hidden states directly, "
            "use the Pooling API (`/pooling`) instead."
        )

        res = await fallback_handler.create_pooling(request, raw_request)

        generator: ErrorResponse | EmbeddingResponse
        if isinstance(res, PoolingResponse):
            generator = EmbeddingResponse(
                id=res.id,
                object=res.object,
                created=res.created,
                model=res.model,
                data=[
                    EmbeddingResponseData(
                        index=d.index,
                        embedding=d.data,  # type: ignore
                    )
                    for d in res.data
                ],
                usage=res.usage,
            )
        else:
            generator = res
    else:
        generator = await handler.create_embedding(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    elif isinstance(generator, EmbeddingResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post("/pooling", dependencies=[Depends(validate_json_request)])
@with_cancellation
@load_aware_call
async def create_pooling(request: PoolingRequest, raw_request: Request):
    handler = pooling(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Pooling API"
        )

    generator = await handler.create_pooling(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    elif isinstance(generator, PoolingResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post("/score", dependencies=[Depends(validate_json_request)])
@with_cancellation
@load_aware_call
async def create_score(request: ScoreRequest, raw_request: Request):
    handler = score(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Score API"
        )

    generator = await handler.create_score(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    elif isinstance(generator, ScoreResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post("/v1/score", dependencies=[Depends(validate_json_request)])
@with_cancellation
@load_aware_call
async def create_score_v1(request: ScoreRequest, raw_request: Request):
    logger.warning(
        "To indicate that Score API is not part of standard OpenAI API, we "
        "have moved it to `/score`. Please update your client accordingly."
    )

    return await create_score(request, raw_request)


@router.post("/v1/audio/transcriptions")
@with_cancellation
@load_aware_call
async def create_transcriptions(
    request: Annotated[TranscriptionRequest, Form()], raw_request: Request
):
    handler = transcription(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Transcriptions API"
        )

    audio_data = await request.file.read()
    generator = await handler.create_transcription(audio_data, request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)

    elif isinstance(generator, TranscriptionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post("/rerank", dependencies=[Depends(validate_json_request)])
@with_cancellation
@load_aware_call
async def do_rerank(request: RerankRequest, raw_request: Request):
    handler = rerank(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Rerank (Score) API"
        )
    generator = await handler.do_rerank(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    elif isinstance(generator, RerankResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post("/v1/rerank", dependencies=[Depends(validate_json_request)])
@with_cancellation
async def do_rerank_v1(request: RerankRequest, raw_request: Request):
    logger.warning_once(
        "To indicate that the rerank API is not part of the standard OpenAI"
        " API, we have located it at `/rerank`. Please update your client "
        "accordingly. (Note: Conforms to JinaAI rerank API)"
    )

    return await do_rerank(request, raw_request)


@router.post("/v2/rerank", dependencies=[Depends(validate_json_request)])
@with_cancellation
async def do_rerank_v2(request: RerankRequest, raw_request: Request):
    return await do_rerank(request, raw_request)


if envs.VLLM_TORCH_PROFILER_DIR:
    logger.warning(
        "Torch Profiler is enabled in the API server. This should ONLY be "
        "used for local development!"
    )

    @router.post("/start_profile")
    async def start_profile(raw_request: Request):
        logger.info("Starting profiler...")
        await engine_client(raw_request).start_profile()
        logger.info("Profiler started.")
        return Response(status_code=200)

    @router.post("/stop_profile")
    async def stop_profile(raw_request: Request):
        logger.info("Stopping profiler...")
        await engine_client(raw_request).stop_profile()
        logger.info("Profiler stopped.")
        return Response(status_code=200)


if envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING:
    logger.warning(
        "Lora dynamic loading & unloading is enabled in the API server. "
        "This should ONLY be used for local development!"
    )

    @router.post("/v1/load_lora_adapter")
    async def load_lora_adapter(request: LoadLoRAAdapterRequest, raw_request: Request):
        for route in [chat, completion, embedding]:
            handler = route(raw_request)
            if handler is not None:
                response = await handler.models.load_lora_adapter(request)
                if isinstance(response, ErrorResponse):
                    return JSONResponse(
                        content=response.model_dump(), status_code=response.code
                    )

        return Response(status_code=200, content=response)  # type: ignore

    @router.post("/v1/unload_lora_adapter")
    async def unload_lora_adapter(
        request: UnloadLoRAAdapterRequest, raw_request: Request
    ):
        for route in [chat, completion, embedding]:
            handler = route(raw_request)
            if handler is not None:
                response = await handler.models.unload_lora_adapter(request)
                if isinstance(response, ErrorResponse):
                    return JSONResponse(
                        content=response.model_dump(), status_code=response.code
                    )

        return Response(status_code=200, content=response)  # type: ignore
