import asyncio
import importlib
import inspect
import logging
import os
import signal
import socket
import sys
import uuid
from argparse import Namespace
from collections.abc import Sequence
from contextlib import AsyncExitStack
from http import HTTPStatus
from multiprocessing.synchronize import Event
from typing import Any

import uvicorn
import uvloop
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.datastructures import State

import vllm.vllm.envs as vllm_envs
from helium.runtime.utils.vllm.config import CompiledServerConfig, VLLMServerConfig
from helium.runtime.utils.vllm.engine.controller import (
    DispatchMethod,
    EngineClientController,
    EngineClientInfo,
)
from helium.runtime.utils.vllm.engine.mock import MockLLMEngine
from helium.runtime.utils.vllm.openai.protocol import ErrorResponse
from helium.runtime.utils.vllm.openai.router import (
    mount_benchmark,
    mount_metrics,
    router,
)
from helium.runtime.utils.vllm.openai.serving_chat import OpenAIServingChat
from helium.runtime.utils.vllm.openai.serving_completion import OpenAIServingCompletion
from vllm.vllm import __version__ as VLLM_VERSION
from vllm.vllm.config import ModelConfig
from vllm.vllm.engine.arg_utils import AsyncEngineArgs
from vllm.vllm.engine.protocol import EngineClient
from vllm.vllm.entrypoints.chat_utils import load_chat_template
from vllm.vllm.entrypoints.launcher import _add_shutdown_handlers
from vllm.vllm.entrypoints.logger import RequestLogger
from vllm.vllm.entrypoints.openai.api_server import (
    TIMEOUT_KEEP_ALIVE,
    build_async_engine_client,
    lifespan,
    logger,
)
from vllm.vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.vllm.entrypoints.openai.serving_models import (
    BaseModelPath,
    OpenAIServingModels,
)
from vllm.vllm.entrypoints.openai.serving_pooling import OpenAIServingPooling
from vllm.vllm.entrypoints.openai.serving_score import ServingScores
from vllm.vllm.entrypoints.openai.serving_tokenization import OpenAIServingTokenization
from vllm.vllm.entrypoints.openai.serving_transcription import (
    OpenAIServingTranscription,
)
from vllm.vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.vllm.entrypoints.ssl import SSLCertRefresher
from vllm.vllm.reasoning import ReasoningParserManager
from vllm.vllm.usage.usage_lib import UsageContext
from vllm.vllm.utils import find_process_using_port, is_valid_ipv6_address


def build_app(args: Namespace) -> FastAPI:
    if args.disable_fastapi_docs:
        app = FastAPI(
            openapi_url=None, docs_url=None, redoc_url=None, lifespan=lifespan
        )
    else:
        app = FastAPI(lifespan=lifespan)

    if args.benchmarking:
        mount_benchmark(router)

    app.include_router(router)
    app.root_path = args.root_path

    mount_metrics(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_, exc):
        err = ErrorResponse(
            message=str(exc), type="BadRequestError", code=HTTPStatus.BAD_REQUEST
        )
        return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)

    if token := vllm_envs.VLLM_API_KEY or args.api_key:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            if request.method == "OPTIONS":
                return await call_next(request)
            url_path = request.url.path
            if app.root_path and url_path.startswith(app.root_path):
                url_path = url_path[len(app.root_path) :]
            if not url_path.startswith("/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + token:
                return JSONResponse(content={"error": "Unauthorized"}, status_code=401)
            return await call_next(request)

    if args.enable_request_id_headers:
        logger.warning(
            "CAUTION: Enabling X-Request-Id headers in the API Server. "
            "This can harm performance at high QPS."
        )

        @app.middleware("http")
        async def add_request_id(request: Request, call_next):
            request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
            response = await call_next(request)
            response.headers["X-Request-Id"] = request_id
            return response

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)  # type: ignore[arg-type]
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(
                f"Invalid middleware {middleware}. " f"Must be a function or a class."
            )

    return app


async def init_app_state(
    engine_client: EngineClient,
    model_config: ModelConfig,
    state: State,
    args: Namespace,
) -> None:
    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    if args.disable_log_requests:
        request_logger = None
    else:
        request_logger = RequestLogger(max_log_len=args.max_log_len)

    base_model_paths = [
        BaseModelPath(name=name, model_path=args.model) for name in served_model_names
    ]

    state.engine_client = engine_client
    state.log_stats = not args.disable_log_stats

    resolved_chat_template = load_chat_template(args.chat_template)
    if resolved_chat_template is not None:
        logger.info("Using supplied chat template:\n%s", resolved_chat_template)

    state.openai_serving_models = OpenAIServingModels(
        engine_client=engine_client,
        model_config=model_config,
        base_model_paths=base_model_paths,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
    )
    await state.openai_serving_models.init_static_loras()
    state.openai_serving_chat = (
        OpenAIServingChat(
            engine_client,
            model_config,
            state.openai_serving_models,
            args.response_role,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_auto_tools=args.enable_auto_tool_choice,
            tool_parser=args.tool_call_parser,
            enable_reasoning=args.enable_reasoning,
            reasoning_parser=args.reasoning_parser,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            benchmarking=args.benchmarking,
        )
        if model_config.runner_type == "generate"
        else None
    )
    state.openai_serving_completion = (
        OpenAIServingCompletion(
            engine_client,
            model_config,
            state.openai_serving_models,
            request_logger=request_logger,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            benchmarking=args.benchmarking,
        )
        if model_config.runner_type == "generate"
        else None
    )
    state.openai_serving_pooling = (
        OpenAIServingPooling(
            engine_client,
            model_config,
            state.openai_serving_models,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
        )
        if model_config.runner_type == "pooling"
        else None
    )
    state.openai_serving_embedding = (
        OpenAIServingEmbedding(
            engine_client,
            model_config,
            state.openai_serving_models,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
        )
        if model_config.task == "embed"
        else None
    )
    state.openai_serving_scores = (
        ServingScores(
            engine_client,
            model_config,
            state.openai_serving_models,
            request_logger=request_logger,
        )
        if model_config.task in ("score", "embed", "pooling")
        else None
    )
    state.jinaai_serving_reranking = (
        ServingScores(
            engine_client,
            model_config,
            state.openai_serving_models,
            request_logger=request_logger,
        )
        if model_config.task == "score"
        else None
    )
    state.openai_serving_tokenization = OpenAIServingTokenization(
        engine_client,
        model_config,
        state.openai_serving_models,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
    )
    state.openai_serving_transcription = (
        OpenAIServingTranscription(
            engine_client,
            model_config,
            state.openai_serving_models,
            request_logger=request_logger,
        )
        if model_config.runner_type == "transcription"
        else None
    )
    state.task = model_config.task

    state.enable_server_load_tracking = args.enable_server_load_tracking
    state.server_load_metrics = 0


async def serve_http(
    app: FastAPI,
    sock: socket.socket | None,
    event: Event | None = None,
    enable_ssl_refresh: bool = False,
    **uvicorn_kwargs: Any,
):
    logger.info("Available routes are:")
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)

        if methods is None or path is None:
            continue

        logger.info("Route: %s, Methods: %s", path, ", ".join(methods))
    config = uvicorn.Config(app, **uvicorn_kwargs)
    config.load()
    server = uvicorn.Server(config)
    _add_shutdown_handlers(app, server)

    loop = asyncio.get_running_loop()

    server_task = loop.create_task(server.serve(sockets=[sock] if sock else None))

    ssl_cert_refresher = (
        None
        if not enable_ssl_refresh
        else SSLCertRefresher(
            ssl_context=config.ssl,  # type: ignore
            key_path=config.ssl_keyfile,  # type: ignore
            cert_path=config.ssl_certfile,  # type: ignore
            ca_path=config.ssl_ca_certs,
        )
    )

    def signal_handler() -> None:
        # prevents the uvicorn signal handler to exit early
        server_task.cancel()
        if ssl_cert_refresher:
            ssl_cert_refresher.stop()

    async def dummy_shutdown() -> None:
        pass

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    if event is not None:
        # Need some delay to ensure the API server is up and running
        asyncio.get_event_loop().call_later(1, lambda: event.set())

    try:
        await server_task
        return dummy_shutdown()
    except asyncio.CancelledError:
        port = uvicorn_kwargs["port"]
        process = find_process_using_port(port)
        if process is not None:
            logger.debug(
                "port %s is used by process %s launched with command:\n%s",
                port,
                process,
                " ".join(process.cmdline()),
            )
        logger.info("Shutting down FastAPI HTTP server.")
        return server.shutdown()


def create_server_socket(addr: tuple[str, int]) -> socket.socket:
    family = socket.AF_INET
    if is_valid_ipv6_address(addr[0]):
        family = socket.AF_INET6

    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind(addr)

    return sock


def configure_vllm_logging(config: CompiledServerConfig) -> None:
    vllm_logger = logging.getLogger("vllm")
    for handler in vllm_logger.handlers:
        vllm_logger.removeHandler(handler)
    if config.args.log_file is not None:
        handler = logging.FileHandler(config.args.log_file, mode="a")
        handler.setLevel(config.inner.log_level)
        vllm_logger.addHandler(handler)


async def run_server(
    config: CompiledServerConfig,
    event: Event | None = None,
    **uvicorn_kwargs,
) -> None:
    args = config.args

    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    valid_tool_parses = ToolParserManager.tool_parsers.keys()
    if args.enable_auto_tool_choice and args.tool_call_parser not in valid_tool_parses:
        raise KeyError(
            f"invalid tool call parser: {args.tool_call_parser} "
            f"(chose from {{ {','.join(valid_tool_parses)} }})"
        )

    valid_reasoning_parses = ReasoningParserManager.reasoning_parsers.keys()
    if args.enable_reasoning and args.reasoning_parser not in valid_reasoning_parses:
        raise KeyError(
            f"invalid reasoning parser: {args.reasoning_parser} "
            f"(chose from {{ {','.join(valid_reasoning_parses)} }})"
        )

    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    # see https://github.com/vllm-project/vllm/issues/8204
    sock_addr = (args.host or "", args.port)
    sock = create_server_socket(sock_addr)

    # workaround to avoid footguns where uvicorn drops requests with too
    # many concurrent requests active
    # set_ulimit()

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.inner.cuda_device or "0"

    async with build_async_engine_client(args) as engine_client:
        app = build_app(args)

        model_config = await engine_client.get_model_config()
        await init_app_state(engine_client, model_config, app.state, args)

        def _listen_addr(a: str) -> str:
            if is_valid_ipv6_address(a):
                return "[" + a + "]"
            return a or "0.0.0.0"

        is_ssl = args.ssl_keyfile and args.ssl_certfile
        logger.info(
            "Starting vLLM API server on http%s://%s:%d",
            "s" if is_ssl else "",
            _listen_addr(sock_addr[0]),
            sock_addr[1],
        )

        shutdown_task = await serve_http(
            app,
            sock=sock,
            event=event,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    await shutdown_task

    sock.close()


def start_server(config: CompiledServerConfig, event: Event | None = None) -> None:
    args = config.args
    log_level = (
        config.inner.log_level
        if getattr(args, "log_level", None) is None
        else args.log_level
    )
    os.environ["VLLM_LOGGING_LEVEL"] = log_level

    if args.profiling:
        os.environ["VLLM_TORCH_PROFILER_DIR"] = str(config.inner.trace_dir)

    if args.use_v1:
        os.environ["VLLM_USE_V1"] = "1"
    else:
        os.environ["VLLM_USE_V1"] = "0"

    uvloop.run(run_server(config, event))


async def run_server_with_controller(
    controller_args: Namespace,
    worker_configs: Sequence[CompiledServerConfig],
    worker_infos: Sequence[EngineClientInfo],
    dispatch_method: DispatchMethod = DispatchMethod.SHORTEST_QUEUE,
    event: Event | None = None,
    **uvicorn_kwargs,
) -> None:
    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", controller_args)

    if (
        controller_args.tool_parser_plugin
        and len(controller_args.tool_parser_plugin) > 3
    ):
        ToolParserManager.import_tool_parser(controller_args.tool_parser_plugin)

    valid_tool_parses = ToolParserManager.tool_parsers.keys()
    if (
        controller_args.enable_auto_tool_choice
        and controller_args.tool_call_parser not in valid_tool_parses
    ):
        raise KeyError(
            f"invalid tool call parser: {controller_args.tool_call_parser} "
            f"(chose from {{ {','.join(valid_tool_parses)} }})"
        )

    valid_reasoning_parses = ReasoningParserManager.reasoning_parsers.keys()
    if (
        controller_args.enable_reasoning
        and controller_args.reasoning_parser not in valid_reasoning_parses
    ):
        raise KeyError(
            f"invalid reasoning parser: {controller_args.reasoning_parser} "
            f"(chose from {{ {','.join(valid_reasoning_parses)} }})"
        )

    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    # see https://github.com/vllm-project/vllm/issues/8204
    sock_addr = (controller_args.host or "", controller_args.port)
    sock = create_server_socket(sock_addr)

    # workaround to avoid footguns where uvicorn drops requests with too
    # many concurrent requests active
    # set_ulimit()

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    async with AsyncExitStack() as stack:
        engine_clients: list[EngineClient] = []
        for worker_config in worker_configs:
            os.environ["CUDA_VISIBLE_DEVICES"] = worker_config.inner.cuda_device or "0"
            engine_client = await stack.enter_async_context(
                build_async_engine_client(worker_config.args)
            )
            engine_clients.append(engine_client)

        app = build_app(controller_args)

        controller = EngineClientController(
            engine_clients, worker_infos, dispatch_method
        )

        model_config = await controller.get_model_config()
        await init_app_state(controller, model_config, app.state, controller_args)

        def _listen_addr(a: str) -> str:
            if is_valid_ipv6_address(a):
                return "[" + a + "]"
            return a or "0.0.0.0"

        is_ssl = controller_args.ssl_keyfile and controller_args.ssl_certfile
        logger.info(
            "Starting vLLM API server with controller on http%s://%s:%d",
            "s" if is_ssl else "",
            _listen_addr(sock_addr[0]),
            sock_addr[1],
        )

        shutdown_task = await serve_http(
            app,
            sock=sock,
            event=event,
            enable_ssl_refresh=controller_args.enable_ssl_refresh,
            host=controller_args.host,
            port=controller_args.port,
            log_level=controller_args.uvicorn_log_level,
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=controller_args.ssl_keyfile,
            ssl_certfile=controller_args.ssl_certfile,
            ssl_ca_certs=controller_args.ssl_ca_certs,
            ssl_cert_reqs=controller_args.ssl_cert_reqs,
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    await shutdown_task

    sock.close()


def start_server_with_controller(
    controller_config: CompiledServerConfig,
    worker_configs: Sequence[CompiledServerConfig],
    worker_infos: Sequence[EngineClientInfo],
    dispatch_method: DispatchMethod = DispatchMethod.SHORTEST_QUEUE,
    event: Event | None = None,
) -> None:
    if len(worker_configs) == 0:
        raise ValueError("Worker arguments must not be empty.")
    if len(worker_configs) != len(worker_infos):
        raise ValueError("The number of worker arguments and worker infos must match.")

    controller_args = controller_config.args
    log_level = (
        controller_config.inner.log_level
        if getattr(controller_args, "log_level", None) is None
        else controller_args.log_level
    )
    os.environ["VLLM_LOGGING_LEVEL"] = log_level

    if controller_args.profiling:
        os.environ["VLLM_TORCH_PROFILER_DIR"] = str(controller_config.inner.trace_dir)

    if controller_args.use_v1:
        os.environ["VLLM_USE_V1"] = "1"
    else:
        os.environ["VLLM_USE_V1"] = "0"

    uvloop.run(
        run_server_with_controller(
            controller_args, worker_configs, worker_infos, dispatch_method, event
        )
    )


async def run_mock_server(
    config: CompiledServerConfig,
    event: Event | None = None,
    **uvicorn_kwargs,
) -> None:
    args = config.args

    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    valid_tool_parses = ToolParserManager.tool_parsers.keys()
    if args.enable_auto_tool_choice and args.tool_call_parser not in valid_tool_parses:
        raise KeyError(
            f"invalid tool call parser: {args.tool_call_parser} "
            f"(chose from {{ {','.join(valid_tool_parses)} }})"
        )

    valid_reasoning_parses = ReasoningParserManager.reasoning_parsers.keys()
    if args.enable_reasoning and args.reasoning_parser not in valid_reasoning_parses:
        raise KeyError(
            f"invalid reasoning parser: {args.reasoning_parser} "
            f"(chose from {{ {','.join(valid_reasoning_parses)} }})"
        )

    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    # see https://github.com/vllm-project/vllm/issues/8204
    sock_addr = (args.host or "", args.port)
    sock = create_server_socket(sock_addr)

    # workaround to avoid footguns where uvicorn drops requests with too
    # many concurrent requests active
    # set_ulimit()

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    engine_args = AsyncEngineArgs.from_cli_args(config.args)
    vllm_config = engine_args.create_engine_config(
        usage_context=UsageContext.OPENAI_API_SERVER
    )
    engine_client = MockLLMEngine(vllm_config)

    app = build_app(args)

    model_config = await engine_client.get_model_config()
    await init_app_state(engine_client, model_config, app.state, args)

    def _listen_addr(a: str) -> str:
        if is_valid_ipv6_address(a):
            return "[" + a + "]"
        return a or "0.0.0.0"

    is_ssl = args.ssl_keyfile and args.ssl_certfile
    logger.info(
        "Starting vLLM API server on http%s://%s:%d",
        "s" if is_ssl else "",
        _listen_addr(sock_addr[0]),
        sock_addr[1],
    )

    shutdown_task = await serve_http(
        app,
        sock=sock,
        event=event,
        enable_ssl_refresh=args.enable_ssl_refresh,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        **uvicorn_kwargs,
    )

    # NB: Await server shutdown only after the backend context is exited
    await shutdown_task

    sock.close()


def start_mock_server(config: CompiledServerConfig, event: Event | None = None) -> None:
    args = config.args
    log_level = (
        config.inner.log_level
        if getattr(args, "log_level", None) is None
        else args.log_level
    )
    os.environ["VLLM_LOGGING_LEVEL"] = log_level

    if sys.platform.startswith("darwin"):
        # vLLM only supports CPU on macOS
        config.inner.device = "cpu"
        config.args.device = "cpu"
    else:
        if args.profiling:
            os.environ["VLLM_TORCH_PROFILER_DIR"] = str(config.inner.trace_dir)

        if args.use_v1:
            os.environ["VLLM_USE_V1"] = "1"
        else:
            os.environ["VLLM_USE_V1"] = "0"

    uvloop.run(run_mock_server(config, event))


if __name__ == "__main__":
    config = VLLMServerConfig()
    cuda_device = config.cuda_device
    if cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    start_server(config.compile())
