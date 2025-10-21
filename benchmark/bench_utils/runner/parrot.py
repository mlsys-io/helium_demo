import asyncio
import logging
import multiprocessing as mp
import os
import signal
from collections.abc import Coroutine
from dataclasses import asdict, dataclass, field
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import Any

import requests
import uvloop
from bench_utils.runner.base import (
    BenchmarkRunner,
    EngineClientInfo,
    RunnerConfig,
    start_gpu_monitor,
    start_llm_server,
    start_llm_server_with_controller,
)
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from parrot.constants import ENGINE_TYPE_OPENAI
from parrot.engine.config import EngineConfig
from parrot.engine.openai.openai_engine import OpenAIEngine
from parrot.serve.config import ServeCoreConfig
from parrot.serve.core import ParrotServeCore
from parrot.utils import redirect_stdout_stderr_to_file, set_log_output_file
from uvicorn import Config, Server

from helium.runtime.utils.vllm.config import BenchVLLMServerConfig
from helium.runtime.utils.vllm.utils import (
    async_request_reset_prefix_cache,
    request_start_benchmark,
    request_stop_benchmark,
    strip_v1_suffix,
)

_UNLIMITED = 999999999


def _default_scheduler_config() -> dict[str, Any]:
    return {
        "app_fifo": True,
        "graph_group": False,
        "ctx_group": True,
        "ctx_aware": True,
        "max_queue_size": _UNLIMITED,
    }


def _default_engine_scheduler_config() -> dict[str, Any]:
    return {
        "max_batch_size": _UNLIMITED,
        "max_num_batched_tokens": _UNLIMITED,
        "max_total_tokens": _UNLIMITED,
    }


@dataclass
class ParrotServerConfig:
    host: str
    port: int
    engine_info: EngineClientInfo
    max_sessions_num: int = _UNLIMITED
    max_engines_num: int = _UNLIMITED
    session_life_span: int = _UNLIMITED
    global_scheduler: dict[str, Any] = field(default_factory=_default_scheduler_config)

    release_mode: bool = True
    log_file: Path | None = None
    log_level: str = "warning"

    def to_dict(self) -> dict[str, Any]:
        config_dict = asdict(self)
        del config_dict["engine_info"]
        del config_dict["release_mode"]
        del config_dict["log_file"]
        del config_dict["log_level"]
        return config_dict


@dataclass
class ParrotEngineConfig:
    engine_name: str
    model: str
    tokenizer: str
    host: str
    port: int
    instance: dict[str, Any]
    serve_core: dict[str, Any]
    tokens_capacity: int
    engine_type: str = ENGINE_TYPE_OPENAI
    random_seed: int = 0
    tasks_capacity: int = _UNLIMITED
    scheduler: dict[str, Any] = field(default_factory=_default_engine_scheduler_config)

    log_file: Path | None = None
    log_level: str = "warning"

    @classmethod
    def from_server_configs(
        cls,
        engine_name: str,
        vllm_config: BenchVLLMServerConfig,
        core_config: ParrotServerConfig,
        cache_capacity: int | None = None,
    ) -> "ParrotEngineConfig":
        if core_config.log_file is None:
            log_file = None
        else:
            suffix = core_config.log_file.suffix
            log_file = core_config.log_file.with_suffix(f".{engine_name}{suffix}")
        return cls(
            engine_name=engine_name,
            model=vllm_config.model,
            tokenizer=vllm_config.model,
            host=vllm_config.host,
            port=vllm_config.port + 1,
            instance={
                "api_key": "EMPTY",
                "api_endpoint": "completion",  # or "chat"
                "base_url": vllm_config.base_url,
                "is_azure": False,
            },
            serve_core={"host": core_config.host, "port": core_config.port},
            tokens_capacity=cache_capacity or _UNLIMITED,
            log_file=log_file,
        )

    def to_dict(self) -> dict[str, Any]:
        config_dict = asdict(self)
        del config_dict["log_file"]
        del config_dict["log_level"]
        return config_dict


class BenchmarkRunnerWithParrot(BenchmarkRunner):
    def __init__(
        self,
        config: RunnerConfig,
        devices: list[int],
        parrot_config: ParrotServerConfig,
        main_config: BenchVLLMServerConfig,
        worker_configs: list[BenchVLLMServerConfig] | None,
        engine_infos: list[EngineClientInfo],
        gpu_util_log_dir: Path | None = None,
        use_parrot_router: bool = True,
    ) -> None:
        super().__init__(config)
        self.devices = devices
        self.gpu_util_log_dir = gpu_util_log_dir
        self.parrot_config = parrot_config
        self.main_config = main_config
        self.worker_configs = worker_configs
        self.engine_infos = engine_infos
        self.use_parrot_router = use_parrot_router
        self._server_proc: mp.Process | None = None
        self._engine_procs: list[mp.Process] | None = None
        self._worker_procs: list[mp.Process] | None = None
        self._gpu_monitor_proc: mp.Process | None = None

    async def init_run(self, run_name: str) -> None:
        if self._server_proc is not None or self._engine_procs is not None:
            raise ValueError("Server process already started")

        # Start parrot server
        self._start_parrot_server()

        if self.gpu_util_log_dir is not None:
            if self._gpu_monitor_proc is not None:
                raise ValueError("GPU monitor process already started")
            log_file = self.gpu_util_log_dir / f"{run_name}.log"
            self._gpu_monitor_proc = start_gpu_monitor(log_file, self.devices)

    async def clean_up_run(self) -> None:
        if self._server_proc is None:
            raise ValueError("Server process not started")

        # Stop parrot server
        self._stop_parrot_server()

        # Reset the prefix cache
        main_config = self.main_config
        worker_configs = self.worker_configs
        if worker_configs is None or not self.use_parrot_router:
            base_url = strip_v1_suffix(main_config.base_url)
            await async_request_reset_prefix_cache(base_url)
        else:
            for config in worker_configs:
                base_url = strip_v1_suffix(config.base_url)
                await async_request_reset_prefix_cache(base_url)

        if self.gpu_util_log_dir is not None:
            self._stop_gpu_monitor()

    async def start_runner(self) -> None:
        if self._worker_procs is not None:
            raise ValueError("LLM worker processes already started")
        self._start_llm_workers()

    async def stop_runner(self) -> None:
        if self._worker_procs is None:
            raise ValueError("LLM worker processes not started")
        self._stop_llm_workers()

    def _start_llm_workers(self) -> None:
        main_config = self.main_config
        worker_configs = self.worker_configs
        engine_infos = self.engine_infos

        num_workers = (
            1
            if worker_configs is None or not self.use_parrot_router
            else len(worker_configs)
        )
        worker_events = [mp.Event() for _ in range(num_workers)]

        # Start vLLM server
        if worker_configs is None:
            # Single LLM engine
            worker_procs = [
                mp.Process(
                    target=start_llm_server,
                    args=(main_config.compile([]), worker_events[0]),
                )
            ]
        elif self.use_parrot_router:
            # Multiple LLM engines
            worker_procs = [
                mp.Process(target=start_llm_server, args=(config.compile([]), event))
                for config, event in zip(worker_configs, worker_events)
            ]
        else:
            # Multiple LLM engines with a controller
            if engine_infos is None:
                raise ValueError("Expected engine infos")
            worker_procs = [
                mp.Process(
                    target=start_llm_server_with_controller,
                    args=(
                        main_config.compile([]),
                        [config.compile([]) for config in worker_configs],
                        engine_infos,
                        worker_events[0],
                    ),
                )
            ]

        for proc in worker_procs:
            proc.start()
        self._worker_procs = worker_procs

        for event in worker_events:
            event.wait()

    def _start_parrot_server(self) -> None:
        if self._worker_procs is None:
            raise ValueError("LLM worker processes not started")

        parrot_config = self.parrot_config
        main_config = self.main_config
        worker_configs = self.worker_configs

        server_event = mp.Event()
        num_workers = len(self._worker_procs)
        engine_events = [mp.Event() for _ in range(num_workers)]

        # Remove Parrot server's log file
        if parrot_config.log_file is not None:
            log_file = parrot_config.log_file
            for f in log_file.parent.glob(f"{log_file.stem}*"):
                f.unlink(missing_ok=True)
        # Start Parrot server
        server_proc = mp.Process(
            target=_start_parrot_server, args=(parrot_config, server_event)
        )
        server_proc.start()
        server_event.wait()
        self._server_proc = server_proc

        # Start Parrot engine
        if worker_configs is None or not self.use_parrot_router:
            # Single Parrot engine
            worker_config = main_config if worker_configs is None else worker_configs[0]
            engine_config = ParrotEngineConfig.from_server_configs(
                "engine", worker_config, parrot_config, cache_capacity=None
            )
            engine_log_files = [engine_config.log_file]
            engine_procs = [
                mp.Process(
                    target=_start_parrot_engine, args=(engine_config, engine_events[0])
                )
            ]
        else:
            # Multiple Parrot engines
            engine_configs = [
                ParrotEngineConfig.from_server_configs(
                    f"engine_{i}", config, parrot_config, cache_capacity=None
                )
                for i, config in enumerate(worker_configs)
            ]
            engine_log_files = [config.log_file for config in engine_configs]
            engine_procs = [
                mp.Process(
                    target=_start_parrot_engine,
                    args=(config, event),
                )
                for config, event in zip(engine_configs, engine_events)
            ]
        # Remove Parrot engine's log file
        for engine_log_file in engine_log_files:
            if engine_log_file is not None:
                for f in engine_log_file.parent.glob(f"{engine_log_file.stem}*"):
                    f.unlink(missing_ok=True)

        for proc in engine_procs:
            proc.start()
        self._engine_procs = engine_procs

        for event in engine_events:
            event.wait()

    def _stop_llm_workers(self) -> None:
        worker_procs = self._worker_procs
        assert worker_procs is not None

        for proc in worker_procs:
            proc.terminate()
        for proc in worker_procs:
            proc.join()

        self._worker_procs = None

    def _stop_parrot_server(self) -> None:
        server_proc = self._server_proc
        engine_procs = self._engine_procs
        assert not (server_proc is None or engine_procs is None)

        for proc in engine_procs:
            proc.terminate()
        for proc in engine_procs:
            proc.join()

        server_proc.terminate()
        server_proc.join()

        self._server_proc = None
        self._engine_procs = None

    def _stop_gpu_monitor(self) -> None:
        if self._gpu_monitor_proc is None:
            raise ValueError("GPU monitor process not started")
        self._gpu_monitor_proc.terminate()
        self._gpu_monitor_proc.join()
        self._gpu_monitor_proc = None


async def _run_server_and_backend(
    server: Server, backend_loop: Coroutine, event: Event
) -> None:
    loop = asyncio.get_running_loop()
    # Run the backend coro
    backend_task = loop.create_task(backend_loop)
    # Run the Uvicorn server
    server_task = loop.create_task(server.serve())

    # Configure signal handlers
    def signal_handler() -> None:
        backend_task.cancel()
        server_task.cancel()

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    # Notify that the server is ready
    asyncio.get_event_loop().call_later(1, lambda: event.set())

    try:
        await server_task
    except asyncio.CancelledError:
        await server.shutdown()


def _configure_parrot_logging(log_level: str, log_file: Path | None) -> None:
    level = logging.getLevelNamesMapping()[log_level.upper()]
    logging.disable(level - 10)
    if log_file is not None:
        log_dir = str(log_file.parent)
        log_file_name = str(log_file.name)
        set_log_output_file(log_dir, log_file_name)
        redirect_stdout_stderr_to_file(log_dir, log_file_name + ".out")


def _create_server_core(config: ParrotServerConfig) -> ParrotServeCore:
    config_dict = config.to_dict()
    if ServeCoreConfig.verify_config(config_dict):
        return ParrotServeCore(config_dict)
    raise ValueError("Invalid Parrot server configuration")


def _mount_server_benchmark_endpoints(app: FastAPI, pcore: ParrotServeCore) -> None:
    @app.get("/start_benchmark")
    def start_benchmark() -> None:
        for engine in pcore.engine_mgr.engines.values():
            config = engine.config
            base_url = f"http://{config.host}:{config.port}"
            try:
                request_start_benchmark(base_url)
            except Exception:
                pass

    @app.get("/stop_benchmark")
    def stop_benchmark() -> JSONResponse:
        bench_results: dict[str, dict[str, Any]] = {}
        for engine in pcore.engine_mgr.engines.values():
            config = engine.config
            base_url = f"http://{config.host}:{config.port}"
            try:
                bench_results[base_url] = request_stop_benchmark(base_url)
            except Exception:
                bench_results[base_url] = {}
        return JSONResponse(bench_results)

    @app.get("/metrics")
    def metrics() -> JSONResponse:
        metrics: dict[str, str] = {}
        for engine in pcore.engine_mgr.engines.values():
            config = engine.config
            base_url = f"http://{config.host}:{config.port}"
            try:
                res = requests.get(f"{base_url}/metrics")
                res.raise_for_status()
                metrics[base_url] = res.text
            except Exception:
                metrics[base_url] = ""
        return JSONResponse(metrics)


def _start_parrot_server(config: ParrotServerConfig, event: Event) -> None:
    # Set required environment variables
    os.environ["SIMULATE_NETWORK_LATENCY_PRT"] = "0"  # 1 to enable, 0 to disable
    os.environ["SIMULATE_NETWORK_LATENCY_FS"] = "0"  # 1 to enable, 0 to disable

    _configure_parrot_logging(config.log_level, config.log_file)
    uvloop.run(_run_parrot_server(config, event))


async def _run_parrot_server(config: ParrotServerConfig, event: Event) -> None:
    from parrot.serve import http_server

    http_server.release_mode = config.release_mode
    http_server.pcore = _create_server_core(config)

    _mount_server_benchmark_endpoints(http_server.app, http_server.pcore)
    uvicorn_config = Config(
        app=http_server.app,
        host=config.host,
        port=config.port,
        log_level=config.log_level,
    )
    uvicorn_server = Server(uvicorn_config)
    await _run_server_and_backend(uvicorn_server, http_server.pcore.serve_loop(), event)


def _create_engine(config: ParrotEngineConfig) -> OpenAIEngine:
    engine_config = config.to_dict()
    if not EngineConfig.verify_config(engine_config):
        raise ValueError(f"Invalid Parrot engine config: {engine_config}")
    if config.engine_type != ENGINE_TYPE_OPENAI:
        raise ValueError(f"Unsupported engine type: {config.engine_type}")
    return OpenAIEngine(engine_config, True)


def _mount_engine_benchmark_endpoints(app: FastAPI, engine: OpenAIEngine) -> None:
    @app.get("/start_benchmark")
    def start_benchmark() -> None:
        base_url = engine.openai_config.base_url
        assert base_url is not None
        base_url = strip_v1_suffix(base_url)
        try:
            request_start_benchmark(base_url)
        except Exception:
            pass

    @app.get("/stop_benchmark")
    def stop_benchmark() -> JSONResponse:
        base_url = engine.openai_config.base_url
        assert base_url is not None
        base_url = strip_v1_suffix(base_url)
        try:
            bench_results = request_stop_benchmark(base_url)
        except Exception:
            bench_results = {}
        return JSONResponse(bench_results)

    @app.get("/metrics")
    def metrics() -> PlainTextResponse:
        base_url = engine.openai_config.base_url
        assert base_url is not None
        base_url = strip_v1_suffix(base_url)
        try:
            res = requests.get(f"{base_url}/metrics")
            res.raise_for_status()
            metrics = res.text
        except Exception:
            metrics = ""
        return PlainTextResponse(metrics)


def _start_parrot_engine(config: ParrotEngineConfig, event: Event) -> None:
    _configure_parrot_logging(config.log_level, config.log_file)
    uvloop.run(_run_parrot_engine(config, event))


async def _run_parrot_engine(config: ParrotEngineConfig, event: Event) -> None:
    from parrot.engine import http_server

    http_server.llm_engine = _create_engine(config)

    _mount_engine_benchmark_endpoints(http_server.app, http_server.llm_engine)
    uvicorn_config = Config(
        app=http_server.app,
        host=config.host,
        port=config.port,
        log_level=config.log_level,
    )
    uvicorn_server = Server(uvicorn_config)
    await _run_server_and_backend(
        uvicorn_server, http_server.llm_engine.engine_loop(), event
    )
