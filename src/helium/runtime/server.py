import json
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, TypeVar

from helium import envs
from helium.graphs import CompiledGraph, Graph
from helium.runtime.cache_manager import CacheManagerConfig
from helium.runtime.llm import LLMProfilingInfo, LLMServiceConfig
from helium.runtime.optimizer import HeliumOptimizer
from helium.runtime.processor import HeliumProcessor, ProcessorOutput
from helium.runtime.profiler import RequestProfiler
from helium.runtime.protocol import (
    HeliumQueryProfile,
    HeliumRequest,
    HeliumRequestConfig,
    HeliumResponse,
    PrefixMap,
    QueryProfilingConfig,
)
from helium.runtime.request import RequestHandler, RequestInfo
from helium.runtime.utils.logger import (
    Logger,
    LogLevel,
    init_child_logger,
    log_on_exception_async,
)
from helium.runtime.utils.loop import AsyncEventLoop
from helium.runtime.utils.queue import TSQueue
from helium.utils import async_runner, run_coroutine_blocking, stop_async_runner

T = TypeVar("T")


class HeliumServerError(Exception):
    pass


class HeliumServerConfig:
    DEFAULT_CONFIG_PATH = Path.cwd() / "configs" / "llm_services.json"

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        is_local: bool = False,
        benchmarking: bool = False,
        llm_service_configs: list[LLMServiceConfig] | str | Path | None = None,
        cache_manager_config: CacheManagerConfig | None = None,
        deployment_mode: Literal["dev", "prod"] = envs.HELIUM_DEPLOYMENT_MODE,
    ) -> None:
        self.is_local = is_local
        """Whether to use a local Helium server."""
        self.benchmarking = benchmarking
        """Whether to enable worker benchmarking."""
        self.llm_service_configs: list[LLMServiceConfig]
        """List of LLM service configurations."""
        self.deployment_mode: Literal["dev", "prod"] = deployment_mode
        """Deployment mode, either 'dev' or 'prod'."""
        if isinstance(llm_service_configs, list):
            self.llm_service_configs = llm_service_configs
        else:
            if llm_service_configs is None:
                llm_service_configs = self.DEFAULT_CONFIG_PATH
            with open(llm_service_configs) as f:
                data = json.load(f)
            self.llm_service_configs = [
                LLMServiceConfig.from_dict(item) for item in data
            ]

        # Replace with mock LLM services
        if envs.DEBUG_MOCK_LLM_ONLY:
            llm_service_configs = []
            for config in self.llm_service_configs:
                if config.name == "vllm-local":
                    config.args["mock"] = True
                else:
                    config = LLMServiceConfig(name="mock")
                llm_service_configs.append(config)
            self.llm_service_configs = llm_service_configs

        self.cache_manager_config = (
            CacheManagerConfig()
            if cache_manager_config is None
            else cache_manager_config
        )

        if is_local:
            self._host = self._port = None
        else:
            self._host = host or envs.HELIUM_SERVER_HOST
            self._port = port or envs.HELIUM_SERVER_PORT

    @property
    def host(self) -> str:
        """Hostname of the Helium server."""
        if self._host is None:
            raise ValueError("Host is not set")
        return self._host

    @property
    def port(self) -> int:
        """Port of the Helium server."""
        if self._port is None:
            raise ValueError("Port is not set")
        return self._port

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


class HeliumServer:
    _instance: "HeliumServer | None" = None

    def __init__(
        self,
        config: HeliumServerConfig | None = None,
        logger: Logger | None = None,
        log_level: LogLevel | None = None,
    ) -> None:
        self.logger: Logger = init_child_logger("Server", logger, log_level)
        if self.__class__._instance is not None:
            self.logger.warning(
                "Server instance already exists. "
                "You may want to use HeliumServer.get_instance() instead."
            )
        else:
            self.__class__._instance = self
        self._request_profiler = RequestProfiler(logger=self.logger)
        self.config = HeliumServerConfig() if config is None else config
        if self.config.benchmarking:
            self.logger.warning(
                "Benchmarking mode is enabled. This may affect performance."
            )
        self.processor = HeliumProcessor(
            llm_service_configs=self.config.llm_service_configs,
            cache_manager_config=self.config.cache_manager_config,
            request_profiler=self._request_profiler,
            deployment_mode=self.config.deployment_mode,
            benchmarking=self.config.benchmarking,
            logger=self.logger,
        )
        self.optimizer = HeliumOptimizer(
            puller=self.processor,
            request_profiler=self._request_profiler,
            logger=self.logger,
        )

        self._event_loop: AsyncEventLoop[RequestHandler, None, None] | None = None

    @property
    def is_started(self) -> bool:
        return self.processor.is_started

    def start(self) -> None:
        async def inner(event_loop: AsyncEventLoop) -> None:
            self.logger.info("Starting server...")
            await self.processor.start()
            await event_loop.start()
            self.logger.info("Server started.")

        if self._event_loop is not None and self._event_loop.is_started():
            self.logger.warning("Server has already been started.")
            return
        self._event_loop = AsyncEventLoop(
            handler_func=self.handle_request, in_channel=TSQueue()
        )
        async_runner().run(inner(self._event_loop))

    def close(self) -> None:
        async def inner(event_loop: AsyncEventLoop) -> None:
            self.logger.info("Terminating server...")
            await self.processor.close()
            await event_loop.stop()
            self.logger.info("Server terminated.")
            self.__class__._instance = None

        if self._event_loop is None or not self._event_loop.is_started():
            self.logger.warning("Server has not been started.")
            return
        if self._event_loop.is_stopped():
            self.logger.warning("Server is already stopped.")
            return

        async_runner().run(inner(self._event_loop))
        stop_async_runner()
        self._event_loop = None

    def __del__(self) -> None:
        if self._event_loop is not None:
            self.close()

    @log_on_exception_async()
    async def handle_request(self, request: RequestHandler, _) -> None:
        request_profiler = self._request_profiler
        request_info = self._prepare_request_info(
            request.request_id, request.query, request.config
        )
        self.logger.info("Processing request %s.", request.request_id)

        do_system_profiling = request_info.system_profiling_config is not None
        if do_system_profiling:
            # Start request profiling
            request_profiler.start()

        # 0. Validate request config
        self._validate_request(request_info)

        # 1. Perform initial query plan rewrite
        request_profiler.push_range("initial_rewrite")
        optimizer_info = self.optimizer.initial_rewrite(request_info)
        request_profiler.pop_range()

        # 2. Resolve LLM info after initial rewrite
        self.processor.resolve_llm_info(request_info)

        # 3. Precompute the query if requested
        if request_info.precompute_mode == "only":
            request_profiler.push_range("precomputation")
            output, prefix_map = await self.processor.precompute_kv_cache(request_info)
            request_profiler.pop_range()

            error_info = output.error_info

            if error_info is not None:
                self.logger.error("Error during precomputation: %s", error_info)

            response = HeliumResponse(
                outputs={
                    key: {op: [] for op in graph.graph.output_ops}
                    for key, graph in request_info.query_graphs.items()
                },
                system_profile=(
                    {"request_profile": request_profiler.stop()}
                    if do_system_profiling
                    else {}
                ),
                query_profile_map=None,
                static_prefix_map=prefix_map,
                dynamic_prefix_map=None,
                error_info=error_info,
            )

            await request.put_result(response)
            return

        # 4. Profile the query if requested
        sampled_output: ProcessorOutput | None = None
        query_profiling_config = request_info.query_profiling_config
        if query_profiling_config is not None:
            has_error = False
            if request_info.query_profile is None:
                request_profiler.push_range("query_profiling")
                sampled_output = await self.processor.profile(request_info)
                request_profiler.pop_range()
                has_error = sampled_output.has_error
            elif query_profiling_config.only_profile:
                self.logger.warning(
                    "Query profiling info is provided, but only profiling is requested. "
                    "Skipping query profiling."
                )

            if query_profiling_config.only_profile or has_error:
                # Return profiling results if only profiling is requested or if there
                # was an error

                # Post-process profiling info
                query_profile = request_info.query_profile
                assert query_profile is not None
                llm_profiling_info: dict[str, LLMProfilingInfo | None] = {
                    op_id: info
                    for op_id, info in query_profile.llm_profiling_info.items()
                    if info is not None
                }
                query_profile_map = {
                    name: HeliumQueryProfile(llm_profiling_info=remapped_info)
                    for name, remapped_info in self.optimizer.remap_ops(
                        llm_profiling_info, optimizer_info
                    ).items()
                }

                # Post-process sampled output
                if sampled_output is None:
                    outputs = {}
                    error_info = None
                else:
                    outputs = sampled_output.outputs
                    error_info = sampled_output.error_info

                if error_info is not None:
                    self.logger.error("Error during query profiling: %s", error_info)

                outputs = self.optimizer.remap_output(outputs, optimizer_info)

                response = HeliumResponse(
                    outputs=outputs,
                    system_profile=(
                        {"request_profile": request_profiler.stop()}
                        if do_system_profiling
                        else {}
                    ),
                    query_profile_map=query_profile_map,
                    error_info=error_info,
                )

                await request.put_result(response)
                return

        # 5. Perform cache-aware logical plan optimization, if applicable
        if self.processor.prompt_cache_manager is not None:
            request_profiler.push_range("cache_aware_optimization")
            # TODO: Make this request-dependent
            await self.optimizer.cache_aware_optimize(request_info)
            request_profiler.pop_range()

        # 6. Process the query
        request_profiler.push_range("query_processing")
        scheduler_output, system_profile, static_prefix_map, dynamic_prefix_map = (
            await self.processor.process(request_info)
        )
        request_profiler.pop_range()

        # Log error during scheduling, if any
        error_info = scheduler_output.error_info
        if error_info is not None:
            self.logger.error("Error during scheduling: %s", error_info)

        outputs = scheduler_output.outputs
        if sampled_output is not None:
            # 7. Combine sampled output and scheduled output
            for key, value in sampled_output.outputs.items():
                new_value = outputs[key]
                outputs[key] = value if new_value is None else value + new_value

        # 8. Remap output
        scheduler_output.outputs = self.optimizer.remap_output(outputs, optimizer_info)

        if do_system_profiling:
            # Stop request profiling
            system_profile["request_profile"] = request_profiler.stop()

        response = HeliumResponse(
            outputs=scheduler_output.outputs,
            system_profile=system_profile,
            query_profile_map=None,
            static_prefix_map=static_prefix_map,
            dynamic_prefix_map=dynamic_prefix_map,
            error_info=error_info,
        )

        await request.put_result(response)

    async def _process_request(self, handler: RequestHandler) -> HeliumResponse:
        if self._event_loop is None:
            raise RuntimeError("Server has not been started.")

        self.logger.info("Request %s received.", handler.request_id)

        start_time = time.perf_counter()
        await self._event_loop.add_event(handler)
        result = await handler.get_result()
        elapsed_time = time.perf_counter() - start_time

        self.logger.info(
            "Request %s completed (%.3f seconds).",
            handler.request_id,
            elapsed_time,
        )
        return result

    async def request(self, request: HeliumRequest) -> HeliumResponse:
        try:
            graphs = self.parse_query(request.query)
        except Exception as e:
            err_message = f"Failed to parse query: {repr(e)}"
            self.logger.exception(err_message)
            return HeliumResponse(error_info=[{"parsing": err_message}])

        return await self.execute(graphs, request.config)

    async def execute(
        self,
        graphs: dict[str, CompiledGraph],
        config: HeliumRequestConfig | None = None,
    ) -> HeliumResponse:
        handler = RequestHandler(graphs, config)
        return await self._process_request(handler)

    async def profile(
        self,
        graphs: dict[str, CompiledGraph],
        config: HeliumRequestConfig | None = None,
    ) -> HeliumResponse:
        if config is None:
            config = HeliumRequestConfig()

        if config.query_profiling_config is None:
            config.query_profiling_config = QueryProfilingConfig(only_profile=True)
        else:
            config.query_profiling_config.only_profile = True

        handler = RequestHandler(graphs, config)
        return await self._process_request(handler)

    async def precompute(
        self,
        graphs: dict[str, CompiledGraph],
        config: HeliumRequestConfig | None = None,
    ) -> HeliumResponse:
        if config is None:
            config = HeliumRequestConfig()

        config.precompute_mode = "only"
        handler = RequestHandler(graphs, config)
        return await self._process_request(handler)

    def parse_query(self, query: dict[str, dict]) -> dict[str, CompiledGraph]:
        parsed_query = {
            name: Graph.from_json(graph["graph"]).compile(**graph["inputs"])
            for name, graph in query.items()
        }
        return parsed_query

    @classmethod
    def get_instance(cls, *args, **kwargs) -> "HeliumServer":
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)
        return cls._instance

    @classmethod
    def get_started_instance(cls, *args, **kwargs) -> "HeliumServer":
        """Asynchronously gets the server instance, starts it if not already

        It does not close the server instance after the context manager exits.
        """
        server = cls.get_instance(*args, **kwargs)
        if not server.is_started:
            server.start()
        return server

    @classmethod
    @contextmanager
    def serve_instance(cls, *args, **kwargs) -> Generator["HeliumServer", None, None]:
        """Asynchronously gets the server instance, starts it if not already

        It closes the server instance after the context manager exits.
        """
        server = cls.get_started_instance(*args, **kwargs)
        try:
            yield server
        finally:
            if server.is_started:
                server.close()

    def precompute_prefixes(self, prefix_map: PrefixMap) -> None:
        output = run_coroutine_blocking(self.processor.precompute_prefixes(prefix_map))
        error_info = output.error_info
        if error_info is not None:
            raise HeliumServerError("Error during prefix precomputation", error_info)

    def reset_prefix_cache(self) -> None:
        run_coroutine_blocking(self.processor.reset_prefix_cache())

    def reset_proactive_cache(self) -> None:
        run_coroutine_blocking(self.processor.reset_proactive_cache())

    def change_llm_workers_kv_role(self, new_role: str) -> None:
        run_coroutine_blocking(self.processor.change_llm_workers_kv_role(new_role))

    def _validate_request(self, info: RequestInfo) -> None:
        if not info.enable_cache_aware_scheduling and info.enable_runtime_adjustment:
            self.logger.info(
                "Disable runtime adjustment as cache-aware scheduling is disabled."
            )
            info.enable_runtime_adjustment = False

    def _prepare_request_info(
        self,
        request_id: str,
        query_graphs: dict[str, CompiledGraph],
        config: HeliumRequestConfig,
    ) -> RequestInfo:
        query_graphs = {
            name: graph.copy(new_ids=False) for name, graph in query_graphs.items()
        }
        precompute_cacheable_inputs = (
            config.precompute_mode == "both"
            and self.processor.kv_cache_manager is not None
        )

        return RequestInfo(
            request_id=request_id,
            query_graphs=query_graphs,
            enable_cache_aware_scheduling=config.enable_cache_aware_scheduling,
            enable_runtime_adjustment=config.enable_runtime_adjustment,
            precompute_mode=config.precompute_mode,
            precompute_cacheable_inputs=precompute_cacheable_inputs,
            query_profiling_config=config.query_profiling_config,
            system_profiling_config=config.system_profiling_config,
        )
