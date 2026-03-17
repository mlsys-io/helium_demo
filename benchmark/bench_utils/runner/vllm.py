import multiprocessing as mp
from pathlib import Path

from bench_utils.runner.base import (
    BenchmarkRunner,
    EngineClientInfo,
    RunnerConfig,
    start_gpu_monitor,
    start_llm_server,
    start_llm_server_with_controller,
)

from helium.runtime.utils.vllm.config import BenchVLLMServerConfig
from helium.runtime.utils.vllm.utils import (
    async_request_reset_prefix_cache,
    strip_v1_suffix,
)


class BenchmarkRunnerWithVLLM(BenchmarkRunner):
    def __init__(
        self,
        config: RunnerConfig,
        devices: list[int],
        main_config: BenchVLLMServerConfig,
        worker_configs: list[BenchVLLMServerConfig] | None,
        engine_infos: list[EngineClientInfo] | None = None,
        gpu_util_log_dir: Path | None = None,
    ) -> None:
        super().__init__(config)
        self.devices = devices
        self.gpu_util_log_dir = gpu_util_log_dir
        self.main_config = main_config
        self.worker_configs = worker_configs
        self.engine_infos = engine_infos
        self._server_proc: mp.Process | None = None
        self._gpu_monitor_proc: mp.Process | None = None

    @property
    def server_url(self) -> str:
        return strip_v1_suffix(self.main_config.base_url)

    async def init_run(self, run_name: str) -> None:
        if self.gpu_util_log_dir is not None:
            # Start GPU monitoring
            if self._gpu_monitor_proc is not None:
                raise ValueError("GPU monitor process already started")
            log_file = self.gpu_util_log_dir / f"{run_name}.log"
            self._gpu_monitor_proc = start_gpu_monitor(log_file, self.devices)

    async def clean_up_run(self) -> None:
        if self._server_proc is None:
            raise ValueError("Server process not started")

        # Reset the prefix cache
        await async_request_reset_prefix_cache(self.server_url)

        if self.gpu_util_log_dir is not None:
            self._stop_gpu_monitor()

    async def start_runner(self) -> None:
        if self._server_proc is not None:
            raise ValueError("Server process already started")
        self._server_proc = self._start_server()

    async def stop_runner(self) -> None:
        if self._server_proc is None:
            raise ValueError("Server process not started")
        self._stop_server(self._server_proc)
        self._server_proc = None

    def _start_server(self) -> mp.Process:
        main_config = self.main_config.compile([])
        engine_infos = self.engine_infos
        start_event = mp.Event()

        if self.worker_configs is None:
            # Single LLM worker
            server_proc = mp.Process(
                target=start_llm_server, args=(main_config, start_event)
            )
        else:
            # Multiple LLM workers with a controller
            if engine_infos is None:
                raise ValueError("Expected engine infos")
            worker_configs = [config.compile([]) for config in self.worker_configs]
            server_proc = mp.Process(
                target=start_llm_server_with_controller,
                args=(main_config, worker_configs, engine_infos, start_event),
            )

        server_proc.start()
        start_event.wait()
        return server_proc

    def _stop_server(self, server_proc: mp.Process) -> None:
        server_proc.terminate()
        server_proc.join()

    def _stop_gpu_monitor(self) -> None:
        if self._gpu_monitor_proc is None:
            raise ValueError("GPU monitor process not started")
        self._gpu_monitor_proc.terminate()
        self._gpu_monitor_proc.join()
        self._gpu_monitor_proc = None
