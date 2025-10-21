import multiprocessing as mp
from pathlib import Path

from bench_utils.runner.base import (
    BenchmarkRunner,
    RunnerConfig,
    start_gpu_monitor,
    start_llm_server,
)

from helium.runtime import HeliumServer, HeliumServerConfig
from helium.runtime.utils.vllm.config import BenchVLLMServerConfig


class BenchmarkRunnerWithLocalHelium(BenchmarkRunner):
    def __init__(
        self,
        config: RunnerConfig,
        devices: list[int],
        gpu_util_log_dir: Path | None = None,
        helium_server_config: HeliumServerConfig | None = None,
        openai_server_configs: list[BenchVLLMServerConfig] | None = None,
    ) -> None:
        super().__init__(config)
        self.devices = devices
        self.gpu_util_log_dir = gpu_util_log_dir
        self.helium_server_config = helium_server_config or HeliumServerConfig(
            is_local=True, benchmarking=True
        )
        self.openai_server_configs = openai_server_configs
        self._openai_server_procs: list[mp.Process] | None = None
        self._gpu_monitor_proc: mp.Process | None = None

    async def init_run(self, run_name: str) -> None:
        server = HeliumServer.get_instance()
        if not server.is_started:
            raise ValueError("Server has not started")

        if self.gpu_util_log_dir is not None:
            # Start GPU monitoring
            if self._gpu_monitor_proc is not None:
                raise ValueError("GPU monitor process already started")
            log_file = self.gpu_util_log_dir / f"{run_name}.log"
            self._gpu_monitor_proc = start_gpu_monitor(log_file, self.devices)

    async def clean_up_run(self) -> None:
        server = HeliumServer.get_instance()
        server.reset_prefix_cache()

        if self.gpu_util_log_dir is not None:
            if self._gpu_monitor_proc is None:
                raise ValueError("GPU monitor process not started")
            self._stop_server(self._gpu_monitor_proc)
            self._gpu_monitor_proc = None

    async def start_runner(self) -> None:
        if self.openai_server_configs is not None:
            # Start the LLM server
            if self._openai_server_procs is not None:
                raise ValueError("OpenAI server process already started")
            self._openai_server_procs = self._start_server()

        if self.helium_server_config.is_local:
            # Start the Helium server
            server = HeliumServer.get_instance(config=self.helium_server_config)
            if server.is_started:
                raise ValueError("Server already started")
            server.start()

    async def stop_runner(self) -> None:
        if self.helium_server_config.is_local:
            server = HeliumServer.get_instance(config=self.helium_server_config)
            if not server.is_started:
                raise ValueError("Server has not started")
            server.close()

        if self.openai_server_configs is not None:
            if self._openai_server_procs is None:
                raise ValueError("OpenAI server process not started")
            for proc in self._openai_server_procs:
                self._stop_server(proc)
            self._openai_server_procs = None

    async def _reset_states(self) -> None:
        server = HeliumServer.get_instance()
        if not server.is_started:
            raise ValueError("Server has not started")
        server.reset_proactive_cache()

    def _start_server(self) -> list[mp.Process]:
        worker_configs = self.openai_server_configs
        assert worker_configs is not None

        start_events = [mp.Event() for _ in worker_configs]

        server_procs: list[mp.Process] = []
        for config, event in zip(worker_configs, start_events):
            compiled_config = config.compile([])
            server_proc = mp.Process(
                target=start_llm_server, args=(compiled_config, event)
            )
            server_proc.start()
            server_procs.append(server_proc)

        for event in start_events:
            event.wait()
        return server_procs

    def _stop_server(self, server_proc: mp.Process) -> None:
        server_proc.terminate()
        server_proc.join()
