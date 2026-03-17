import asyncio
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from bench_utils.mixin import BenchmarkMixin
from bench_utils.runner.base import BenchmarkRunner, RunnerConfig, start_gpu_monitor

from helium import envs
from helium.common import GenerationConfig
from helium.runtime.protocol import HeliumSystemProfile
from helium.runtime.utils.sglang.config import SGLangRouterConfig, SGLangServerConfig
from helium.runtime.utils.sglang.server import (
    popen_py_router,
    popen_sglang_server,
    terminate_process_group,
    wait_for_health,
)
from helium.runtime.utils.sglang.utils import (
    async_request_abort_all,
    async_request_flush_cache,
    async_request_metrics,
    async_request_start_benchmark,
    async_request_stop_benchmark,
    async_request_update_agent_step_graph,
    get_metric_values,
    strip_v1_suffix,
)
from helium.runtime.utils.vllm.config import BenchVLLMServerConfig

SGLANG_ROUTER_POLICY: str = "round_robin"
SGLANG_ROUTER_REQUEST_TIMEOUT_SECS: int = 24 * 60 * 60
SGLANG_ROUTER_QUEUE_TIMEOUT_SECS: int = 24 * 60 * 60
SGLANG_ROUTER_MAX_CONCURRENT_REQUESTS: int = 100_000
SGLANG_ROUTER_QUEUE_SIZE: int = 100_000


def _normalize_rust_log_level(level: str | None) -> str:
    """
    Rust router uses `tracing_subscriber::EnvFilter::try_from_default_env()` first.
    Setting RUST_LOG here is the reliable way to suppress `sglang_router_rs::*` INFO logs.
    """
    if level is None:
        rust_level = "info"
    else:
        normalized = level.strip().lower()
        if normalized in {"warn", "warning"}:
            rust_level = "warn"
        elif normalized in {"error", "critical", "fatal"}:
            rust_level = "error"
        elif normalized == "debug":
            rust_level = "debug"
        elif normalized == "trace":
            rust_level = "trace"
        else:
            rust_level = "info"
    return f"sglang_router_rs={rust_level}"


def _vllm_to_sglang_server_config(
    config: BenchVLLMServerConfig, index: int | None
) -> tuple[SGLangServerConfig, dict[str, str], Path | None]:
    """Converts VLLM server config to SGLang server config

    Returns
    -------
    tuple[SGLangServerConfig, dict[str, str], Path | None]
        The SGLang server config, environment variables, and log file path.
    """
    env: dict[str, str] = {}
    base_gpu_id: int | None = None
    device: str | None = None
    extra_args: list[str] = []

    if config.device.startswith("cuda:"):
        env["CUDA_VISIBLE_DEVICES"] = config.device.removeprefix("cuda:")
        device = "cuda"
        base_gpu_id = 0
    elif config.device == "cuda":
        device = "cuda"
        base_gpu_id = 0
    else:
        device = config.device

    # Thinking mode:
    env["HELIUM_VLLM_ENABLE_THINKING"] = (
        "1" if envs.HELIUM_VLLM_ENABLE_THINKING else "0"
    )

    # Long context override:
    if config.hf_overrides is not None and "rope_parameters" in config.hf_overrides:
        env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
        extra_args.extend(
            ["--json-model-override-args", json.dumps(config.hf_overrides)]
        )

    sglang_cfg = SGLangServerConfig(
        model=config.model,
        host=config.host,
        port=config.port,
        device=device,
        base_gpu_id=base_gpu_id,
        log_level=config.log_level,
        log_level_http=config.uvicorn_log_level,
        enable_metrics=True,  # To enable benchmark metrics collection
        context_length=config.max_model_len,
        max_running_requests=config.max_num_seqs,
        chunked_prefill_size=config.max_num_batched_tokens,
        max_prefill_tokens=config.max_num_batched_tokens,
        disable_radix_cache=not config.enable_prefix_caching,
        enable_hierarchical_cache=True,
        hicache_ratio=1.001,
        hicache_write_policy="write_through",
        disable_prefetch=False,
        disable_lr_pf=True,
        disable_kv_pf=False,
        kv_pf_reserve_tokens=config.max_model_len,
        extra_args=extra_args,
    )
    if config.log_file is None:
        log_file = None
    elif index is None:
        log_file = config.log_file
    else:
        log_file = config.log_file.with_stem(f"{config.log_file.stem}_{index}")
    return sglang_cfg, env, log_file


class BenchmarkRunnerWithSGLang(BenchmarkRunner):
    def __init__(
        self,
        config: RunnerConfig,
        devices: list[int],
        main_config: BenchVLLMServerConfig,
        worker_configs: list[BenchVLLMServerConfig] | None,
        gpu_util_log_dir: Path | None = None,
    ) -> None:
        super().__init__(config)
        self.devices = devices
        self.gpu_util_log_dir = gpu_util_log_dir

        self._mode: Literal["single", "router"]
        self._server_cfg: SGLangServerConfig | None
        self._server_env: dict[str, str]
        self._server_log_file: Path | None

        self._router_cfg: SGLangRouterConfig | None
        self._router_log_file: Path | None
        self._router_env: dict[str, str]
        self._worker_cfgs: list[SGLangServerConfig]
        self._worker_envs: list[dict[str, str]]
        self._worker_log_files: list[Path | None]

        if worker_configs is None:
            server_cfg, env, log_file = _vllm_to_sglang_server_config(main_config, None)
            self._mode = "single"
            self._server_cfg = server_cfg
            self._server_env = env
            self._server_log_file = log_file
            self._router_cfg = None
            self._router_log_file = None
            self._router_env = {}
            self._worker_cfgs = []
            self._worker_envs = []
            self._worker_log_files = []
        else:
            if len(worker_configs) == 0:
                raise ValueError("Expected at least one worker config")
            model = main_config.model
            for wc in worker_configs:
                if wc.model != model:
                    raise ValueError(
                        "SGLang router mode expects all workers to serve the same model"
                    )

            self._mode = "router"
            self._server_cfg = None
            self._server_env = {}
            self._server_log_file = None

            self._router_cfg = SGLangRouterConfig(
                host=main_config.host,
                port=main_config.port,
                policy=SGLANG_ROUTER_POLICY,
                log_level=main_config.uvicorn_log_level,
                extra_args=[
                    "--request-timeout-secs",
                    str(SGLANG_ROUTER_REQUEST_TIMEOUT_SECS),
                    "--queue-timeout-secs",
                    str(SGLANG_ROUTER_QUEUE_TIMEOUT_SECS),
                    "--max-concurrent-requests",
                    str(SGLANG_ROUTER_MAX_CONCURRENT_REQUESTS),
                    "--queue-size",
                    str(SGLANG_ROUTER_QUEUE_SIZE),
                    "--rate-limit-tokens-per-second",
                    str(SGLANG_ROUTER_MAX_CONCURRENT_REQUESTS),
                ],
            )
            # Use main_config log_file as the router log if provided.
            self._router_log_file = main_config.log_file
            self._router_env = {
                "RUST_LOG": _normalize_rust_log_level(main_config.uvicorn_log_level)
            }

            self._worker_cfgs = []
            self._worker_envs = []
            self._worker_log_files = []
            for i, wc in enumerate(worker_configs):
                s_cfg, env, log_file = _vllm_to_sglang_server_config(wc, i)
                env = dict(env)
                env["SGLANG_DP_RANK"] = str(i)
                self._worker_cfgs.append(s_cfg)
                self._worker_envs.append(env)
                self._worker_log_files.append(log_file)

        self._server_proc: subprocess.Popen[bytes] | None = None
        self._router_proc: subprocess.Popen[bytes] | None = None
        self._worker_procs: list[subprocess.Popen[bytes]] = []

        self._gpu_monitor_proc = None

    @property
    def server_url(self) -> str:
        if self._mode == "single":
            assert self._server_cfg is not None
            return strip_v1_suffix(self._server_cfg.base_url)
        assert self._router_cfg is not None
        return strip_v1_suffix(self._router_cfg.base_url)

    async def init_run(self, run_name: str) -> None:
        if self.gpu_util_log_dir is not None:
            if self._gpu_monitor_proc is not None:
                raise ValueError("GPU monitor process already started")
            log_file = self.gpu_util_log_dir / f"{run_name}.log"
            self._gpu_monitor_proc = start_gpu_monitor(log_file, self.devices)

    async def clean_up_run(self) -> None:
        urls = (
            [self.server_url]
            if self._mode == "single"
            else [cfg.base_url for cfg in self._worker_cfgs]
        )
        for url in urls:
            try:
                await async_request_flush_cache(url)
                continue
            except RuntimeError as e:
                # SGLang returns 400 when there are pending/running requests.
                msg = str(e)
                if "HTTP 400 Error" not in msg or "/flush_cache" not in msg:
                    raise

            # Best-effort cleanup: abort all requests, then retry flush.
            try:
                await async_request_abort_all(url)
            except Exception:
                pass

            for _ in range(30):
                await asyncio.sleep(0.5)
                try:
                    await async_request_flush_cache(url)
                    break
                except RuntimeError as e:
                    msg = str(e)
                    if "HTTP 400 Error" not in msg or "/flush_cache" not in msg:
                        raise

        if self.gpu_util_log_dir is not None:
            self._stop_gpu_monitor()

    async def start_runner(self) -> None:
        if self._mode == "single":
            if self._server_proc is not None:
                raise ValueError("Server already started")
            assert self._server_cfg is not None
            self._server_proc = popen_sglang_server(
                self._server_cfg,
                env=self._server_env,
                log_file=self._server_log_file,
            )
            wait_for_health(self._server_cfg.host, self._server_cfg.port)
            return

        if self._router_proc is not None or self._worker_procs:
            raise ValueError("Router/workers already started")
        assert self._router_cfg is not None

        # Start workers first.
        for cfg, env, log_file in zip(
            self._worker_cfgs, self._worker_envs, self._worker_log_files
        ):
            proc = popen_sglang_server(cfg, env=env, log_file=log_file)
            self._worker_procs.append(proc)
        for cfg in self._worker_cfgs:
            wait_for_health(cfg.host, cfg.port)

        worker_urls = [strip_v1_suffix(w.base_url) for w in self._worker_cfgs]
        router_cfg = self._router_cfg.replace(worker_urls=worker_urls)
        self._router_proc = popen_py_router(
            router_cfg, env=self._router_env, log_file=self._router_log_file
        )
        wait_for_health(router_cfg.host, router_cfg.port)

    async def stop_runner(self) -> None:
        if self._mode == "single":
            if self._server_proc is None:
                raise ValueError("Server not started")
            terminate_process_group(self._server_proc)
            self._server_proc = None
            return

        if self._router_proc is None:
            raise ValueError("Router not started")

        terminate_process_group(self._router_proc)
        self._router_proc = None

        for proc in self._worker_procs:
            terminate_process_group(proc)
        self._worker_procs = []

    async def _run_bench(self, bench: BenchmarkMixin, *args, **kwargs) -> None:
        await bench.run_async(
            *args,
            **kwargs,
            start_benchmark=self._start_benchmark,
            stop_benchmark=self._stop_benchmark,
            update_agent_step_graph=self._update_agent_step_graph,
            get_worker_generation_configs=self._get_worker_generation_configs,
        )

    async def _start_benchmark(self) -> None:
        if self._mode == "single":
            assert self._server_cfg is not None
            await async_request_start_benchmark(self._server_cfg.base_url)
        else:
            for worker in self._worker_cfgs:
                await async_request_start_benchmark(worker.base_url)

    async def _stop_benchmark(self) -> HeliumSystemProfile:
        llm_benchmark: dict[str, Any] = {}
        if self._mode == "single":
            assert self._server_cfg is not None
            llm_benchmark["sglang"] = await self._stop_benchmark_worker(
                self._server_cfg
            )
        else:
            for i, worker in enumerate(self._worker_cfgs):
                llm_benchmark[f"sglang_{i}"] = await self._stop_benchmark_worker(worker)
        return HeliumSystemProfile(llm_benchmark=llm_benchmark)

    async def _stop_benchmark_worker(
        self, config: SGLangServerConfig
    ) -> dict[str, Any]:
        base_url = config.base_url
        metric_values = await async_request_stop_benchmark(base_url)
        metrics = await async_request_metrics(base_url)
        metric_values.update(get_metric_values(metrics))
        return {base_url: metric_values}

    async def _update_agent_step_graph(
        self,
        agent_data: dict[str, Any],
        timestep_data: dict[int, list[str]],
        timestep_cnt: int,
    ) -> None:
        if self._mode == "single":
            assert self._server_cfg is not None
            urls = [self._server_cfg.base_url]
        else:
            urls = [cfg.base_url for cfg in self._worker_cfgs]
        await asyncio.gather(
            *[
                async_request_update_agent_step_graph(
                    url,
                    agent_data=agent_data,
                    timestep_data=timestep_data,
                    timestep_cnt=timestep_cnt,
                )
                for url in urls
            ]
        )

    def _get_worker_generation_configs(
        self, config: GenerationConfig
    ) -> list[GenerationConfig]:
        if self._mode == "single":
            return [replace(config, base_url=self.server_url)]
        return [replace(config, base_url=cfg.base_url) for cfg in self._worker_cfgs]

    def _stop_gpu_monitor(self) -> None:
        if self._gpu_monitor_proc is None:
            raise ValueError("GPU monitor process not started")
        self._gpu_monitor_proc.terminate()
        self._gpu_monitor_proc.join()
        self._gpu_monitor_proc = None
