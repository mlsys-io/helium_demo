from contextlib import asynccontextmanager
from pathlib import Path

from bench_utils.runner.base import BenchmarkRunner, EngineClientInfo, RunnerConfig
from bench_utils.runner.local import BenchmarkRunnerWithLocalHelium
from bench_utils.runner.parrot import BenchmarkRunnerWithParrot, ParrotServerConfig
from bench_utils.runner.vllm import BenchmarkRunnerWithVLLM, BenchVLLMServerConfig

from helium.constants import (
    DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_NUM_SEQS,
    LLM_CONTEXT_LENGTH,
    VLLM_CACHE_CAPACITY,
)
from helium.runtime import HeliumServerConfig
from helium.runtime.llm import LLMServiceConfig, LLMServiceInfo


def _get_lmcache_config_file(
    base_config_file: Path | None, engine_idx: int
) -> Path | None:
    if base_config_file is None:
        return None
    return base_config_file.with_stem(f"{base_config_file.stem}_{engine_idx}")


def _get_runner(config: RunnerConfig) -> BenchmarkRunner:
    llm_server_config = config.llm_server_config

    model_name = llm_server_config.model
    system = config.system

    system_name = system.lower()
    device_name = config.cuda_device_name or next(iter(VLLM_CACHE_CAPACITY.keys()))
    cache_capacity = (
        VLLM_CACHE_CAPACITY[device_name][model_name]
        if config.num_gpu_blocks_override is None
        else config.num_gpu_blocks_override * 16
    )
    max_model_len = (
        min(LLM_CONTEXT_LENGTH[model_name], cache_capacity)
        if llm_server_config.context_length is None
        else llm_server_config.context_length
    )
    cache_manager_config = llm_server_config.cache_manager_config
    if cache_manager_config is None:
        base_lmcache_config = None
    else:
        base_lmcache_config = (
            None
            if cache_manager_config.kv_cache_config_file is None
            else Path(cache_manager_config.kv_cache_config_file)
        )
    if system_name in [
        "querywise",
        "opwise",
        "autogen",
        "langgraph",
        "agentscope",
    ]:
        main_config = BenchVLLMServerConfig(
            model=model_name,
            host=llm_server_config.host,
            port=llm_server_config.port,
            log_file=llm_server_config.log_files[0],
            enable_prefix_caching=config.enable_prefix_caching,
            kv_transfer_config=llm_server_config.kv_transfer_config,
            lmcache_config_file=_get_lmcache_config_file(base_lmcache_config, 0),
            num_gpu_blocks_override=config.num_gpu_blocks_override,
            max_num_seqs=DEFAULT_MAX_NUM_SEQS,
            max_num_batched_tokens=DEFAULT_MAX_NUM_BATCHED_TOKENS,
            max_model_len=max_model_len,
            hf_overrides=llm_server_config.hf_overrides,
            benchmarking=True,
        )
        worker_configs = [
            BenchVLLMServerConfig(
                **kwargs,
                lmcache_config_file=_get_lmcache_config_file(base_lmcache_config, i),
                log_file=log_file,
                enable_prefix_caching=config.enable_prefix_caching,
                num_gpu_blocks_override=config.num_gpu_blocks_override,
                max_num_seqs=DEFAULT_MAX_NUM_SEQS,
                max_num_batched_tokens=DEFAULT_MAX_NUM_BATCHED_TOKENS,
                max_model_len=max_model_len,
                benchmarking=True,
            )
            for i, (kwargs, log_file) in enumerate(
                zip(llm_server_config.kwargs_list, llm_server_config.log_files)
            )
        ]
        engine_infos = [EngineClientInfo(cache_capacity=cache_capacity)] * len(
            worker_configs
        )
        return BenchmarkRunnerWithVLLM(
            config=config,
            devices=config.devices,
            main_config=main_config,
            worker_configs=None if len(worker_configs) == 1 else worker_configs,
            engine_infos=engine_infos,
            gpu_util_log_dir=config.gpu_util_log_dir,
        )
    elif system_name == "parrot":
        worker_configs = [
            BenchVLLMServerConfig(
                **kwargs,
                lmcache_config_file=_get_lmcache_config_file(base_lmcache_config, i),
                log_file=log_file,
                enable_prefix_caching=config.enable_prefix_caching,
                num_gpu_blocks_override=config.num_gpu_blocks_override,
                max_num_seqs=DEFAULT_MAX_NUM_SEQS,
                max_num_batched_tokens=DEFAULT_MAX_NUM_BATCHED_TOKENS,
                max_model_len=max_model_len,
                benchmarking=True,
            )
            for i, (kwargs, log_file) in enumerate(
                zip(llm_server_config.kwargs_list, llm_server_config.log_files)
            )
        ]
        engine_info = EngineClientInfo(cache_capacity=cache_capacity)
        return BenchmarkRunnerWithParrot(
            config=config,
            devices=config.devices,
            parrot_config=ParrotServerConfig(
                host=llm_server_config.host,
                port=llm_server_config.port,
                engine_info=engine_info,
                log_file=llm_server_config.main_log_file,
            ),
            main_config=BenchVLLMServerConfig(
                model=model_name,
                host=llm_server_config.host,
                port=llm_server_config.port + 1,
                log_file=llm_server_config.log_files[0],
                enable_prefix_caching=config.enable_prefix_caching,
                kv_transfer_config=llm_server_config.kv_transfer_config,
                lmcache_config_file=_get_lmcache_config_file(base_lmcache_config, 0),
                num_gpu_blocks_override=config.num_gpu_blocks_override,
                max_num_seqs=DEFAULT_MAX_NUM_SEQS,
                max_num_batched_tokens=DEFAULT_MAX_NUM_BATCHED_TOKENS,
                max_model_len=max_model_len,
                hf_overrides=llm_server_config.hf_overrides,
                benchmarking=True,
            ),
            worker_configs=None if len(worker_configs) == 1 else worker_configs,
            engine_infos=[engine_info] * len(worker_configs),
            gpu_util_log_dir=config.gpu_util_log_dir,
        )
    elif system_name == "helium":
        helium_server_config = HeliumServerConfig(
            is_local=True,
            benchmarking=True,
            llm_service_configs=[
                LLMServiceConfig(
                    name="vllm-local",
                    args={
                        "enable_prefix_caching": config.enable_prefix_caching,
                        "num_gpu_blocks_override": config.num_gpu_blocks_override,
                        "max_num_seqs": DEFAULT_MAX_NUM_SEQS,
                        "max_num_batched_tokens": DEFAULT_MAX_NUM_BATCHED_TOKENS,
                        "log_file": log_file,
                        "max_model_len": max_model_len,
                        "lmcache_config_file": _get_lmcache_config_file(
                            base_lmcache_config, i
                        ),
                        **kwargs,
                    },
                    info=LLMServiceInfo(
                        cache_capacity=cache_capacity,
                        max_num_reqs=DEFAULT_MAX_NUM_SEQS,
                        max_num_batched_tokens=DEFAULT_MAX_NUM_BATCHED_TOKENS,
                        alpha=1,
                        is_memory_limited=cache_capacity <= 1024,
                        prefix_caching_enabled=config.enable_prefix_caching,
                    ),
                )
                for i, (kwargs, log_file) in enumerate(
                    zip(llm_server_config.kwargs_list, llm_server_config.log_files)
                )
            ],
            cache_manager_config=llm_server_config.cache_manager_config,
        )
        return BenchmarkRunnerWithLocalHelium(
            config=config,
            devices=config.devices,
            gpu_util_log_dir=config.gpu_util_log_dir,
            helium_server_config=helium_server_config,
        )
    elif system_name == "helium-vllm-openai":
        openai_server_configs = [
            BenchVLLMServerConfig(
                **kwargs,
                lmcache_config_file=_get_lmcache_config_file(base_lmcache_config, i),
                log_file=log_file,
                enable_prefix_caching=config.enable_prefix_caching,
                num_gpu_blocks_override=config.num_gpu_blocks_override,
                max_num_seqs=DEFAULT_MAX_NUM_SEQS,
                max_num_batched_tokens=DEFAULT_MAX_NUM_BATCHED_TOKENS,
                max_model_len=max_model_len,
                benchmarking=True,
            )
            for i, (kwargs, log_file) in enumerate(
                zip(llm_server_config.kwargs_list, llm_server_config.log_files)
            )
        ]
        llm_service_info = LLMServiceInfo(
            cache_capacity=cache_capacity,
            max_num_reqs=DEFAULT_MAX_NUM_SEQS,
            max_num_batched_tokens=DEFAULT_MAX_NUM_BATCHED_TOKENS,
            alpha=1,
            is_memory_limited=cache_capacity <= 1024,
            prefix_caching_enabled=config.enable_prefix_caching,
        )
        helium_server_config = HeliumServerConfig(
            is_local=True,
            benchmarking=True,
            llm_service_configs=[
                LLMServiceConfig(
                    name="vllm-openai",
                    info=llm_service_info,
                    args={"api_key": "EMPTY", **kwargs},
                )
                for kwargs in llm_server_config.kwargs_list
            ],
            cache_manager_config=llm_server_config.cache_manager_config,
        )
        return BenchmarkRunnerWithLocalHelium(
            config=config,
            devices=config.devices,
            gpu_util_log_dir=config.gpu_util_log_dir,
            helium_server_config=helium_server_config,
            openai_server_configs=openai_server_configs,
        )
    elif system_name.endswith("-openai"):
        return BenchmarkRunner(config=config)
    else:
        raise ValueError(f"Unsupported system: {config.system}")


class RunnerHandler:
    def __init__(self, runner: BenchmarkRunner) -> None:
        self._runner = runner

    @asynccontextmanager
    async def task_context(self):
        async with self._runner.task_context() as runner:
            yield runner

    @classmethod
    @asynccontextmanager
    async def build_from_config(cls, config: RunnerConfig):
        runner = _get_runner(config)
        try:
            await runner.start_runner()
            yield cls(runner)
        finally:
            await runner.stop_runner()
