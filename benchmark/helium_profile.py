import asyncio
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field

from bench import (
    _get_llm_server_config,
    generate_tasks_from_system_params,
    get_prebuilt_file_path,
)
from bench_programs.debate.helium import HeliumDebateProgram
from bench_programs.iterative.helium import HeliumIterativeProgram
from bench_programs.map_reduce.helium import HeliumMapReduceProgram
from bench_programs.parallel.helium import HeliumParallelProgram
from bench_programs.reflection.helium import HeliumReflectionProgram
from bench_programs.trading.helium import HeliumTradingProgram
from bench_tasks.base import BenchmarkTask
from bench_tasks.debate import DebateBenchmarkTask
from bench_tasks.iterative import IterativeBenchmarkTask
from bench_tasks.map_reduce import MapReduceBenchmarkTask
from bench_tasks.parallel import ParallelBenchmarkTask
from bench_tasks.reflection import ReflectionBenchmarkTask
from bench_tasks.trading import TradingBenchmarkTask
from bench_utils.runner.base import BenchmarkRunner, LLMServerConfig, RunnerConfig
from bench_utils.runner.handler import RunnerHandler

from helium import helium
from helium.constants import (
    DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_NUM_SEQS,
    LLM_CONTEXT_LENGTH,
    VLLM_CACHE_CAPACITY,
)
from helium.runtime import HeliumServerConfig
from helium.runtime.cache_manager import CacheManagerConfig
from helium.runtime.llm import LLMServiceConfig, LLMServiceInfo
from helium.runtime.protocol import HeliumRequestConfig, QueryProfilingConfig

HeliumProgramTypes = (
    HeliumMapReduceProgram,
    HeliumDebateProgram,
    HeliumReflectionProgram,
    HeliumIterativeProgram,
    HeliumParallelProgram,
    HeliumTradingProgram,
)
BENCHMARK_TASK_MAP = {
    "mapreduce": MapReduceBenchmarkTask,
    "debate": DebateBenchmarkTask,
    "reflection": ReflectionBenchmarkTask,
    "iterative": IterativeBenchmarkTask,
    "parallel": ParallelBenchmarkTask,
    "trading": TradingBenchmarkTask,
}


@dataclass
class ProfilingConfig:
    system: str = "Helium"
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    cuda_device_name: str = "NVIDIA H100 NVL"

    llm_server_host: str | list[str] | None = None
    llm_server_port: int | list[int] | None = None
    num_llm_workers: int = 1
    enable_prefix_caching: bool = True
    enable_proactive_kv_cache: bool = False
    enable_prompt_cache: bool = False
    enable_cache_aware_scheduling: bool = True
    enable_runtime_adjustment: bool = False
    enable_query_profiling: bool = True
    num_gpu_blocks_override: int | None = None

    num_trials: int = 1
    verbose: bool = False
    helium_profiling: bool = True
    helium_prebuilt_file: str | None = None

    prebuilt_file_path_fmt: str = "./benchmark/prebuilt/{}/helium_{}.dill"

    workload_scales: list[int] = field(init=False)
    cache_manager_config: CacheManagerConfig = field(init=False)
    llm_server_config: LLMServerConfig = field(init=False)
    query_profiling_config: QueryProfilingConfig = field(init=False)

    def __post_init__(self):
        self.workload_scales = [1]
        self.cache_manager_config = CacheManagerConfig(
            enable_proactive_kv_cache=self.enable_proactive_kv_cache,
            enable_prompt_cache=self.enable_prompt_cache,
        )
        self.llm_server_config = _get_llm_server_config(
            self.system,
            self.model,
            self.num_llm_workers,
            self.cache_manager_config,
            None,
            None,
            self.llm_server_host,
            self.llm_server_port,
        )
        self.query_profiling_config = QueryProfilingConfig(
            only_profile=True,
            sampling_ratio=1,
            max_sampling_size=500,  # Large enough to cover most cases
        )

    @property
    def runner_config(self) -> RunnerConfig:
        return RunnerConfig(
            system=self.system,
            cuda_device_name=self.cuda_device_name,
            gpu_util_log_dir=None,
            llm_server_config=self.llm_server_config,
            enable_prefix_caching=self.enable_prefix_caching,
            num_gpu_blocks_override=self.num_gpu_blocks_override,
            verbose=self.verbose,
        )


def get_helium_server_config(profiling_config: ProfilingConfig) -> HeliumServerConfig:
    device_name = profiling_config.cuda_device_name or next(
        iter(VLLM_CACHE_CAPACITY.keys())
    )
    llm_server_config = profiling_config.llm_server_config

    if llm_server_config.model.startswith("Qwen/Qwen3-"):
        llm_server_config.extend_context_length(2.0, 32768)

    cache_capacity = (
        VLLM_CACHE_CAPACITY[device_name][llm_server_config.model]
        if profiling_config.num_gpu_blocks_override is None
        else profiling_config.num_gpu_blocks_override * 16
    )
    max_model_len = (
        min(LLM_CONTEXT_LENGTH[llm_server_config.model], cache_capacity)
        if llm_server_config.context_length is None
        else llm_server_config.context_length
    )
    return HeliumServerConfig(
        is_local=True,
        benchmarking=False,
        llm_service_configs=[
            LLMServiceConfig(
                name="vllm-local",
                args={
                    "enable_prefix_caching": profiling_config.enable_prefix_caching,
                    "num_gpu_blocks_override": profiling_config.num_gpu_blocks_override,
                    "max_num_seqs": DEFAULT_MAX_NUM_SEQS,
                    "max_num_batched_tokens": DEFAULT_MAX_NUM_BATCHED_TOKENS,
                    "log_file": log_file,
                    "max_model_len": max_model_len,
                    "disable_log_stats": False,
                    **kwargs,
                },
                info=LLMServiceInfo(
                    cache_capacity=cache_capacity,
                    max_num_reqs=DEFAULT_MAX_NUM_SEQS,
                    max_num_batched_tokens=DEFAULT_MAX_NUM_BATCHED_TOKENS,
                    alpha=1,
                    is_memory_limited=False,
                    prefix_caching_enabled=profiling_config.enable_prefix_caching,
                ),
            )
            for kwargs, log_file in zip(
                llm_server_config.kwargs_list, llm_server_config.log_files
            )
        ],
        cache_manager_config=profiling_config.cache_manager_config,
    )


async def main(
    profiling_config: ProfilingConfig, to_profile: tuple[type[BenchmarkTask], ...]
) -> None:
    verbose = profiling_config.verbose
    prebuilt_file_path_fmt = profiling_config.prebuilt_file_path_fmt

    helium_server_config = get_helium_server_config(profiling_config)
    with helium.serve_instance(config=helium_server_config):
        handler = RunnerHandler(BenchmarkRunner(profiling_config.runner_config))
        async for workload_name, bench_task in generate_tasks_from_system_params(
            workload_scales=profiling_config.workload_scales,
            model=profiling_config.model,
            num_trials=profiling_config.num_trials,
            helium_profiling=profiling_config.helium_profiling,
            enable_cache_aware_scheduling=profiling_config.enable_cache_aware_scheduling,
            enable_runtime_adjustment=profiling_config.enable_runtime_adjustment,
            enable_query_profiling=profiling_config.enable_query_profiling,
            helium_prebuilt_file=profiling_config.helium_prebuilt_file,
            runner_config=profiling_config.runner_config,
            handler=handler,
        ):
            if not isinstance(bench_task, to_profile):
                if verbose:
                    print(
                        f"Skipping {workload_name} as it is not in the profiling list."
                    )
                continue

            if verbose:
                print(f"Profiling {workload_name}...")

            _, program_list, kwargs_list = bench_task.get_run_configurations()
            assert len(program_list) == 1 and len(kwargs_list) == 1
            program = program_list[0]
            if not isinstance(program, HeliumProgramTypes):
                raise ValueError(f"Unsupported program type: {type(program)}")

            # Create benchmark agent
            kwargs = program.prepare_kwargs(**kwargs_list[0][0])
            agent = program.create_agent(**kwargs, set_cacheable=True)

            # Prepare the prebuilt file path
            prebuilt_file_path = get_prebuilt_file_path(
                prebuilt_file_path_fmt, profiling_config.model, workload_name
            )
            assert prebuilt_file_path is not None
            prebuilt_file_path.parent.mkdir(parents=False, exist_ok=True)
            if prebuilt_file_path.exists():
                resp = input(
                    f"Prebuilt file exists at '{prebuilt_file_path}'. "
                    "Do you want to overwrite it? (y/n): "
                )
                if resp.lower() != "y":
                    raise FileExistsError()

            # Profile the benchmark task
            query_profile = helium.profile(
                agent.get_and_reset_compiled_graph(),
                config=HeliumRequestConfig(
                    query_profiling_config=profiling_config.query_profiling_config
                ),
            )

            # Save the prebuilt file
            agent.save(prebuilt_file_path, query_profile)
            if verbose:
                print(f"Prebuilt file saved to '{prebuilt_file_path}'.")


def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--system", type=str, default="Helium")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--cuda_device_name", type=str, default="NVIDIA H100 NVL")
    parser.add_argument("--num-llm-workers", type=int, default=1)

    parser.add_argument("-v", "--verbose", action="store_true", default=False)

    parser.add_argument("--to-exclude", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    profiling_config = ProfilingConfig(
        system=args.system,
        model=args.model,
        cuda_device_name=args.cuda_device_name,
        num_llm_workers=args.num_llm_workers,
        verbose=args.verbose,
    )

    to_exclude_str = args.to_exclude.strip().split(",") if args.to_exclude else []
    to_exclude = set(task_name.lower() for task_name in to_exclude_str)
    to_profile = tuple(
        task_type
        for task_name, task_type in BENCHMARK_TASK_MAP.items()
        if task_name not in to_exclude
    )

    asyncio.run(main(profiling_config, to_profile))
