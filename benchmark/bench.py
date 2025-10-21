import itertools
from collections import defaultdict
from collections.abc import AsyncGenerator, Callable, Iterable
from pathlib import Path

import pandas as pd
from bench_tasks.base import BenchmarkTask
from bench_tasks.debate import DebateBenchmarkConfig, DebateBenchmarkTask
from bench_tasks.iterative import IterativeBenchmarkConfig, IterativeBenchmarkTask
from bench_tasks.map_reduce import MapReduceBenchmarkConfig, MapReduceBenchmarkTask
from bench_tasks.parallel import ParallelBenchmarkConfig, ParallelBenchmarkTask
from bench_tasks.reflection import ReflectionBenchmarkConfig, ReflectionBenchmarkTask
from bench_tasks.trading import TradingBenchmarkConfig, TradingBenchmarkTask
from bench_utils.runner.base import (
    BenchmarkRunner,
    KVTransferConfig,
    LLMServerConfig,
    RunnerConfig,
)
from bench_utils.runner.handler import RunnerHandler

from helium import envs
from helium.runtime.cache_manager import CacheManagerConfig
from helium.runtime.protocol import PrefixMap

_AVAILABLE_PORTS = [4220, 6330, 8440]


def get_prebuilt_file_path(
    template: str | None, model: str, workload: str
) -> Path | None:
    model = model.rsplit("/", maxsplit=1)[-1]
    return None if template is None else Path(template.format(model, workload))


def check_benchmark_config(
    enable_cache_aware_scheduling: bool,
    enable_runtime_adjustment: bool,
    enable_query_profiling: bool,
    helium_prebuilt_file: str | None,
) -> bool:
    """Validates benchmark configuration parameters.

    Rules:
    - Query profiling requires cache-aware scheduling
    - Prebuilt files require query profiling to be enabled
    """
    if not enable_cache_aware_scheduling and enable_query_profiling:
        return False
    if not enable_cache_aware_scheduling and enable_runtime_adjustment:
        return False
    if not enable_query_profiling and (helium_prebuilt_file is not None):
        return False
    return True


def get_kv_transfer_config(enable_proactive_kv_cache: bool) -> KVTransferConfig | None:
    # if enable_proactive_kv_cache:
    #     return KVTransferConfig(
    #         kv_connector="LMCacheConnectorV1", kv_role="kv_consumer"
    #     )
    return None


def kv_transfer_config_with_id(
    config: KVTransferConfig | None, engine_id: int
) -> KVTransferConfig | None:
    if config is None:
        return None
    return config.model_copy(update={"engine_id": str(engine_id)})


def _get_server_log_files(
    log_file: Path | list[Path | None] | None, num_llm_workers: int
) -> tuple[Path | None, list[Path | None]]:
    if isinstance(log_file, list):
        if len(log_file) < num_llm_workers + 1:
            raise ValueError(
                "server_log_file must have at least num_llm_workers+1 elements."
            )
        main_log_file, *log_files = log_file[: num_llm_workers + 1]
    else:
        main_log_file = log_file
        log_files = [log_file] * num_llm_workers
    return main_log_file, log_files


def _get_llm_server_config(
    system: str,
    model: str,
    num_llm_workers: int,
    cache_manager_config: CacheManagerConfig | None,
    kv_transfer_config: KVTransferConfig | None,
    server_log_files: Path | list[Path | None] | None,
    llm_server_host: str | list[str] | None,
    llm_server_port: int | list[int] | None,
) -> LLMServerConfig:
    assert num_llm_workers > 0
    system = system.lower()
    main_log_file, server_log_files = _get_server_log_files(
        server_log_files, num_llm_workers
    )
    if system in ["helium", "helium-vllm-openai"]:
        # Create host list
        if isinstance(llm_server_host, list):
            host_list = llm_server_host[:num_llm_workers]
            if len(host_list) < num_llm_workers:
                raise ValueError(
                    "llm_server_host must have at least num_llm_workers elements."
                )
        else:
            host_list = [llm_server_host or envs.HELIUM_VLLM_HOST] * num_llm_workers
        # Create port list
        if isinstance(llm_server_port, list):
            port_list = llm_server_port[:num_llm_workers]
            if len(port_list) < num_llm_workers:
                raise ValueError(
                    "llm_server_port must have at least num_llm_workers elements."
                )
        elif num_llm_workers > 1:
            port_list = _AVAILABLE_PORTS[:num_llm_workers]
            if len(port_list) < num_llm_workers:
                raise ValueError(
                    "Not enough available ports for the number of LLM workers."
                )
        else:
            port_list = [llm_server_port or envs.HELIUM_VLLM_PORT]
        # Create device list
        device_list = (
            ["cuda"]
            if num_llm_workers == 1
            else [f"cuda:{i}" for i in range(num_llm_workers)]
        )
        return LLMServerConfig(
            model=model,
            host=host_list[0],
            port=port_list[0],
            num_llm_workers=num_llm_workers,
            main_log_file=main_log_file,
            log_files=server_log_files,
            kwargs_list=[
                {
                    "model": model,
                    "host": host,
                    "port": port,
                    "device": device,
                    "hf_overrides": None,
                    "kv_transfer_config": kv_transfer_config_with_id(
                        kv_transfer_config, i
                    ),
                }
                for i, (host, port, device) in enumerate(
                    zip(host_list, port_list, device_list)
                )
            ],
            kv_transfer_config=kv_transfer_config,
            cache_manager_config=cache_manager_config,
        )
    elif system.endswith("-openai"):
        if isinstance(llm_server_host, list):
            llm_server_host = llm_server_host[0]
        if isinstance(llm_server_port, list):
            llm_server_port = llm_server_port[0]
        # Only support one endpoint for OpenAI systems
        return LLMServerConfig(
            model=model,
            host=llm_server_host or envs.HELIUM_VLLM_HOST,
            port=llm_server_port or envs.HELIUM_VLLM_PORT,
            num_llm_workers=num_llm_workers,
            main_log_file=main_log_file,
            log_files=[],
            kwargs_list=[],
            cache_manager_config=None,
            kv_transfer_config=None,
        )
    else:
        # Create host list
        if isinstance(llm_server_host, list):
            main_host = llm_server_host[0]
            host_list = llm_server_host[1 : num_llm_workers + 1]
            if len(host_list) < num_llm_workers:
                raise ValueError(
                    "llm_server_host must have more than num_llm_workers elements."
                )
        else:
            main_host = llm_server_host or envs.HELIUM_VLLM_HOST
            host_list = [main_host] * num_llm_workers
        # Create port list
        if isinstance(llm_server_port, list):
            main_port = llm_server_port[0]
            port_list = llm_server_port[1 : num_llm_workers + 1]
            if len(port_list) < num_llm_workers:
                raise ValueError(
                    "llm_server_port must have more than num_llm_workers elements."
                )
        else:
            main_port = _AVAILABLE_PORTS[0]
            if system == "parrot" or num_llm_workers > 1:
                # Parrot always uses one port for its main server
                port_list = _AVAILABLE_PORTS[1 : num_llm_workers + 1]
            else:
                port_list = [main_port]
            if len(port_list) < num_llm_workers:
                raise ValueError(
                    "Not enough available ports for the number of LLM workers."
                )
        # Create device list
        device_list = (
            ["cuda"]
            if num_llm_workers == 1
            else [f"cuda:{i}" for i in range(num_llm_workers)]
        )
        return LLMServerConfig(
            model=model,
            host=main_host,
            port=main_port,
            num_llm_workers=num_llm_workers,
            main_log_file=main_log_file,
            log_files=server_log_files,
            kwargs_list=[
                {
                    "model": model,
                    "host": host,
                    "port": port,
                    "device": device,
                    "hf_overrides": None,
                    "kv_transfer_config": None,
                }
                for host, port, device in zip(host_list, port_list, device_list)
            ],
            cache_manager_config=None,
            kv_transfer_config=None,
        )


def _get_prefix_map(
    static_map: PrefixMap | None,
    dynamic_map: dict[int, PrefixMap] | None,
    input_size: int,
) -> PrefixMap | None:
    if static_map is None and dynamic_map is None:
        return None
    prefix_map: PrefixMap = defaultdict(list)
    if static_map is not None:
        for k, v in static_map.items():
            prefix_map[k].extend(v)
    if dynamic_map is not None:
        for i, prefixes in dynamic_map.items():
            if i < 0 or i >= input_size:
                continue
            for k, v in prefixes.items():
                prefix_map[k].extend(v)
    return dict(prefix_map)


async def generate_tasks_from_system_params(
    workload_scales: Iterable[int],
    model: str,
    num_trials: int,
    helium_profiling: bool,
    enable_cache_aware_scheduling: bool,
    enable_runtime_adjustment: bool,
    enable_query_profiling: bool,
    helium_prebuilt_file: str | None,
    runner_config: RunnerConfig,
    handler: RunnerHandler,
) -> AsyncGenerator[tuple[str, BenchmarkTask]]:
    def get_helium_prebuilt_file(workload_name: str) -> Path | None:
        return get_prebuilt_file_path(helium_prebuilt_file, model, workload_name)

    cache_manager_config = runner_config.llm_server_config.cache_manager_config
    if cache_manager_config is None:
        enable_proactive_kv_cache = enable_prompt_cache = False
    else:
        enable_proactive_kv_cache = cache_manager_config.enable_proactive_kv_cache
        enable_prompt_cache = cache_manager_config.enable_prompt_cache

    async def maybe_precompute(
        get_precompute_task: Callable[[int, BenchmarkRunner], BenchmarkTask],
        get_bench_task: Callable[[int, BenchmarkRunner], BenchmarkTask],
        cacheable: bool = False,
    ) -> AsyncGenerator[BenchmarkTask]:
        async with handler.task_context() as runner:
            if not (enable_proactive_kv_cache or enable_prompt_cache):
                # No precomputation needed
                for workload_scale in workload_scales:
                    yield get_bench_task(workload_scale, runner)
            else:
                # Precompute for the largest workload scale
                if enable_proactive_kv_cache and enable_prompt_cache:
                    precompute_mode = "both"
                elif enable_proactive_kv_cache:
                    precompute_mode = "both" if cacheable else "only"
                else:
                    precompute_mode = "none"
                workload_scale = max(workload_scales)
                bench_task = get_precompute_task(workload_scale, runner)
                static_map, dynamic_map = await bench_task.precompute(precompute_mode)
                # Yield the tasks
                for workload_scale in workload_scales:
                    task = get_bench_task(workload_scale, runner)
                    prefix_map = _get_prefix_map(
                        static_map, dynamic_map, task.config.input_size
                    )
                    if prefix_map is not None:
                        task.precompute_prefixes(prefix_map)
                    yield task

    if not check_benchmark_config(
        enable_cache_aware_scheduling,
        enable_runtime_adjustment,
        enable_query_profiling,
        helium_prebuilt_file,
    ):
        return

    get_bench_task: Callable[[int, BenchmarkRunner], BenchmarkTask]

    if not enable_prompt_cache:
        mapreduce_benchmark_settings = [
            # ("mmlu", 14, False, 50, None, "mapreduce_mmlu"),
            ("mmlu", 14, True, 50, None, "mapreduce_mmlu_roles"),
            # ("tatqa", 7, False, 25, 6, "mapreduce_tatqa"),
            ("tatqa", 7, True, 25, 6, "mapreduce_tatqa_roles"),
        ]
        for (
            dataset_name,
            num_agents,
            different_roles,
            base_num_contexts,
            num_questions_per_context,
            workload_name,
        ) in mapreduce_benchmark_settings:
            get_bench_task = lambda workload_scale, runner: (  # noqa: E731
                MapReduceBenchmarkTask(
                    MapReduceBenchmarkConfig(
                        num_trials=num_trials,
                        helium_profiling=helium_profiling,
                        enable_cache_aware_scheduling=enable_cache_aware_scheduling,
                        enable_runtime_adjustment=enable_runtime_adjustment,
                        enable_query_profiling=enable_query_profiling,
                        helium_prebuilt_file=get_helium_prebuilt_file(workload_name),
                        runner_config=runner.config,
                        dataset_name=dataset_name,
                        num_agents=num_agents,
                        num_contexts=base_num_contexts * workload_scale,
                        num_questions_per_context=num_questions_per_context,
                        different_roles=different_roles,
                    ),
                    runner=runner,
                )
            )
            async for task in maybe_precompute(get_bench_task, get_bench_task):
                yield workload_name, task

    if not enable_prompt_cache:
        debate_benchmark_settings = [
            # ("mmlu", 50, None, False, "multiagent_debate_mmlu"),
            ("mmlu", 50, None, True, "multiagent_debate_mmlu_roles"),
            # ("tatqa", 25, 6, False, "multiagent_debate_tatqa"),
            ("tatqa", 25, 6, True, "multiagent_debate_tatqa_roles"),
        ]
        for (
            dataset_name,
            base_num_contexts,
            num_questions_per_context,
            different_roles,
            workload_name,
        ) in debate_benchmark_settings:
            get_bench_task = lambda workload_scale, runner: (  # noqa: E731
                DebateBenchmarkTask(
                    DebateBenchmarkConfig(
                        num_trials=num_trials,
                        helium_profiling=helium_profiling,
                        enable_cache_aware_scheduling=enable_cache_aware_scheduling,
                        enable_runtime_adjustment=enable_runtime_adjustment,
                        enable_query_profiling=enable_query_profiling,
                        helium_prebuilt_file=get_helium_prebuilt_file(workload_name),
                        runner_config=runner.config,
                        dataset_name=dataset_name,
                        num_agents=3,
                        num_rounds=2,
                        num_contexts=base_num_contexts * workload_scale,
                        num_questions_per_context=num_questions_per_context,
                        different_roles=different_roles,
                        dump_conversations=False,
                    ),
                    runner=runner,
                )
            )
            async for task in maybe_precompute(get_bench_task, get_bench_task):
                yield workload_name, task

    if not enable_prompt_cache:
        reflection_benchmark_settings = [
            ("finqa", 100, 1, "reflection_finqa"),
            ("tatqa", 50, 6, "reflection_tatqa"),
        ]
        for (
            dataset_name,
            base_num_contexts,
            num_questions_per_context,
            workload_name,
        ) in reflection_benchmark_settings:
            get_bench_task = lambda workload_scale, runner: (  # noqa: E731
                ReflectionBenchmarkTask(
                    ReflectionBenchmarkConfig(
                        num_trials=num_trials,
                        helium_profiling=helium_profiling,
                        enable_cache_aware_scheduling=enable_cache_aware_scheduling,
                        enable_runtime_adjustment=enable_runtime_adjustment,
                        enable_query_profiling=enable_query_profiling,
                        helium_prebuilt_file=get_helium_prebuilt_file(workload_name),
                        runner_config=runner.config,
                        dataset_name=dataset_name,
                        num_contexts=base_num_contexts * workload_scale,
                        num_questions_per_context=num_questions_per_context,
                    ),
                    runner=runner,
                )
            )
            async for task in maybe_precompute(get_bench_task, get_bench_task):
                yield workload_name, task

    if not enable_prompt_cache:
        iterative_benchmark_settings = [
            ("arxiv", 50, 6, 1, "iterative_arxiv"),
            ("amazon", 50, 6, 10, "iterative_amazon"),
        ]
        for (
            dataset_name,
            base_num_items,
            num_review_chunks_per_item,
            num_reviews_per_chunk,
            workload_name,
        ) in iterative_benchmark_settings:
            get_bench_task = lambda workload_scale, runner: (  # noqa: E731
                IterativeBenchmarkTask(
                    IterativeBenchmarkConfig(
                        num_trials=num_trials,
                        helium_profiling=helium_profiling,
                        enable_cache_aware_scheduling=enable_cache_aware_scheduling,
                        enable_runtime_adjustment=enable_runtime_adjustment,
                        enable_query_profiling=enable_query_profiling,
                        helium_prebuilt_file=get_helium_prebuilt_file(workload_name),
                        runner_config=runner.config,
                        dataset_name=dataset_name,
                        num_items=base_num_items * workload_scale,
                        num_review_chunks_per_item=num_review_chunks_per_item,
                        num_reviews_per_chunk=num_reviews_per_chunk,
                    ),
                    runner=runner,
                )
            )
            async for task in maybe_precompute(get_bench_task, get_bench_task):
                yield workload_name, task

    if not enable_prompt_cache:
        parallel_benchmark_settings = [("amazon", 25, 6, 10, "parallel_amazon")]
        for (
            dataset_name,
            base_num_items,
            num_review_chunks_per_item,
            num_reviews_per_chunk,
            workload_name,
        ) in parallel_benchmark_settings:
            get_bench_task = lambda workload_scale, runner: (  # noqa: E731
                ParallelBenchmarkTask(
                    ParallelBenchmarkConfig(
                        num_trials=num_trials,
                        helium_profiling=helium_profiling,
                        enable_cache_aware_scheduling=enable_cache_aware_scheduling,
                        enable_runtime_adjustment=enable_runtime_adjustment,
                        enable_query_profiling=enable_query_profiling,
                        helium_prebuilt_file=get_helium_prebuilt_file(workload_name),
                        runner_config=runner.config,
                        dataset_name=dataset_name,
                        num_experts=7,
                        num_items=base_num_items * workload_scale,
                        num_review_chunks_per_item=num_review_chunks_per_item,
                        num_reviews_per_chunk=num_reviews_per_chunk,
                    ),
                    runner=runner,
                )
            )
            async for task in maybe_precompute(get_bench_task, get_bench_task):
                yield workload_name, task

    trading_benchmark_settings = [("fin_data", 8, 5, 50, 3, 30, "trading_findata")]
    for (
        dataset_name,
        num_stocks,
        num_news_chunks,
        max_news,
        num_social_chunks,
        max_social_posts,
        workload_name,
    ) in trading_benchmark_settings:
        get_precompute_task = lambda workload_scale, runner: (  # noqa: E731
            TradingBenchmarkTask(
                TradingBenchmarkConfig(
                    num_trials=num_trials,
                    helium_profiling=helium_profiling,
                    enable_cache_aware_scheduling=enable_query_profiling,
                    enable_runtime_adjustment=False,
                    enable_query_profiling=enable_query_profiling,
                    helium_prebuilt_file=get_helium_prebuilt_file(workload_name),
                    runner_config=runner.config,
                    dataset_name=dataset_name,
                    num_stocks=num_stocks * (2 ** (workload_scale - 1)),
                    num_news_chunks=num_news_chunks,
                    max_news=max_news,
                    num_social_chunks=num_social_chunks,
                    max_social_posts=max_social_posts,
                    num_debate_rounds=2,
                    split="2024-06-01",
                ),
                runner=runner,
            )
        )
        get_bench_task = lambda workload_scale, runner: (  # noqa: E731
            TradingBenchmarkTask(
                TradingBenchmarkConfig(
                    num_trials=num_trials,
                    helium_profiling=helium_profiling,
                    enable_cache_aware_scheduling=enable_cache_aware_scheduling,
                    enable_runtime_adjustment=enable_runtime_adjustment,
                    enable_query_profiling=enable_query_profiling,
                    helium_prebuilt_file=get_helium_prebuilt_file(workload_name),
                    runner_config=runner.config,
                    dataset_name=dataset_name,
                    num_stocks=num_stocks * (2 ** (workload_scale - 1)),
                    num_news_chunks=num_news_chunks,
                    max_news=max_news,
                    num_social_chunks=num_social_chunks,
                    max_social_posts=max_social_posts,
                    num_debate_rounds=2,
                    split="2024-06-02",
                ),
                runner=runner,
            )
        )
        async for task in maybe_precompute(
            get_precompute_task, get_bench_task, cacheable=False
        ):
            yield workload_name, task


async def generate_tasks(
    num_trials: int,
    max_workload_scale: int,
    cuda_device_name: str | None,
    llm_server_hosts: str | list[str] | None,
    llm_server_ports: int | list[int] | None,
    server_log_files: Path | list[Path | None] | None,
    gpu_util_log_dir: Path | None,
    verbose: bool,
    helium_profiling: bool,
) -> AsyncGenerator[BenchmarkTask]:
    """
    Lazily yield benchmark tasks.
    This function generates tasks for each system and for each configuration (with or without prefix caching).
    """

    columns = [
        "model",
        "system",
        "num_llm_workers",
        "enable_prefix_caching",
        "enable_proactive_kv_cache",
        "enable_prompt_cache",
        "enable_cache_aware_scheduling",
        "enable_runtime_adjustment",
        "enable_query_profiling",
        "helium_prebuilt_file",
        "num_gpu_blocks_override",
    ]

    model_list = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-32B",
    ]

    configs_df = pd.DataFrame()
    for model in model_list:
        querywise = pd.DataFrame(
            list(
                itertools.product(
                    [model],
                    ["QueryWise"],
                    [1, 2],
                    [True],
                    [False],
                    [False],
                    [False],
                    [False],
                    [False],
                    [None],
                    [None],
                )
            ),
            columns=columns,
            dtype=object,
        )

        opwise = pd.DataFrame(
            list(
                itertools.product(
                    [model],
                    ["OpWise"],
                    [1, 2],
                    [True],
                    [False],
                    [False],
                    [False],
                    [False],
                    [False],
                    [None],
                    [None],
                )
            ),
            columns=columns,
            dtype=object,
        )

        langgraph = pd.DataFrame(
            list(
                itertools.product(
                    [model],
                    ["LangGraph"],
                    [1, 2],
                    [True],
                    [False],
                    [False],
                    [False],
                    [False],
                    [False],
                    [None],
                    [None],
                )
            ),
            columns=columns,
            dtype=object,
        )

        agentscope = pd.DataFrame(
            list(
                itertools.product(
                    [model],
                    ["AgentScope"],
                    [1, 2],
                    [True],
                    [False],
                    [False],
                    [False],
                    [False],
                    [False],
                    [None],
                    [None],
                )
            ),
            columns=columns,
            dtype=object,
        )

        parrot = pd.DataFrame(
            list(
                itertools.product(
                    [model],
                    ["Parrot"],
                    [1, 2],
                    [True],
                    [False],
                    [False],
                    [False],
                    [False],
                    [False],
                    [None],
                    [None],
                )
            ),
            columns=columns,
            dtype=object,
        )

        helium = pd.DataFrame(
            list(
                itertools.product(
                    # model
                    [model],
                    # system
                    ["Helium"],
                    # num_llm_workers
                    [1, 2],
                    # enable_prefix_caching
                    [True],
                    # enable_proactive_kv_cache
                    [False, True],
                    # enable_prompt_cache
                    [False, True],
                    # enable_cache_aware_scheduling
                    [False, True],
                    # enable_runtime_adjustment
                    [False, True],
                    # enable_query_profiling
                    [False, True],
                    # helium_prebuilt_file
                    [None, "./benchmark/prebuilt/{}/helium_{}.dill"],
                    # num_gpu_blocks_override
                    [None],
                )
            ),
            columns=columns,
            dtype=object,
        )

        # Concatenate both DataFrames.
        configs_df = pd.concat(
            [
                configs_df,
                helium,
                querywise,
                opwise,
                langgraph,
                agentscope,
                parrot,
            ],
            ignore_index=True,
        )

    # Iterate over the configurations to yield tasks.
    for _, config in configs_df.iterrows():
        model = config["model"]
        system = config["system"]
        num_llm_workers = config["num_llm_workers"]
        enable_proactive_kv_cache = config["enable_proactive_kv_cache"]
        cache_manager_config = CacheManagerConfig(
            enable_proactive_kv_cache=enable_proactive_kv_cache,
            kv_cache_config_file=envs.LMCACHE_CONFIG_FILE,
            enable_prompt_cache=config["enable_prompt_cache"],
        )
        llm_server_config = _get_llm_server_config(
            system=system,
            model=model,
            num_llm_workers=num_llm_workers,
            cache_manager_config=cache_manager_config,
            kv_transfer_config=get_kv_transfer_config(enable_proactive_kv_cache),
            server_log_files=server_log_files,
            llm_server_host=llm_server_hosts,
            llm_server_port=llm_server_ports,
        )
        runner_config = RunnerConfig(
            system=system,
            cuda_device_name=cuda_device_name,
            gpu_util_log_dir=gpu_util_log_dir,
            llm_server_config=llm_server_config,
            enable_prefix_caching=config["enable_prefix_caching"],
            num_gpu_blocks_override=config["num_gpu_blocks_override"],
            verbose=verbose,
        )
        async with RunnerHandler.build_from_config(runner_config) as handler:
            async for _, task in generate_tasks_from_system_params(
                workload_scales=range(1, max_workload_scale + 1),
                model=model,
                num_trials=num_trials,
                helium_profiling=helium_profiling,
                enable_cache_aware_scheduling=config["enable_cache_aware_scheduling"],
                enable_runtime_adjustment=config["enable_runtime_adjustment"],
                enable_query_profiling=config["enable_query_profiling"],
                helium_prebuilt_file=config["helium_prebuilt_file"],
                runner_config=runner_config,
                handler=handler,
            ):
                yield task
