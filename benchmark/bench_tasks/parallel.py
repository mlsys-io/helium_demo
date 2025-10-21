from pathlib import Path
from typing import Any

from bench_programs.parallel.agentscope import ASParallelProgram
from bench_programs.parallel.autogen import AutoGenParallelProgram
from bench_programs.parallel.helium import HeliumParallelProgram, ParallelAgent
from bench_programs.parallel.langgraph import LangGraphParallelProgram
from bench_programs.parallel.opwise import OpWiseParallelProgram
from bench_programs.parallel.parrot import ParrotParallelProgram
from bench_programs.parallel.querywise import QueryWiseParallelProgram
from bench_tasks.base import BenchmarkConfig, BenchmarkTask
from bench_utils.datasets.amazon import AmazonReviewsDataset
from bench_utils.runner.base import BenchmarkRunner, RunnerConfig

from helium.common import GenerationConfig
from helium.runtime import HeliumServerConfig
from helium.runtime.protocol import (
    HeliumRequestConfig,
    QueryProfilingConfig,
    SystemProfilingConfig,
)
from helium.utils import iter_batch


class ParallelBenchmarkConfig(BenchmarkConfig):
    """
    Configuration specific to Parallel Chain benchmark tasks.
    """

    def __init__(
        self,
        num_trials: int,
        helium_profiling: bool,
        enable_cache_aware_scheduling: bool,
        enable_runtime_adjustment: bool,
        enable_query_profiling: bool,
        helium_prebuilt_file: str | Path | None,
        runner_config: RunnerConfig,
        dataset_name: str,
        num_experts: int,
        num_items: int,
        num_review_chunks_per_item: int,
        num_reviews_per_chunk: int,
        dev_size: int = 30,
    ) -> None:
        super().__init__(
            system=runner_config.system,
            num_trials=num_trials,
            helium_profiling=helium_profiling,
            enable_cache_aware_scheduling=enable_cache_aware_scheduling,
            enable_runtime_adjustment=enable_runtime_adjustment,
            enable_query_profiling=enable_query_profiling,
            helium_prebuilt_file=helium_prebuilt_file,
        )

        llm_server_config = runner_config.llm_server_config
        kwargs: dict[str, Any] = {
            "model": llm_server_config.model,
            "base_url": f"http://{llm_server_config.host}:{llm_server_config.port}/v1",
            "temperature": 0,
            "max_tokens": 512,
        }
        self.generation_config = GenerationConfig.from_env(**kwargs)

        self.num_experts = num_experts
        self.num_review_chunks_per_item = num_review_chunks_per_item

        self.dataset_name = dataset_name
        # Load the dataset
        if helium_profiling:
            num_items = dev_size
            split = "dev"
        else:
            split = "test"
        item_reviews: list[tuple[dict[str, Any], list[list[dict[str, Any]]]]]
        num_reviews_per_item = num_reviews_per_chunk * num_review_chunks_per_item
        match dataset_name:
            case "amazon":
                dataset = AmazonReviewsDataset(split=split)
                item_reviews = [
                    (item, list(iter_batch(reviews, num_reviews_per_chunk)))
                    for item, reviews in dataset.iter_item_reviews(
                        num_items, num_reviews_per_item
                    )
                ]
            case _:
                raise ValueError(f"Unsupported dataset: {dataset_name}")
        self.item_reviews = item_reviews

    @property
    def input_size(self) -> int:
        return len(self.item_reviews)


class ParallelBenchmarkTask(BenchmarkTask):
    """
    Benchmark task based on Parallel Chain Generation
    """

    def __init__(
        self,
        config: ParallelBenchmarkConfig,
        runner: BenchmarkRunner,
        name_suffix: str | None = None,
    ) -> None:
        workload = f"parallel-{config.dataset_name}"
        if name_suffix is not None:
            workload += f"-{name_suffix}"
        self.config: ParallelBenchmarkConfig
        super().__init__(workload=workload, config=config, runner=runner)

    def create_program(self) -> Any:
        system_name = self.config.system.lower()
        runner_config = self.runner.config
        if system_name.startswith("querywise"):
            return QueryWiseParallelProgram()
        elif system_name.startswith("opwise"):
            return OpWiseParallelProgram()
        elif system_name.startswith("autogen"):
            return AutoGenParallelProgram()
        elif system_name.startswith("langgraph"):
            return LangGraphParallelProgram()
        elif system_name.startswith("agentscope"):
            return ASParallelProgram()
        elif system_name.startswith("parrot"):
            llm_service_config = runner_config.llm_server_config
            return ParrotParallelProgram(
                llm_service_config.host, llm_service_config.port
            )
        elif system_name.startswith("helium"):
            if self.config.helium_prebuilt_file:
                agent, query_profile = ParallelAgent.load(
                    self.config.helium_prebuilt_file
                )
                query_profile_map = (
                    None if query_profile is None else {agent.name: query_profile}
                )
            else:
                agent = query_profile_map = None
            helium_server_config = HeliumServerConfig(is_local=True, benchmarking=True)
            helium_request_config = HeliumRequestConfig(
                enable_cache_aware_scheduling=self.config.enable_cache_aware_scheduling,
                enable_runtime_adjustment=self.config.enable_runtime_adjustment,
                system_profiling_config=SystemProfilingConfig(),
                query_profiling_config=(
                    QueryProfilingConfig(query_profile_map=query_profile_map)
                    if self.config.enable_query_profiling
                    else None
                ),
            )
            return HeliumParallelProgram(
                helium_request_config, helium_server_config, agent
            )
        else:
            raise ValueError(f"Unsupported system: {self.config.system}")

    def get_run_configurations(
        self,
    ) -> tuple[list[str], list[Any], list[list[dict]]]:
        config = self.config
        run_names = [self.system_name]
        kwargs = [
            [
                {
                    "dataset": config.dataset_name,
                    "item_reviews": config.item_reviews,
                    "num_experts": config.num_experts,
                    "num_review_chunks_per_item": config.num_review_chunks_per_item,
                    "generation_config": config.generation_config,
                }
            ]
            * config.num_trials
        ]
        return run_names, [self.create_program()], kwargs
