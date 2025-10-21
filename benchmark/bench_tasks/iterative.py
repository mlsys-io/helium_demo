from pathlib import Path
from typing import Any

from bench_programs.iterative.agentscope import ASIterativeProgram
from bench_programs.iterative.autogen import AutoGenIterativeProgram
from bench_programs.iterative.helium import HeliumIterativeProgram, IterativeAgent
from bench_programs.iterative.langgraph import LangGraphIterativeProgram
from bench_programs.iterative.opwise import OpWiseIterativeProgram
from bench_programs.iterative.parrot import ParrotIterativeProgram
from bench_programs.iterative.querywise import QueryWiseIterativeProgram
from bench_tasks.base import BenchmarkConfig, BenchmarkTask
from bench_utils.datasets.amazon import AmazonReviewsDataset
from bench_utils.datasets.arxiv import ArxivDataset
from bench_utils.runner.base import BenchmarkRunner, RunnerConfig

from helium.common import GenerationConfig
from helium.runtime import HeliumServerConfig
from helium.runtime.protocol import (
    HeliumRequestConfig,
    QueryProfilingConfig,
    SystemProfilingConfig,
)
from helium.utils import iter_batch


class IterativeBenchmarkConfig(BenchmarkConfig):
    """
    Configuration specific to Iterative Refinement benchmark tasks.
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

        self.dataset_name = dataset_name
        # Load the dataset
        if helium_profiling:
            num_items = dev_size
            split = "dev"
        else:
            split = "test"
        num_reviews_per_item = num_reviews_per_chunk * num_review_chunks_per_item
        document_chunks: list[tuple[Any, ...]]
        match dataset_name:
            case "arxiv":
                if split == "dev":
                    split = "val"
                dataset = ArxivDataset(split=split)
                document_chunks = list(
                    dataset.iter_articles(num_items, num_review_chunks_per_item)
                )
            case "amazon":
                dataset = AmazonReviewsDataset(split=split)
                document_chunks = [
                    tuple(iter_batch(reviews, num_reviews_per_chunk))
                    for _, reviews in dataset.iter_item_reviews(
                        num_items, num_reviews_per_item
                    )
                ]
            case _:
                raise ValueError(f"Unsupported dataset: {dataset_name}")
        self.document_chunks = document_chunks

    @property
    def input_size(self) -> int:
        return len(self.document_chunks)


class IterativeBenchmarkTask(BenchmarkTask):
    """
    Benchmark task based on Iterative Generation
    """

    def __init__(
        self,
        config: IterativeBenchmarkConfig,
        runner: BenchmarkRunner,
        name_suffix: str | None = None,
    ) -> None:
        workload = f"iterative-{config.dataset_name}"
        if name_suffix is not None:
            workload += f"-{name_suffix}"
        self.config: IterativeBenchmarkConfig
        super().__init__(workload=workload, config=config, runner=runner)

    def create_program(self) -> Any:
        system_name = self.config.system.lower()
        runner_config = self.runner.config
        if system_name.startswith("querywise"):
            return QueryWiseIterativeProgram()
        elif system_name.startswith("opwise"):
            return OpWiseIterativeProgram()
        elif system_name.startswith("autogen"):
            return AutoGenIterativeProgram()
        elif system_name.startswith("langgraph"):
            return LangGraphIterativeProgram()
        elif system_name.startswith("agentscope"):
            return ASIterativeProgram()
        elif system_name.startswith("parrot"):
            llm_service_config = runner_config.llm_server_config
            return ParrotIterativeProgram(
                llm_service_config.host, llm_service_config.port
            )
        elif system_name.startswith("helium"):
            if self.config.helium_prebuilt_file:
                agent, query_profile = IterativeAgent.load(
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
            return HeliumIterativeProgram(
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
                    "document_chunks": config.document_chunks,
                    "generation_config": config.generation_config,
                }
            ]
            * config.num_trials
        ]
        return run_names, [self.create_program()], kwargs
