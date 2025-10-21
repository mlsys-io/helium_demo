from pathlib import Path
from typing import Any

from bench_programs.reflection.agentscope import ASReflectionProgram
from bench_programs.reflection.autogen import AutoGenReflectionProgram
from bench_programs.reflection.helium import HeliumReflectionProgram, ReflectionAgent
from bench_programs.reflection.langgraph import LangGraphReflectionProgram
from bench_programs.reflection.opwise import OpWiseReflectionProgram
from bench_programs.reflection.parrot import ParrotReflectionProgram
from bench_programs.reflection.querywise import QueryWiseReflectionProgram
from bench_tasks.base import BenchmarkConfig, BenchmarkTask
from bench_utils.datasets.finqa import FinQADataset
from bench_utils.datasets.tatqa import TatQADataset
from bench_utils.runner.base import BenchmarkRunner, RunnerConfig

from helium.common import GenerationConfig
from helium.runtime import HeliumServerConfig
from helium.runtime.protocol import (
    HeliumRequestConfig,
    QueryProfilingConfig,
    SystemProfilingConfig,
)


class ReflectionBenchmarkConfig(BenchmarkConfig):
    """
    Configuration specific to Reflection benchmark tasks.
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
        num_contexts: int,
        num_questions_per_context: int | None,
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
            num_contexts = dev_size
            split = "dev"
        else:
            split = "test"
        contexts: list[str]
        context_questions: tuple[list[str], ...]
        context_questions_raw: list[tuple[str, ...]]
        match dataset_name:
            case "finqa":
                num_questions_per_context = 1
                dataset = FinQADataset(split=split)
                contexts = []
                context_questions_raw = []
                for context, questions, _ in dataset.iter_context_qa_pairs(
                    num_contexts
                ):
                    contexts.append(context)
                    context_questions_raw.append((questions,))
                # Pivot the context question list
                context_questions = tuple(
                    list(qs) for qs in zip(*context_questions_raw)
                )
            case "tatqa":
                dataset = TatQADataset(
                    split=split, num_questions=num_questions_per_context
                )
                contexts = []
                context_questions_raw = []
                for context, questions, _ in dataset.iter_context_qa_pairs(
                    num_contexts
                ):
                    contexts.append(context)
                    context_questions_raw.append(questions)
                # Pivot the context question list
                context_questions = tuple(
                    list(qs) for qs in zip(*context_questions_raw)
                )
            case _:
                raise ValueError(f"Unsupported dataset: {dataset_name}")
        self.contexts = contexts
        self.context_questions = context_questions

    @property
    def input_size(self) -> int:
        return len(self.contexts)


class ReflectionBenchmarkTask(BenchmarkTask):
    """
    Benchmark task based on a [Reflection Framework](https://arxiv.org/pdf/2410.21741v1)
    """

    def __init__(
        self,
        config: ReflectionBenchmarkConfig,
        runner: BenchmarkRunner,
        name_suffix: str | None = None,
    ) -> None:
        workload = f"reflection-{config.dataset_name}"
        if name_suffix is not None:
            workload += f"-{name_suffix}"
        self.config: ReflectionBenchmarkConfig
        super().__init__(workload=workload, config=config, runner=runner)

    def create_program(self) -> Any:
        system_name = self.config.system.lower()
        runner_config = self.runner.config
        if system_name.startswith("querywise"):
            return QueryWiseReflectionProgram()
        elif system_name.startswith("opwise"):
            return OpWiseReflectionProgram()
        elif system_name.startswith("autogen"):
            return AutoGenReflectionProgram()
        elif system_name.startswith("langgraph"):
            return LangGraphReflectionProgram()
        elif system_name.startswith("agentscope"):
            return ASReflectionProgram()
        elif system_name.startswith("parrot"):
            llm_service_config = runner_config.llm_server_config
            return ParrotReflectionProgram(
                llm_service_config.host, llm_service_config.port
            )
        elif system_name.startswith("helium"):
            if self.config.helium_prebuilt_file:
                agent, query_profile = ReflectionAgent.load(
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
            return HeliumReflectionProgram(
                helium_request_config, helium_server_config, agent
            )
        else:
            raise ValueError(f"Unsupported system: {runner_config.system}")

    def get_run_configurations(
        self,
    ) -> tuple[list[str], list[Any], list[list[dict]]]:
        config = self.config
        run_names = [self.system_name]
        kwargs = [
            [
                {
                    "dataset": config.dataset_name,
                    "contexts": config.contexts,
                    "context_questions": config.context_questions,
                    "generation_config": config.generation_config,
                }
            ]
            * config.num_trials
        ]
        return run_names, [self.create_program()], kwargs
