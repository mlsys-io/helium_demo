from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from bench_utils.mixin import BenchmarkMixin
from bench_utils.runner.base import BenchmarkRunner
from bench_utils.runner.local import BenchmarkRunnerWithLocalHelium

from helium import helium
from helium.runtime.protocol import PrefixMap


class BenchmarkConfig:
    def __init__(
        self,
        system: str,
        num_trials: int,
        helium_profiling: bool,
        enable_cache_aware_scheduling: bool,
        enable_runtime_adjustment: bool,
        enable_query_profiling: bool,
        helium_prebuilt_file: str | Path | None,
    ):
        self.system = system
        """Name of the system to benchmark, e.g., "autogen" or "helium"."""
        self.num_trials = num_trials
        """Number of trials to run for each benchmark configuration."""
        self.helium_profiling = helium_profiling
        """Whether to run in profiling mode for Helium."""
        self.helium_prebuilt_file = helium_prebuilt_file
        """Path to a prebuilt Helium file, if applicable."""
        self.enable_cache_aware_scheduling = enable_cache_aware_scheduling
        """Whether to enable cache-aware scheduling for the benchmark."""
        self.enable_runtime_adjustment = enable_runtime_adjustment
        """Whether to enable runtime adjustment for CAS"""
        self.enable_query_profiling = enable_query_profiling
        """Whether to enable query profiling for the benchmark."""

        self.validate()

    def validate(self) -> None:
        if self.helium_profiling:
            if "helium" not in self.system.lower():
                raise ValueError(
                    "helium_profiling is only supported for the Helium system."
                )
            if self.helium_prebuilt_file is not None:
                raise ValueError(
                    "Helium profiling mode is enabled, but a prebuilt file is provided."
                )

    @property
    @abstractmethod
    def input_size(self) -> int:
        pass


class BenchmarkTask(ABC):
    """Abstract base class for benchmark tasks."""

    def __init__(
        self, workload: str, config: BenchmarkConfig, runner: BenchmarkRunner
    ) -> None:
        self.workload = workload
        self.config = config
        self.runner = runner

    @property
    def system_name(self) -> str:
        return self.config.system

    @abstractmethod
    def create_program(self, *args, **kwargs) -> BenchmarkMixin:
        pass

    @abstractmethod
    def get_run_configurations(
        self,
    ) -> tuple[list[str], list[BenchmarkMixin], list[list[dict[str, Any]]]]:
        """
        Returns
        -------
        list[str]
            A list of names for each run variant,
        list[Any]
            A list of benchmark programs (one per variant)
        list[list[dict[str, Any]]]
            A list (per variant) of keyword argument dictionaries for repeated trials.
        """
        pass

    async def run(self) -> dict[str, pd.DataFrame]:
        runner = self._set_workload()
        run_names, bench_list, kwargs_lists = self.get_run_configurations()
        await runner.run_all(run_names, bench_list, kwargs_lists=kwargs_lists)
        runner.print_summary()
        summary_df_dict = runner.get_summary_dfs()
        return summary_df_dict

    async def precompute(
        self, precompute_mode: Literal["none", "only", "both"]
    ) -> tuple[PrefixMap | None, dict[int, PrefixMap] | None]:
        system_name = self.config.system.lower()
        if "helium" not in system_name:
            return None, None

        runner = self._set_workload()
        assert isinstance(runner, BenchmarkRunnerWithLocalHelium)

        run_names, bench_list, kwargs_lists = self.get_run_configurations()
        if not (
            len(run_names) == 1 and len(bench_list) == 1 and len(kwargs_lists) == 1
        ):
            raise ValueError("Precomputation only supports a single benchmark.")
        run_name = run_names[0]
        bench = bench_list[0]
        kwargs = kwargs_lists[0][0]

        print(f"Precomputing benchmark '{runner.get_bench_name()}' on '{run_name}'...")

        server = helium.get_instance()
        assert server.is_started
        precompute_kv_cache = (
            precompute_mode in ("only", "both")
            and runner.config.llm_server_config.kv_transfer_config is not None
        )
        if precompute_kv_cache:
            # Change the KV role to "kv_both" for precomputation
            server.change_llm_workers_kv_role("kv_both")

        await runner.init_run(run_name)

        print(f"Run {run_name} initialized.", flush=True)
        resp = await bench.precompute(**kwargs, precompute_mode=precompute_mode)

        # Get the KV cache manager
        prompt_cache_manager = server.processor.prompt_cache_manager
        if prompt_cache_manager is not None:
            prompt_cache_manager.freeze()

        await runner.clean_up_run()

        if precompute_kv_cache:
            # Change the KV role back to "kv_consumer" after precomputation
            server.change_llm_workers_kv_role("kv_consumer")

        return resp.static_prefix_map, resp.dynamic_prefix_map

    def precompute_prefixes(self, prefix_map: PrefixMap) -> None:
        print("Precomputing prefixes...")
        server = helium.get_instance()
        assert server.is_started
        server.precompute_prefixes(prefix_map)

    def _set_workload(self) -> BenchmarkRunner:
        self.runner.set_workload(
            self.workload,
            self.config.input_size,
            self.config.enable_cache_aware_scheduling,
            self.config.enable_runtime_adjustment,
            self.config.enable_query_profiling,
            self.config.helium_prebuilt_file,
        )
        return self.runner
