import itertools
import multiprocessing as mp
import os
import time
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import Any, Literal, NamedTuple, NoReturn, TypedDict

import matplotlib.pyplot as plt
import pandas as pd
import pynvml
import tqdm
from bench_utils.mixin import BenchmarkMixin
from matplotlib.axes import Axes

from helium import envs
from helium.runtime.cache_manager import CacheManagerConfig
from helium.runtime.protocol import HeliumSystemProfile
from helium.runtime.utils.vllm.config import CompiledServerConfig
from helium.runtime.utils.vllm.server import (
    EngineClientInfo,
    configure_vllm_logging,
    start_mock_server,
    start_server,
    start_server_with_controller,
)
from vllm.vllm.config import KVTransferConfig

_SYSTEM_PROFILE_KEYS = Literal[
    "llm_benchmark", "task_profile", "range_profile", "request_profile"
]
_GET_DF_FUNC_TYPE = Callable[[str, str, str, str, int, str, int, Any], pd.DataFrame]


class LLMServerKwargs(TypedDict):
    model: str
    host: str
    port: int
    device: str
    hf_overrides: dict[str, Any] | None
    kv_transfer_config: KVTransferConfig | None


@dataclass
class LLMServerConfig:
    model: str
    host: str
    port: int
    num_llm_workers: int
    main_log_file: Path | None
    log_files: list[Path | None]
    kwargs_list: list[LLMServerKwargs]
    cache_manager_config: CacheManagerConfig | None
    kv_transfer_config: KVTransferConfig | None
    hf_overrides: dict[str, Any] | None = None
    context_length: int | None = None

    def __post_init__(self):
        if len(self.log_files) != len(self.kwargs_list):
            raise ValueError("log_files and kwargs_list must have the same length.")

    def extend_context_length(
        self, factor: float, original_max_position_embeddings: int
    ) -> None:
        hf_overrides = {
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": factor,
                "original_max_position_embeddings": original_max_position_embeddings,
            }
        }
        # Set the main config's hf_overrides
        if self.hf_overrides is None:
            self.hf_overrides = hf_overrides
        else:
            self.hf_overrides.update(hf_overrides)
        # Set each worker config's hf_overrides
        for kwargs in self.kwargs_list:
            if kwargs["hf_overrides"] is None:
                kwargs["hf_overrides"] = hf_overrides
            else:
                kwargs["hf_overrides"].update(hf_overrides)
        # Set the context length
        self.context_length = int(factor * original_max_position_embeddings)


@dataclass
class RunnerConfig:
    system: str
    """Name of the system to benchmark, e.g., "autogen" or "helium"."""
    cuda_device_name: str | None
    """Name of the CUDA device to use for benchmarking."""
    gpu_util_log_dir: Path | None
    """Directory to save GPU utilization logs. If None, GPU monitoring is disabled."""
    llm_server_config: LLMServerConfig
    """Configuration for the LLM server."""
    enable_prefix_caching: bool
    """Whether to enable prefix caching for the benchmark."""
    num_gpu_blocks_override: int | None
    """Override for the number of GPU blocks to use, if applicable."""
    verbose: bool
    """Whether to print verbose output during benchmarking."""

    def __post_init__(self):
        # Extend context length
        model = self.llm_server_config.model
        if model.startswith("Qwen/Qwen3-"):
            if "32B" in model:
                self.llm_server_config.extend_context_length(2.0, 32768)
            else:
                self.llm_server_config.extend_context_length(4.0, 32768)

    def get_config_str(
        self,
        enable_cache_aware_scheduling: bool,
        enable_runtime_adjustment: bool,
        enable_query_profiling: bool,
        helium_prebuilt_file: str | Path | None,
    ) -> str:
        llm_server_config = self.llm_server_config
        config_list = []
        config_list.append(f"{llm_server_config.num_llm_workers}wkrs")
        if self.enable_prefix_caching:
            config_list.append("APC")
        if enable_cache_aware_scheduling:
            config_list.append("CAS")
        if llm_server_config.cache_manager_config is not None:
            cache_manager_config = llm_server_config.cache_manager_config
            if cache_manager_config.enable_proactive_kv_cache:
                config_list.append("KV")
            if cache_manager_config.enable_prompt_cache:
                config_list.append("PC")
        if enable_runtime_adjustment:
            config_list.append("RA")
        if enable_query_profiling:
            config_list.append("QP")
        if helium_prebuilt_file is not None:
            config_list.append("PB")
        if self.num_gpu_blocks_override is not None:
            config_list.append(f"{self.num_gpu_blocks_override}blk")
        config = "+".join(config_list)
        return config

    @property
    def devices(self) -> list[int]:
        devices: list[int] = []
        for kwargs in self.llm_server_config.kwargs_list:
            device = kwargs["device"]
            if device == "cuda":
                devices.append(0)
            else:
                devices.append(int(device.removeprefix("cuda:")))
        return devices


@dataclass
class BenchmarkResult:
    model: str
    system: str
    workload: str
    config: str
    input_size: int
    run_name: str
    elapsed_times: list[dict[str, list[float]]]
    """List of elapsed times (per key) for each trial."""
    system_profiles: list[HeliumSystemProfile]
    """List of system profiles (per key) for each trial."""

    @property
    def total_elapsed_times(self) -> list[dict[str, float]]:
        """List of total elapsed times (per key) for each trial."""
        return [
            {key: sum(times) for key, times in trial_times.items()}
            for trial_times in self.elapsed_times
        ]

    @property
    def average_elapsed_times(self) -> dict[str, float]:
        """Average elapsed times (per key) across all trials."""
        total_elapsed_times = self.total_elapsed_times
        key_times: dict[str, list[float]] = {}
        for trial_times in total_elapsed_times:
            for key, trial_time in trial_times.items():
                if key not in key_times:
                    key_times[key] = []
                key_times[key].append(trial_time)
        average_times = {
            key: (sum(times) / len(times)) for key, times in key_times.items()
        }
        return average_times

    def get_result_dfs(
        self,
        elapsed_time: bool = True,
        llm_benchmark: bool = True,
        task_profile: bool = False,
        range_profile: bool = True,
        request_profile: bool = True,
    ) -> dict[str, pd.DataFrame]:
        results_df_dict: dict[str, pd.DataFrame] = {}

        # Average elapsed times
        if elapsed_time:
            elapsed_time_df = pd.DataFrame()
            for i, trial_results in enumerate(self.total_elapsed_times):
                new_df = self._get_elapsed_time_df(
                    self.model,
                    self.system,
                    self.workload,
                    self.config,
                    self.input_size,
                    self.run_name,
                    i,
                    trial_results,
                )
                elapsed_time_df = pd.concat(
                    [elapsed_time_df, new_df], ignore_index=True
                )
            results_df_dict["elapsed_time"] = elapsed_time_df

        # System profiling results
        to_process: list[tuple[bool, _SYSTEM_PROFILE_KEYS, _GET_DF_FUNC_TYPE]] = [
            (llm_benchmark, "llm_benchmark", self._get_llm_benchmark_df),
            (task_profile, "task_profile", self._get_task_profile_df),
            (range_profile, "range_profile", self._get_range_profile_df),
            (request_profile, "request_profile", self._get_request_profile_df),
        ]
        for to_include, result_key, get_df_func in to_process:
            if to_include:
                self._get_profiling_result_df(result_key, get_df_func, results_df_dict)

        return results_df_dict

    def _get_profiling_result_df(
        self,
        result_key: _SYSTEM_PROFILE_KEYS,
        get_df_func: _GET_DF_FUNC_TYPE,
        results_df_dict: dict[str, pd.DataFrame],
    ) -> None:
        result_df: pd.DataFrame | None = None
        for i, trial_results in enumerate(self.system_profiles):
            if result_key not in trial_results:
                continue
            new_df = get_df_func(
                self.model,
                self.system,
                self.workload,
                self.config,
                self.input_size,
                self.run_name,
                i,
                trial_results.get(result_key),
            )
            result_df = (
                new_df
                if result_df is None
                else pd.concat([result_df, new_df], ignore_index=True)
            )
        if result_df is not None:
            results_df_dict[result_key] = result_df

    def _get_elapsed_time_df(
        self,
        model: str,
        system: str,
        workload: str,
        config: str,
        input_size: int,
        run_name: str,
        trial: int,
        total_elapsed_times: dict[str, float],
    ) -> pd.DataFrame:
        indices = {
            "model": [model],
            "system": [system],
            "workload": [workload],
            "config": [config],
            "input_size": [input_size],
            "run_name": [run_name],
            "trial": [trial],
        }
        results_dict: dict[str, list[float]] = {
            k: [v] for k, v in total_elapsed_times.items()
        }
        return pd.DataFrame(indices | results_dict)

    def _get_llm_benchmark_df(
        self,
        model: str,
        system: str,
        workload: str,
        config: str,
        input_size: int,
        run_name: str,
        trial: int,
        llm_benchmarks: dict[str, dict[str, dict[str, Any]]],
    ) -> pd.DataFrame:
        # Get all percentiles
        percentile_sets: dict[str, set[float]] = {
            "percentiles_ttft_ms": set(),
            "percentiles_tpot_ms": set(),
            "percentiles_itl_ms": set(),
            "percentiles_e2el_ms": set(),
        }
        for llm_benchmark in llm_benchmarks.values():
            for results in llm_benchmark.values():
                for metric, values in percentile_sets.items():
                    key = f"percentiles_{metric}"
                    if key in results:
                        for percentile in results[key]:
                            values.add(int(percentile[0]))
        percentiles_dict = {p: sorted(v) for p, v in percentile_sets.items()}

        indices: dict[str, list[Any]] = {
            "model": [],
            "system": [],
            "workload": [],
            "config": [],
            "input_size": [],
            "run_name": [],
            "llm_service": [],
            "base_url": [],
            "trial": [],
        }
        results_dict: dict[str, list[float | None]] = {
            "completed": [],
            "total_input": [],
            "total_output": [],
            "request_throughput": [],
            "request_goodput": [],
            "output_throughput": [],
            "total_token_throughput": [],
            "mean_ttft_ms": [],
            "median_ttft_ms": [],
            "std_ttft_ms": [],
            "mean_tpot_ms": [],
            "median_tpot_ms": [],
            "std_tpot_ms": [],
            "mean_itl_ms": [],
            "median_itl_ms": [],
            "std_itl_ms": [],
            "mean_e2el_ms": [],
            "median_e2el_ms": [],
            "std_e2el_ms": [],
            "num_preemptions": [],
            "num_preemptions_total": [],
            "gpu_prefix_cache_queries": [],
            "gpu_prefix_cache_hits": [],
            "gpu_prefix_cache_hit_rate": [],
        }
        # Add percentiles
        for key, percentiles in percentiles_dict.items():
            for percentile in percentiles:
                results_dict[f"{key}_{percentile}"] = []

        for llm_service, llm_benchmark in llm_benchmarks.items():
            for base_url, results in llm_benchmark.items():
                indices["model"].append(model)
                indices["system"].append(system)
                indices["workload"].append(workload)
                indices["config"].append(config)
                indices["input_size"].append(input_size)
                indices["run_name"].append(run_name)
                indices["llm_service"].append(llm_service)
                indices["base_url"].append(base_url)
                indices["trial"].append(trial)
                for key in results_dict:
                    if key.startswith("percentiles_"):
                        metric_key, percentile_str = key.rsplit("_", maxsplit=1)
                        if metric_key not in results:
                            results_dict[key].append(None)
                            continue
                        percentile = int(percentile_str)
                        percentile_res: list[list[float]] = results[metric_key]
                        for p, v in percentile_res:
                            if p == percentile:
                                results_dict[key].append(v)
                                break
                        else:
                            raise ValueError("Percentile not found")
                    else:
                        if key in results:
                            results_dict[key].append(results.get(key))
                        else:
                            results_dict[key].append(None)

        df = pd.DataFrame(indices | results_dict)
        if df["gpu_prefix_cache_hit_rate"].isna().all():
            df["gpu_prefix_cache_hit_rate"] = (
                df["gpu_prefix_cache_hits"] / df["gpu_prefix_cache_queries"]
            )
        return df

    def _get_task_profile_df(
        self,
        model: str,
        system: str,
        workload: str,
        config: str,
        input_size: int,
        run_name: str,
        trial: int,
        task_profile: dict[str, list[tuple[int, float]] | None],
    ) -> pd.DataFrame:
        indices: dict[str, list[Any]] = {
            "model": [],
            "system": [],
            "workload": [],
            "config": [],
            "input_size": [],
            "run_name": [],
            "trial": [],
            "task_id": [],
            "iteration": [],
        }
        results_dict: dict[str, list[float | None]] = {"elapsed_time": []}

        for task_id, iter_elapsed_times in task_profile.items():
            if iter_elapsed_times is None:
                indices["model"].append(model)
                indices["system"].append(system)
                indices["workload"].append(workload)
                indices["config"].append(config)
                indices["input_size"].append(input_size)
                indices["run_name"].append(run_name)
                indices["trial"].append(trial)
                indices["task_id"].append(task_id)
                indices["iteration"].append(None)
                results_dict["elapsed_time"].append(None)
                continue
            for iteration, elapsed_time in iter_elapsed_times:
                indices["model"].append(model)
                indices["system"].append(system)
                indices["workload"].append(workload)
                indices["config"].append(config)
                indices["input_size"].append(input_size)
                indices["run_name"].append(run_name)
                indices["trial"].append(trial)
                indices["task_id"].append(task_id)
                indices["iteration"].append(iteration)
                results_dict["elapsed_time"].append(elapsed_time)

        return pd.DataFrame(indices | results_dict)

    def _get_range_profile_df(
        self,
        model: str,
        system: str,
        workload: str,
        config: str,
        input_size: int,
        run_name: str,
        trial: int,
        range_profile: dict[str, float],
    ) -> pd.DataFrame:
        indices: dict[str, list[Any]] = {
            "model": [],
            "system": [],
            "workload": [],
            "config": [],
            "input_size": [],
            "run_name": [],
            "trial": [],
            "range_id": [],
        }
        results_dict: dict[str, list[float]] = {"elapsed_time": []}

        for range_id, elapsed_time in range_profile.items():
            indices["model"].append(model)
            indices["system"].append(system)
            indices["workload"].append(workload)
            indices["config"].append(config)
            indices["input_size"].append(input_size)
            indices["run_name"].append(run_name)
            indices["trial"].append(trial)
            indices["range_id"].append(range_id)
            results_dict["elapsed_time"].append(elapsed_time)

        return pd.DataFrame(indices | results_dict)

    def _get_request_profile_df(
        self,
        model: str,
        system: str,
        workload: str,
        config: str,
        input_size: int,
        run_name: str,
        trial: int,
        request_profile: dict[str, float],
    ) -> pd.DataFrame:
        indices: dict[str, list[Any]] = {
            "model": [],
            "system": [],
            "workload": [],
            "config": [],
            "input_size": [],
            "run_name": [],
            "trial": [],
            "range_id": [],
        }
        results_dict: dict[str, list[float]] = {"elapsed_time": []}

        for range_id, elapsed_time in request_profile.items():
            indices["model"].append(model)
            indices["system"].append(system)
            indices["workload"].append(workload)
            indices["config"].append(config)
            indices["input_size"].append(input_size)
            indices["run_name"].append(run_name)
            indices["trial"].append(trial)
            indices["range_id"].append(range_id)
            results_dict["elapsed_time"].append(elapsed_time)

        return pd.DataFrame(indices | results_dict)


class WorkloadInfo(NamedTuple):
    system: str
    model: str
    workload: str
    input_size: int
    config_str: str
    bench_name: str


class BenchmarkRunner:
    def __init__(self, config: RunnerConfig) -> None:
        self.config = config
        self._bench_results: dict[str, BenchmarkResult] = {}

        self._in_task_context = False
        self._workload_info: WorkloadInfo | None = None

    def reset(self) -> None:
        self._bench_results = {}

    @asynccontextmanager
    async def task_context(self):
        try:
            self._in_task_context = True
            yield self
        finally:
            await self._reset_states()
            self._in_task_context = False
            self._workload_info = None

    def set_workload(
        self,
        workload: str,
        input_size: int,
        enable_cache_aware_scheduling: bool,
        enable_runtime_adjustment: bool,
        enable_query_profiling: bool,
        helium_prebuilt_file: str | Path | None,
    ):
        if not self._in_task_context:
            raise ValueError("set_workload() must be called within task_context().")
        config = self.config
        system = config.system
        model = config.llm_server_config.model
        config_str = config.get_config_str(
            enable_cache_aware_scheduling,
            enable_runtime_adjustment,
            enable_query_profiling,
            helium_prebuilt_file,
        )
        bench_name = f"{system} {model} {workload}-{input_size}" + (
            f" ({config_str})" if len(config_str) > 0 else ""
        )
        self._workload_info = WorkloadInfo(
            system, model, workload, input_size, config_str, bench_name
        )

    def get_bench_name(self) -> str:
        if self._workload_info is None:
            raise ValueError("Workload not set. Call set_workload() first.")
        return self._workload_info.bench_name

    async def run(
        self,
        run_name: str,
        bench: BenchmarkMixin,
        args_list: list[list[Any]] | None,
        kwargs_list: list[dict[str, Any]] | None,
    ) -> BenchmarkResult:
        """Runs a benchmark for multiple trials

        Parameters
        ----------
        run_name : str
            Name of the run.
        bench : BenchmarkMixin
            Benchmark object to run.
        args_list : list[list[Any]] | None
            List of arguments for each trial.
        kwargs_list : list[dict[str, Any]] | None
            List of keyword arguments for each trial.

        Returns
        -------
        BenchmarkResult
            Benchmark result object.
        """
        if self._workload_info is None:
            raise ValueError("Workload not set. Call set_workload() first.")

        info = self._workload_info
        print(f"Running benchmark '{info.bench_name}' on {run_name}...")

        num_trials = max(len(args_list or []), len(kwargs_list or []))
        if args_list is None:
            args_list = [[]] * num_trials
        if kwargs_list is None:
            kwargs_list = [{}] * num_trials

        if len(args_list) != len(kwargs_list):
            raise ValueError("args_list and kwargs_list must have the same length.")

        elapsed_times: list[dict[str, list[float]]] = []
        system_profiles: list[HeliumSystemProfile] = []
        for args, kwargs in tqdm.tqdm(
            zip(args_list, kwargs_list),
            total=num_trials,
            disable=(not self.config.verbose) or (num_trials <= 1),
        ):
            await self.init_run(run_name)
            print(f"Run {run_name} initialized.", flush=True)
            bench.reset_timer()
            await bench.run_async(*args, **kwargs)
            elapsed_time = bench.get_elapsed_times()
            elapsed_times.append(elapsed_time)
            system_profiles.append(bench.get_and_reset_system_profile())
            await self.clean_up_run()
        result = BenchmarkResult(
            info.model,
            info.system,
            info.workload,
            info.config_str,
            info.input_size,
            run_name,
            elapsed_times,
            system_profiles,
        )
        self._bench_results[run_name] = result
        return result

    async def run_all(
        self,
        run_names: list[str],
        bench_list: Sequence[BenchmarkMixin],
        args_lists: list[list[list[Any]]] | None = None,
        kwargs_lists: list[list[dict[str, Any]]] | None = None,
    ) -> list[BenchmarkResult]:
        """Runs all benchmarks, each for multiple trials

        Parameters
        ----------
        run_names : list[str]
            List of run names.
        bench_list : list[BenchmarkMixin]
            List of benchmark objects.
        args_lists : list[list[list[Any]]] | None
            List of arguments for each trial in each run.
        kwargs_lists : list[list[dict[str, Any]]] | None
            List of keyword arguments for each trial in each run.

        Returns
        -------
        list[BenchmarkResult]
            List of benchmark results for each run.
        """
        if self._workload_info is None:
            raise ValueError("Workload not set. Call set_workload() first.")

        if len(run_names) != len(bench_list):
            raise ValueError("All arguments must be the same length.")
        else:
            if (
                (args_lists is not None)
                and (len(run_names) != len(args_lists))
                or (kwargs_lists is not None)
                and (len(run_names) != len(kwargs_lists))
            ):
                raise ValueError("All arguments must be the same length.")

        info = self._workload_info
        print(
            f"Running '{info.bench_name}' benchmarks on {len(run_names)} runs {tuple(run_names)}..."
        )

        bench_results = []
        for run_name, bench, args_list, kwargs_list in zip(
            run_names,
            bench_list,
            (itertools.repeat(None) if args_lists is None else args_lists),
            (itertools.repeat(None) if kwargs_lists is None else kwargs_lists),
        ):
            result = await self.run(run_name, bench, args_list, kwargs_list)
            bench_results.append(result)
        return bench_results

    def plot_summary(self) -> Axes:
        """Plots a summary of the benchmark results

        Returns
        -------
        Axes
            Matplotlib axes object.
        """
        average_elapsed_times = {
            run_name: bench_result.average_elapsed_times
            for run_name, bench_result in self._bench_results.items()
        }

        # Extract keys and values for plotting
        keys = list(next(iter(average_elapsed_times.values())).keys())
        runs = list(average_elapsed_times.keys())
        values = {
            key: [average_elapsed_times[run][key] for run in runs] for key in keys
        }

        # Plotting
        _, ax = plt.subplots()

        bottom = [0.0] * len(runs)
        for key in keys:
            ax.bar(runs, values[key], bottom=bottom, label=key)
            bottom = [i + j for i, j in zip(bottom, values[key])]

        ax.set_xlabel("Runs")
        ax.set_ylabel("Elapsed Time")
        ax.set_title("Benchmark Results")
        ax.legend()

        return ax

    def print_summary(self) -> None:
        """Prints a summary of the benchmark results"""
        if self._workload_info is None:
            raise ValueError("Workload not set. Call set_workload() first.")
        info = self._workload_info

        print(f"Benchmark Summary ({info.bench_name}):")
        run_id_columns = [
            "model",
            "system",
            "workload",
            "config",
            "input_size",
            "run_name",
        ]
        for run_name, bench_result in self._bench_results.items():
            print(f"Run: {run_name}")
            result_df_dict = bench_result.get_result_dfs()

            print("  Elapsed Times:")
            elapsed_time_df = result_df_dict["elapsed_time"].drop(
                columns=run_id_columns + ["trial"]
            )
            elapsed_time_df = elapsed_time_df.mean()
            for k, v in elapsed_time_df.items():
                print(f"    {k}: {v:.6f} s")
            print()

            if "llm_benchmark" in result_df_dict:
                print("  LLM Benchmarks:")
                llm_benchmark_df = result_df_dict["llm_benchmark"].drop(
                    columns=run_id_columns + ["trial"]
                )
                llm_benchmark_df = llm_benchmark_df.groupby(
                    ["llm_service", "base_url"]
                ).mean()
                for key, row in llm_benchmark_df.iterrows():
                    assert isinstance(key, tuple)
                    llm_service, base_url = key
                    print(f"    LLM Service = {llm_service} ({base_url}):")
                    for k, v in row.items():
                        print(f"      {k}: {v:.6f}")
                print()

            if "task_profile" in result_df_dict:
                print("  Task Profile:")
                task_profile_df = result_df_dict["task_profile"].drop(
                    columns=run_id_columns + ["iteration"]
                )
                task_profile_df = task_profile_df.groupby(
                    ["trial", "task_id"], as_index=False
                ).sum()
                task_profile_df = (
                    task_profile_df.drop(columns=["trial"]).groupby("task_id").mean()
                )
                for task_id, row in task_profile_df.iterrows():
                    print(f"    {task_id}: {row['elapsed_time']:.6f} s")
                print()

            if "range_profile" in result_df_dict:
                print("  Range Profile:")
                range_profile_df = result_df_dict["range_profile"].drop(
                    columns=run_id_columns + ["trial"]
                )
                range_profile_df = range_profile_df.groupby("range_id").mean()
                for range_id, row in range_profile_df.iterrows():
                    print(f"    {range_id}: {row['elapsed_time']:.6f} s")
                print()

            if "request_profile" in result_df_dict:
                print("  Request Profile:")
                range_profile_df = result_df_dict["request_profile"].drop(
                    columns=run_id_columns + ["trial"]
                )
                range_profile_df = range_profile_df.groupby("range_id").mean()
                for range_id, row in range_profile_df.iterrows():
                    print(f"    {range_id}: {row['elapsed_time']:.6f} s")
                print()

    def get_summary_dfs(self) -> dict[str, pd.DataFrame]:
        summary_df_dict: dict[str, pd.DataFrame] = {}
        for bench_results in self._bench_results.values():
            results_dfs = bench_results.get_result_dfs()
            for key, df in results_dfs.items():
                if key not in summary_df_dict:
                    summary_df_dict[key] = df
                else:
                    prev_df = summary_df_dict[key]
                    summary_df_dict[key] = pd.concat([prev_df, df], ignore_index=True)
        return summary_df_dict

    async def init_run(self, run_name: str) -> None:
        pass

    async def clean_up_run(self) -> None:
        pass

    async def start_runner(self) -> None:
        pass

    async def stop_runner(self) -> None:
        pass

    async def _reset_states(self) -> None:
        pass


def start_llm_server(config: CompiledServerConfig, event: Event) -> None:
    configure_vllm_logging(config)
    if config.inner.lmcache_config_file is not None:
        os.environ["LMCACHE_CONFIG_FILE"] = str(config.inner.lmcache_config_file)
    if envs.DEBUG_MOCK_LLM_ONLY:
        start_mock_server(config, event)
    else:
        start_server(config, event)


def start_llm_server_with_controller(
    controller_config: CompiledServerConfig,
    worker_configs: Sequence[CompiledServerConfig],
    worker_infos: Sequence[EngineClientInfo],
    event: Event,
) -> None:
    configure_vllm_logging(controller_config)
    if controller_config.inner.lmcache_config_file is not None:
        os.environ["LMCACHE_CONFIG_FILE"] = str(
            controller_config.inner.lmcache_config_file
        )
    if envs.DEBUG_MOCK_LLM_ONLY:
        start_mock_server(controller_config, event)
    else:
        start_server_with_controller(
            controller_config, worker_configs, worker_infos, event=event
        )


def start_gpu_monitor(
    log_file: Path, devices: list[int], sampling_period: float = 1 / 6
) -> mp.Process:
    event = mp.Event()
    gpu_monitor_proc = mp.Process(
        target=monitor_gpu, args=(log_file, devices, event, sampling_period)
    )
    gpu_monitor_proc.start()
    event.wait()
    return gpu_monitor_proc


def monitor_gpu(
    log_file: Path, devices: list[int], event: Event, sampling_period: float
) -> NoReturn:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    pynvml.nvmlInit()
    file_handles = [
        (
            open(log_file.with_suffix(f".gpu{device}.log"), "w"),
            pynvml.nvmlDeviceGetHandleByIndex(device),
        )
        for device in devices
    ]
    event.set()
    while True:
        for f, handle in file_handles:
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            f.write(f"[{time.time()}] gpu={gpu_util.gpu}, mem={gpu_util.memory}\n")
            f.flush()
        time.sleep(sampling_period)
