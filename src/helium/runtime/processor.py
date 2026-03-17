import asyncio
import enum
import itertools
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any, Generic, Hashable, Literal, TypeVar, cast

from helium.common import GenerationConfig, Message, Slice
from helium.graphs import CompiledGraph
from helium.graphs.utils import partition_op, sample_disjoint_graph
from helium.ops import (
    AppendMessageOp,
    CacheFetchOp,
    ConcatOp,
    DataOp,
    EnterOp,
    ExitOp,
    FormatOp,
    FutureOp,
    InputOp,
    LambdaOp,
    LastMessageOp,
    LLMChatOp,
    LLMCompletionOp,
    LLMOp,
    MergeOp,
    MessageOp,
    Op,
    OutputOp,
    SliceOp,
    SwitchOp,
)
from helium.runtime.cache_manager import (
    CacheManagerConfig,
    KVCacheManager,
    PromptCacheManager,
)
from helium.runtime.data import Data
from helium.runtime.functional import (
    AppendMessageFnInput,
    CacheFetchFnInput,
    ConcatFnInput,
    DataFnInput,
    EnterFnInput,
    ExitFnInput,
    FnInputBatch,
    FormatFnInput,
    InputFnInput,
    LambdaFnInput,
    LastMessageFnInput,
    MergeFnInput,
    MessageFnInput,
    OutputFnInput,
    SliceFnInput,
    SwitchFnInput,
)
from helium.runtime.llm import LLMProfilingInfo, LLMServiceConfig, LLMServiceInfo
from helium.runtime.llm.schedule import (
    batch_wise_schedule,
    cache_aware_schedule,
    op_wise_schedule,
)
from helium.runtime.profiler import RequestProfiler
from helium.runtime.protocol import (
    HeliumQueryProfile,
    HeliumSystemProfile,
    PrefixKey,
    PrefixMap,
    SystemProfilingConfig,
)
from helium.runtime.request import RequestInfo
from helium.runtime.utils.logger import (
    Logger,
    LogLevel,
    init_child_logger,
    log_on_exception_async,
)
from helium.runtime.worker.llm_worker import CacheItemType
from helium.runtime.worker.llm_worker import DispatchMode as LLMDispatchMode
from helium.runtime.worker.llm_worker import (
    LLMChatWorkerInput,
    LLMCompletionWorkerInput,
    LLMPrecomputeInput,
    LLMPrecomputeInputBatch,
    LLMWorker,
    LLMWorkerInput,
    LLMWorkerInputBatch,
)
from helium.runtime.worker.util_worker import AsyncFnExecutor
from helium.runtime.worker.worker import Worker, WorkerManager
from helium.runtime.worker.worker_input import (
    ResultPuller,
    ResultStreamIterator,
    WorkerArg,
    WorkerInput,
    WorkerInputBatch,
    WorkerRequest,
)
from helium.utils import run_coroutine_blocking, unique_id
from helium.utils.prefix.radix_tree import (
    StaticPrefixType,
    TemplatedRadixTree,
    to_raw_prefix,
)


class ProcessorOutput:
    def __init__(
        self, outputs: dict[str, Any], errors: dict[str, BaseException] | None
    ) -> None:
        self.outputs = outputs
        self.error_info: list[dict[str, Any]] | None = (
            None
            if errors is None
            else [
                {"sources": sources.split(","), "details": repr(e)}
                for sources, e in errors.items()
            ]
        )

    @classmethod
    def merge(cls, outputs: Iterable["ProcessorOutput"]) -> "ProcessorOutput":
        merged_outputs: dict[str, Any] = {}
        merged_error_infos: list[dict[str, Any]] | None = None
        for output in outputs:
            merged_outputs.update(output.outputs)
            if output.error_info is not None:
                if merged_error_infos is None:
                    merged_error_infos = []
                merged_error_infos.extend(output.error_info)
        merged = cls(merged_outputs, None)
        merged.error_info = merged_error_infos
        return merged

    @property
    def has_error(self) -> bool:
        return self.error_info is not None


class LLMSchedulingMethod(enum.Enum):
    CAS = enum.auto()
    BATCH_WISE = enum.auto()
    OP_WISE = enum.auto()
    OP_WISE_NO_WAIT = enum.auto()


_EAGERLY_EXECUTABLE_OPS: set[type[Op]] = {
    AppendMessageOp,
    CacheFetchOp,
    ConcatOp,
    DataOp,
    FormatOp,
    InputOp,
    LastMessageOp,
    MergeOp,
    MessageOp,
    SliceOp,
    SwitchOp,
}


OpType = TypeVar("OpType", bound=Op)


class OpMeta(Generic[OpType]):
    __slots__ = ("op", "op_id", "op_inputs", "ref_count", "input_slice", "is_eager")

    def __init__(
        self,
        op: OpType,
        op_inputs: dict[str, WorkerArg],
        ref_count: int,
        input_slice: Slice,
        is_eager: bool,
    ) -> None:
        self.op = op
        self.op_id = op.id
        self.op_inputs = op_inputs
        self.ref_count = ref_count
        self.input_slice = input_slice
        self.is_eager = is_eager


class OpGroupMeta(Generic[OpType]):
    __slots__ = ("op_type", "_worker", "op_metas")

    def __init__(
        self,
        op_type: type[Op],
        worker: Worker | None,
        op_metas: list[OpMeta[OpType]],
    ) -> None:
        self.op_type = op_type
        self._worker = worker
        self.op_metas = op_metas

    @property
    def worker(self) -> Worker:
        return cast(Worker, self._worker)


class HeliumProcessor(ResultPuller):
    def __init__(
        self,
        llm_service_configs: list[LLMServiceConfig],
        cache_manager_config: CacheManagerConfig,
        request_profiler: RequestProfiler,
        deployment_mode: Literal["dev", "prod"],
        name: str = "Processor",
        logger: Logger | None = None,
        log_level: LogLevel | None = None,
        benchmarking: bool = False,
    ) -> None:
        self.name: str = name
        self.logger: Logger = init_child_logger(self.name, logger, log_level)
        self._deployment_mode = deployment_mode
        self._request_profiler = request_profiler
        self._benchmarking = benchmarking
        self._is_started: bool = False
        self._manager = WorkerManager(
            llm_service_configs,
            cache_manager_config,
            logger=self.logger,
            log_level=log_level,
        )

        # Initialize workers
        self._executor = AsyncFnExecutor(
            self._manager, name="executor", logger=self.logger
        )
        self._util_workers: dict[str, Worker] = {"executor": self._executor}

        # Initialize LLM workers by configs
        # LLM service -> List of LLM workers
        llm_services: dict[str, list[LLMWorker]] = defaultdict(list)
        # LLM worker name -> LLM worker
        llm_workers: dict[str, LLMWorker] = {}
        # LLM worker name -> LLM service info
        llm_worker_info: dict[str, LLMServiceInfo] = {}
        for config in llm_service_configs:
            service_name = LLMWorker.get_llm_service_name(config)
            service_workers = llm_services[service_name]
            worker_name = f"llm-{service_name}-{len(service_workers)}"
            llm_worker = LLMWorker(
                self._manager,
                service_config=config,
                name=worker_name,
                logger=self.logger,
                benchmarking=self._benchmarking,
            )
            service_workers.append(llm_worker)
            llm_workers[llm_worker.name] = llm_worker
            llm_worker_info[llm_worker.name] = config.info
        self._llm_services = dict(llm_services)
        self._llm_workers = llm_workers
        self._llm_worker_info = llm_worker_info
        self._default_llm_service: str | None = (
            llm_service_configs[0].name if len(llm_service_configs) > 0 else None
        )

        self._all_workers = {
            worker.name: worker
            for worker in itertools.chain(
                self._util_workers.values(), self._llm_workers.values()
            )
        }

        # Construct dispatch maps
        self._worker_input_map: dict[type[Op], type[WorkerInput]] = {
            AppendMessageOp: AppendMessageFnInput,
            CacheFetchOp: CacheFetchFnInput,
            ConcatOp: ConcatFnInput,
            DataOp: DataFnInput,
            EnterOp: EnterFnInput,
            ExitOp: ExitFnInput,
            FormatOp: FormatFnInput,
            # FutureOp has no corresponding WorkerInput.
            InputOp: InputFnInput,
            LambdaOp: LambdaFnInput,
            LastMessageOp: LastMessageFnInput,
            LLMChatOp: LLMChatWorkerInput,
            LLMCompletionOp: LLMCompletionWorkerInput,
            MergeOp: MergeFnInput,
            MessageOp: MessageFnInput,
            OutputOp: OutputFnInput,
            SliceOp: SliceFnInput,
            SwitchOp: SwitchFnInput,
        }
        self._dispatch_map: dict[
            type[Op], Callable[..., Awaitable[list[WorkerArg]]]
        ] = {
            AppendMessageOp: self._dispatch_append_message_op,
            CacheFetchOp: self._dispatch_cache_fetch_op,
            ConcatOp: self._dispatch_concat_op,
            DataOp: self._dispatch_data_op,
            EnterOp: self._dispatch_enter_op,
            ExitOp: self._dispatch_exit_op,
            FormatOp: self._dispatch_format_op,
            FutureOp: self._dispatch_future_op,
            InputOp: self._dispatch_input_op,
            LambdaOp: self._dispatch_lambda_op,
            LastMessageOp: self._dispatch_last_message_op,
            LLMChatOp: self._dispatch_llm_chat_op,
            LLMCompletionOp: self._dispatch_llm_completion_op,
            MergeOp: self._dispatch_merge_op,
            MessageOp: self._dispatch_message_op,
            OutputOp: self._dispatch_output_op,
            SliceOp: self._dispatch_slice_op,
            SwitchOp: self._dispatch_switch_op,
        }
        self._worker_map: dict[type[Op], Worker | None] = {
            AppendMessageOp: self._util_workers["executor"],
            CacheFetchOp: self._util_workers["executor"],
            ConcatOp: self._util_workers["executor"],
            DataOp: None,
            EnterOp: self._util_workers["executor"],
            ExitOp: self._util_workers["executor"],
            FormatOp: self._util_workers["executor"],
            # FutureOp has no corresponding Worker.
            InputOp: None,
            LambdaOp: self._util_workers["executor"],
            LastMessageOp: self._util_workers["executor"],
            LLMChatOp: None,  # Assigned by op.
            LLMCompletionOp: None,  # Assigned by op.
            MergeOp: self._util_workers["executor"],
            MessageOp: self._util_workers["executor"],
            OutputOp: self._util_workers["executor"],
            SliceOp: self._util_workers["executor"],
            SwitchOp: self._util_workers["executor"],
        }

    @property
    def is_started(self) -> bool:
        return self._is_started

    @property
    def llm_service_counts(self) -> dict[str, int]:
        """The count of LLM workers for each service."""
        return {
            service_name: len(service_workers)
            for service_name, service_workers in self._llm_services.items()
        }

    @property
    def kv_cache_manager(self) -> KVCacheManager | None:
        return self._manager.kv_cache_manager

    @property
    def prompt_cache_manager(self) -> PromptCacheManager | None:
        return self._manager.prompt_cache_manager

    async def start(self) -> None:
        if self._is_started:
            self.logger.warning("Processor has already been started.")
            return

        # Initialize the profiler first to allow workers to send profiling events
        if self._benchmarking:
            self._manager.init_profiler()

        # Start the state manager to allow workers to access job states
        await self._manager.start_state_manager()

        self.logger.info("Starting workers.")
        await asyncio.gather(*(worker.start() for worker in self._all_workers.values()))
        self._is_started = True

        # Start the profiler after all workers are started
        if self._benchmarking:
            await self._manager.start_profiling()

        self.logger.info("All workers started.")

    async def close(self) -> None:
        if not self._is_started:
            self.logger.warning("Processor has not been not started.")
            return

        self.logger.info("Terminating workers.")
        self._is_started = False
        # This will send a termination signal to all workers.
        await self._manager.close()
        for worker in self._all_workers.values():
            await worker.join()  # So you only have to wait until they terminate.
        self.logger.info("All workers terminated.")

    def __del__(self) -> None:
        if self._is_started:
            run_coroutine_blocking(self.close())

    def resolve_llm_info(self, info: RequestInfo) -> None:
        info.llm_service_map = self._get_llm_service_map(info.compiled_graph)
        info.llm_partition_counts = self._get_llm_partition_counts(info.llm_service_map)

    # ---- Core processing methods ---- #

    async def profile(self, info: RequestInfo) -> ProcessorOutput:
        """Profiles the compute graph by sampling a portion of the data"""
        self.logger.info("Profiling compute graph.")

        request_id = info.request_id
        llm_service_map = info.llm_service_map
        llm_partition_counts = info.llm_partition_counts
        enable_cache_aware_scheduling = info.enable_cache_aware_scheduling

        query_profiling_config = info.query_profiling_config
        assert query_profiling_config is not None

        self._manager.track_job_state(request_id)

        # Sample disjoint graphs
        sampled_graphs: list[CompiledGraph] = []
        remaining_graphs: list[CompiledGraph] = []
        for graph in info.disjoint_graphs:
            sampled_graph, remaining_graph = sample_disjoint_graph(
                graph, query_profiling_config
            )
            if sampled_graph is not None:
                sampled_graphs.append(sampled_graph)
            remaining_graphs.append(remaining_graph)

        # Partition LLM ops in the sampled graphs
        sampled_graphs, llm_service_map, llm_worker_idx_map, sliced_op_map = (
            self._partition_llm_ops(
                sampled_graphs, llm_service_map, llm_partition_counts
            )
        )

        # Profile the sampled graphs
        profiling_tasks: list[asyncio.Task[ProcessorOutput]] = []
        for sampled_graph in sampled_graphs:
            worker_assignment = self._assign_workers(
                sampled_graph, llm_service_map, llm_worker_idx_map
            )
            eager_ops = self._get_eagerly_executable_ops(sampled_graph)
            task = asyncio.create_task(
                self._process_graph(
                    request_id,
                    sampled_graph,
                    worker_assignment,
                    sampled_graph.input_slices,
                    sliced_op_map,
                    eager_ops,
                    enable_cache_aware_scheduling,
                    # Always enable runtime adjustment for profiling
                    enable_cache_aware_scheduling,
                    True,
                    None,
                )
            )
            profiling_tasks.append(task)

        # Wait for profiling tasks to finish.
        if profiling_tasks:
            await asyncio.wait(profiling_tasks)

        # Get query profiling results.
        job_state = await self._manager.dump_job_state(
            self._executor.name, request_id, "llm_profiling_info"
        )
        self._manager.untrack_job_state(request_id)

        # Prepare LLM profiling info
        llm_profiling_info: dict[str, LLMProfilingInfo | None] = {}
        to_remap: dict[str, list[LLMProfilingInfo]] = defaultdict(list)
        for op_id, profiling_info in job_state.items():
            og_op_id = sliced_op_map.get(op_id)
            if og_op_id is None:
                llm_profiling_info[op_id] = profiling_info
            else:
                to_remap[og_op_id].append(profiling_info)

        for op_id, profiling_infos in to_remap.items():
            llm_profiling_info[op_id] = LLMProfilingInfo.merge(profiling_infos)

        # Get the outputs of the profiling inputs.
        processor_output = ProcessorOutput.merge(
            [task.result() for task in profiling_tasks]
        )

        # Prepare query profile.
        for graph in remaining_graphs:
            # Update profiling info
            for op in graph.iter_ops(LLMOp):
                if op.id not in llm_profiling_info:
                    llm_profiling_info[op.id] = None
        query_profile = HeliumQueryProfile(llm_profiling_info=llm_profiling_info)

        # Update request info
        info.disjoint_graphs = remaining_graphs
        info.query_profile = query_profile

        return processor_output

    async def process(self, info: RequestInfo) -> tuple[
        ProcessorOutput,
        HeliumSystemProfile,
        PrefixMap | None,
        dict[int, PrefixMap] | None,
    ]:
        """Processes the query

        Returns
        -------
        ProcessorOutput
            Outputs of the compute graph.
        HeliumSystemProfile
            System profiling results of the compute graph.
        PrefixMap | None
            The static prefix map if precomputation was performed, else None.
        dict[int, PrefixMap] | None
            The dynamic prefix map if precomputation was performed, else None.
        """
        # Extract request info
        request_id = info.request_id
        disjoint_graphs = info.disjoint_graphs
        llm_service_map = info.llm_service_map
        llm_partition_counts = info.llm_partition_counts
        enable_cache_aware_scheduling = info.enable_cache_aware_scheduling
        enable_runtime_adjustment = info.enable_runtime_adjustment
        query_profile = info.query_profile
        system_profiling_config = info.system_profiling_config

        self._manager.track_job_state(request_id)

        # Partitions LLM ops in the disjoint graphs
        llm_profiling_info = (
            None if query_profile is None else query_profile.llm_profiling_info
        )
        disjoint_graphs, llm_service_map, llm_worker_idx_map, sliced_op_map = (
            self._partition_llm_ops(
                disjoint_graphs,
                llm_service_map,
                llm_partition_counts,
                llm_profiling_info,
            )
        )
        # Remove redundant ConcatOps and SliceOps
        disjoint_graphs = [
            _remove_redundant_slicing(graph) for graph in disjoint_graphs
        ]
        # Merge disjoint graphs
        compiled_graph = CompiledGraph.merge(disjoint_graphs)
        # Assign workers to the disjoint graphs
        worker_assignment = self._assign_workers(
            compiled_graph, llm_service_map, llm_worker_idx_map
        )
        # Get eagerly executable ops
        eager_ops = self._get_eagerly_executable_ops(compiled_graph)

        # ----- System profiling starts ----- #
        if system_profiling_config is not None:
            self.logger.info("Starting system profiling.")
            await self._start_llm_benchmark(system_profiling_config)

        # ----- Query processing starts ----- #
        self.logger.info("Processing compute graph.")
        processor_output = await self._process_graph(
            request_id,
            compiled_graph,
            worker_assignment,
            compiled_graph.input_slices,
            sliced_op_map,
            eager_ops,
            enable_cache_aware_scheduling,
            enable_runtime_adjustment,
            False,
            query_profile,
        )
        # ----- Query processing ends ----- #

        # Handle system profiling.
        system_profile: HeliumSystemProfile
        if self._benchmarking:
            # Need to do this regardless of profiling to clear the profiling data.
            op_ids = [op.id for op in compiled_graph.iter_ops()]
            system_profile = await self._manager.get_profiling_results(op_ids)
        else:
            system_profile = {}
        # ----- System profiling ends ----- #

        if system_profiling_config is not None:
            # LLM benchmark
            system_profile["llm_benchmark"] = await self._stop_llm_benchmark(
                system_profiling_config
            )

            # System profiles
            if "task_profile" in system_profile:
                # Map op ids to op types for readability
                op_id_map = {
                    op.id: op.__class__.__name__ for op in compiled_graph.iter_ops()
                }
                system_profile["task_profile"] = {
                    f"{op_id_map[k]}-{k}": v
                    for k, v in system_profile["task_profile"].items()
                }

        static_map: PrefixMap | None
        dynamic_map: dict[int, PrefixMap] | None
        if info.precompute_cacheable_inputs:
            # Precompute cacheable inputs and store them in the KV cache
            precompute_output, static_map, dynamic_map = (
                await self._precompute_cacheable_inputs(info)
            )
            processor_output = ProcessorOutput.merge(
                [processor_output, precompute_output]
            )
            if self._benchmarking:
                # Need to do this regardless of profiling to clear the profiling data.
                op_ids = [op.id for op in compiled_graph.iter_ops()]
                await self._manager.get_profiling_results(op_ids)
        else:
            static_map = dynamic_map = None

        self._manager.untrack_job_state(request_id)

        return processor_output, system_profile, static_map, dynamic_map

    # ----- Cache precomputation methods ----- #

    async def precompute_kv_cache(
        self, info: RequestInfo
    ) -> tuple[ProcessorOutput, PrefixMap]:
        """Precomputes the static prefixes in the compute graph and stores them in the KV cache"""
        if self._deployment_mode == "dev":
            # Clear the KV cache to avoid interference from unrelated cache entries
            await self.reset_prefix_cache()
            await self._clear_kv_cache()

        self.logger.info("Precomputing KV cache for static prefixes.")

        request_id = info.request_id
        self._manager.track_job_state(request_id)
        processor_output, prefix_map = await self._precompute_static_prefixes(info)
        self._manager.untrack_job_state(request_id)

        return processor_output, prefix_map

    async def precompute_prefixes(self, prefix_map: PrefixMap) -> ProcessorOutput:
        """Precomputes the prefixes in the KV cache"""
        if self._deployment_mode == "dev":
            # Clear the KV cache to avoid interference from unrelated cache entries
            await self.reset_prefix_cache()
            await self._clear_kv_cache()

        self.logger.info("Precomputing KV cache for the provided prefixes.")

        start_time = time.perf_counter()
        request_id = unique_id()
        self._manager.track_job_state(request_id)
        processor_output = await self._precompute_prefixes(request_id, prefix_map)
        self._manager.untrack_job_state(request_id)
        elapsed_time = time.perf_counter() - start_time

        self.logger.info("Precomputation completed (%.5f seconds).", elapsed_time)

        return processor_output

    async def reset_prefix_cache(self) -> None:
        """Resets the prefix cache of all LLM workers"""
        self.logger.info("Resetting prefix cache of all LLM workers.")
        for llm_worker in self._llm_workers.values():
            await llm_worker.reset_prefix_cache()

    async def reset_proactive_cache(self) -> None:
        """Resets the proactive KV and prompt cache of all LLM workers"""
        self.logger.info("Resetting proactive KV and prompt cache.")
        if self.kv_cache_manager is not None:
            # Clear the proactive KV cache
            for llm_worker in self._llm_workers.values():
                await llm_worker.clear_kv_cache()
        if self.prompt_cache_manager is not None:
            # Clear the proactive prompt cache
            self.prompt_cache_manager.clear()

    async def change_llm_workers_kv_role(self, new_role: str) -> None:
        """Changes the KV cache role of all LLM workers"""
        self.logger.info(f"Changing KV cache role of all LLM workers to '{new_role}'.")
        for llm_worker in self._llm_workers.values():
            await llm_worker.change_kv_role(new_role)

    # ----- ResultPuller interface methods ----- #

    def result_iterator(
        self, worker_name: str, looping: bool, request: WorkerRequest
    ) -> ResultStreamIterator:
        return self._manager.result_iterator(worker_name, looping, request)

    async def unsubscribe(self, worker_name: str, unsub_request: WorkerRequest) -> None:
        return await self._manager.unsubscribe(worker_name, unsub_request)

    def start_profiling_task(
        self, worker_name: str, op_id: str, iteration: int
    ) -> None:
        self._manager.start_profiling_task(worker_name, op_id, iteration)

    async def stop_profiling_task(
        self, worker_name: str, op_id: str, iteration: int
    ) -> None:
        await self._manager.stop_profiling_task(worker_name, op_id, iteration)

    def start_profiling_range(self, worker_name: str, range_id: str) -> None:
        self._manager.start_profiling_range(worker_name, range_id)

    async def stop_profiling_range(self, worker_name: str, range_id: str) -> None:
        await self._manager.stop_profiling_range(worker_name, range_id)

    async def get_cached_results(
        self, inp: WorkerInput, keys: Sequence[Hashable]
    ) -> list[str | None] | list[list[Message] | None]:
        return self._manager.batch_query_cache(inp, keys)

    async def cache_results(
        self,
        inp: WorkerInput,
        batch: dict[Hashable, str] | dict[Hashable, list[Message]],
        overwrite: bool = False,
    ) -> None:
        self._manager.batch_store_cache(inp, batch, overwrite)

    # ---- LLM-related helper methods ---- #

    def _get_llm_model_map(self, compiled_graph: CompiledGraph) -> dict[str, str]:
        """Resolves LLM models for the LLM ops in the compute graph

        Returns
        -------
        dict[str, str]
            Mapping from LLM op ID to the resolved LLM model name.
        """
        return {
            llm_op.id: llm_op.config.model for llm_op in compiled_graph.iter_ops(LLMOp)
        }

    def _get_llm_service_map(self, compiled_graph: CompiledGraph) -> dict[str, str]:
        """Resolves LLM services for the LLM ops in the compute graph

        Returns
        -------
        dict[str, str]
            Mapping from LLM op ID to the resolved LLM service name.
        """
        return {
            llm_op.id: self._resolve_llm_service(llm_op.config)
            for llm_op in compiled_graph.iter_ops(LLMOp)
        }

    def _get_llm_partition_counts(
        self, llm_service_map: dict[str, str]
    ) -> dict[str, int]:
        """Calculates the number of partitions for each LLM op in the compute graph

        Parameters
        ----------
        llm_service_map : dict[str, str]
            Mapping from LLM op ID to the LLM service name.

        Returns
        -------
        dict[str, int]
            Mapping from LLM op ID to the number of partitions for that op.
        """
        llm_service_counts = self.llm_service_counts
        partition_counts: dict[str, int] = {}
        for llm_op_id, service_name in llm_service_map.items():
            service_count = llm_service_counts.get(service_name)
            if service_count is None:
                raise ValueError(f"LLM service '{service_name}' not found.")
            partition_counts[llm_op_id] = service_count
        return partition_counts

    def _partition_llm_ops(
        self,
        disjoint_graphs: list[CompiledGraph],
        llm_service_map: dict[str, str],
        llm_partition_counts: dict[str, int],
        llm_profiling_info: dict[str, LLMProfilingInfo | None] | None = None,
    ) -> tuple[list[CompiledGraph], dict[str, str], dict[str, int], dict[str, str]]:
        """Partitions LLM ops in the disjoint graphs

        Returns
        -------
        list[CompiledGraph]
            List of disjoint graphs with partitioned LLM ops.
        dict[str, str]
            New mapping from LLM op ID to the LLM service name.
        dict[str, int]
            Mapping from LLM op ID to its partition index.
        dict[str, str]
            Mapping from newly created LLM op ID to the original LLM op ID.
        """
        new_disjoint_graphs: list[CompiledGraph] = []
        new_llm_service_map: dict[str, str] = {}
        worker_idx_map: dict[str, int] = {}
        llm_op_map: dict[str, str] = {}
        for disjoint_graph in disjoint_graphs:
            dependencies = disjoint_graph.dependencies()
            input_slices = disjoint_graph.input_slices
            to_recompile = False
            for llm_op in disjoint_graph.iter_ops(LLMOp):
                llm_op_id = llm_op.id
                llm_partition_count = llm_partition_counts[llm_op_id]
                llm_service_name = llm_service_map[llm_op_id]
                if llm_partition_count <= 1:
                    # Skip partitioning
                    worker_idx_map[llm_op_id] = 0
                    new_llm_service_map[llm_op_id] = llm_service_name
                    continue
                to_recompile = True
                # Partition the LLM op
                new_llm_ops = partition_op(
                    llm_op,
                    input_slices[llm_op_id],
                    llm_partition_counts[llm_op_id],
                    dependencies,
                )
                if llm_profiling_info is not None:
                    # Update profiling info
                    op_profiling_info = llm_profiling_info.pop(llm_op_id)
                    for new_llm_op in new_llm_ops:
                        llm_profiling_info[new_llm_op.id] = op_profiling_info
                # Update LLM service map, worker index map, and LLM op map
                for i, new_llm_op in enumerate(new_llm_ops):
                    new_llm_op_id = new_llm_op.id
                    new_llm_service_map[new_llm_op_id] = llm_service_name
                    worker_idx_map[new_llm_op_id] = i
                    llm_op_map[new_llm_op_id] = llm_op_id
            new_disjoint_graphs.append(
                disjoint_graph.recompile(check_inputs=False)
                if to_recompile
                else disjoint_graph
            )

        return new_disjoint_graphs, new_llm_service_map, worker_idx_map, llm_op_map

    # ---- Helper methods for precomputation ---- #

    async def _clear_kv_cache(self) -> None:
        for llm_worker in self._llm_workers.values():
            await llm_worker.clear_kv_cache()

    async def _precompute_prefixes(
        self, request_id: str, prefix_map: PrefixMap
    ) -> ProcessorOutput:
        # Create inputs for each LLM worker
        op_ids: list[str] = []
        worker_inputs: dict[LLMWorker, list[LLMPrecomputeInput]] = defaultdict(list)
        for (llm_service, model, base_url, api_key), prompts in prefix_map.items():
            generation_config = GenerationConfig(
                model=model, base_url=base_url, api_key=api_key, max_tokens=1
            )
            service_name = LLMWorker.service_name_with_model(llm_service, model)
            for worker in self._llm_services[service_name]:
                op_id = unique_id()
                op_ids.append(op_id)
                worker_inputs[worker].append(
                    LLMPrecomputeInput(
                        request_id=request_id,
                        op_id=op_id,
                        is_eager=True,
                        ref_count=1,
                        generation_config=generation_config,
                        prompts=prompts,
                    )
                )

        # Dispatch precompute tasks
        tasks: list[asyncio.Task[list[WorkerArg]]] = []
        for worker, inputs in worker_inputs.items():
            task = asyncio.create_task(
                worker.add_task(LLMPrecomputeInputBatch(inputs), puller=self._executor)
            )
            tasks.append(task)

        if tasks:
            args = await asyncio.gather(*tasks)
            has_error = any(arg.is_dead() for arg in itertools.chain(*args))
        else:
            has_error = False

        # Fetch errors, if any
        errors = (
            await self._manager.dump_job_state(self._executor.name, request_id, "error")
            if has_error
            else None
        )
        processor_output = ProcessorOutput({}, errors)

        if self._benchmarking:
            # Need to do this regardless of profiling to clear the profiling data.
            await self._manager.get_profiling_results(op_ids)

        return processor_output

    async def _precompute_static_prefixes(
        self, info: RequestInfo
    ) -> tuple[ProcessorOutput, PrefixMap]:
        def get_prefix_key(config: GenerationConfig) -> PrefixKey:
            llm_service = config.llm_service or self._default_llm_service
            if llm_service is None:
                raise ValueError("No LLM service available.")
            return (llm_service, config.model, config.base_url, config.api_key)

        request_id = info.request_id
        compiled_graph = info.compiled_graph
        llm_service_map = info.llm_service_map

        # Assign workers to the compute graph
        worker_assignment = self._assign_workers(
            compiled_graph,
            llm_service_map,
            {op.id: 0 for op in compiled_graph.iter_ops(LLMOp)},
        )

        # Build templated radix tree
        radix_tree = compiled_graph.build_radix_tree(worker_assignment)
        static_prefixes = radix_tree.get_static_prefixes()

        graph_dict = compiled_graph.graph.as_dict()
        prefix_groups: dict[PrefixKey, set[StaticPrefixType]] = defaultdict(set)
        for op_id, prefix in static_prefixes.items():
            op = graph_dict[op_id]
            assert isinstance(op, LLMOp)
            prefix_key = get_prefix_key(op.config)
            prefix_groups[prefix_key].add(prefix)

        prefix_map: PrefixMap = {
            prefix_key: [to_raw_prefix(prefix) for prefix in prefixes]
            for prefix_key, prefixes in prefix_groups.items()
        }

        processor_output = await self._precompute_prefixes(request_id, prefix_map)

        return processor_output, dict(prefix_map)

    async def _precompute_dynamic_prefixes(
        self, info: RequestInfo
    ) -> tuple[ProcessorOutput, dict[int, PrefixMap]]:
        request_id = info.request_id

        # (worker_name, model, base_url, api_key) -> list of prompts
        cacheable_inputs: dict[str, CacheItemType] = await self._manager.dump_job_state(
            self._executor.name, request_id, "llm_cacheable"
        )
        if not cacheable_inputs:
            return ProcessorOutput({}, None), {}

        # Group inputs by (service_name, model, base_url, api_key)
        prefix_map: PrefixMap = defaultdict(list)
        query_prefix_map: dict[int, PrefixMap] = defaultdict(dict)
        for item in cacheable_inputs.values():
            llm_service, i, content, generation_config = item
            key = (
                llm_service,
                generation_config.model,
                generation_config.base_url,
                generation_config.api_key,
            )
            prefix_map[key].append(content)
            query_prefix_map[i].setdefault(key, []).append(content)

        processor_output = await self._precompute_prefixes(request_id, prefix_map)

        return processor_output, dict(query_prefix_map)

    async def _precompute_cacheable_inputs(
        self, info: RequestInfo
    ) -> tuple[ProcessorOutput, PrefixMap, dict[int, PrefixMap]]:
        """Precomputes cacheable LLM inputs and stores them in the KV cache."""
        if self._deployment_mode == "dev":
            # Clear the KV cache to avoid interference from unrelated cache entries
            await self.reset_prefix_cache()
            await self._clear_kv_cache()

        self.logger.info("Precomputing cacheable inputs.")

        static_output, static_map = await self._precompute_static_prefixes(info)
        dynamic_output, dynamic_map = await self._precompute_dynamic_prefixes(info)
        # Merge processor outputs
        processor_output = ProcessorOutput.merge([static_output, dynamic_output])
        return processor_output, static_map, dynamic_map

    # ---- Helper methods for processing ---- #

    def _assign_workers(
        self,
        compiled_graph: CompiledGraph,
        llm_service_map: dict[str, str],
        llm_worker_idx_map: dict[str, int],
    ) -> dict[str, str | None]:
        worker_assignment: dict[str, str | None] = {}
        for op in compiled_graph.iter_ops():
            worker: Worker | None
            if isinstance(op, LLMOp):
                service_name = llm_service_map[op.id]
                service_idx = llm_worker_idx_map[op.id]
                worker = self._llm_services[service_name][service_idx]
            else:
                # Fetch worker from the worker map.
                op_type = type(op.op) if isinstance(op, FutureOp) else type(op)
                if op_type in self._worker_map:
                    worker = self._worker_map[op_type]
                else:
                    raise NotImplementedError(f"No worker for op: {op.serialize()}")
            worker_assignment[op.id] = None if worker is None else worker.name

        return worker_assignment

    def _get_eagerly_executable_ops(self, compiled_graph: CompiledGraph) -> set[Op]:
        eager_ops: set[Op] = set()
        for op in compiled_graph.iter_ops():
            if type(op) in _EAGERLY_EXECUTABLE_OPS and all(
                inp_op in eager_ops for inp_op in op.inputs
            ):
                eager_ops.add(op)
        return eager_ops

    @log_on_exception_async()
    async def _process_graph(
        self,
        request_id: str,
        compiled_graph: CompiledGraph,
        worker_assignment: dict[str, str | None],
        input_slices: dict[str, Slice],
        sliced_op_map: dict[str, str],
        eager_ops: set[Op],
        enable_cache_aware_scheduling: bool,
        enable_runtime_adjustment: bool,
        profiling: bool,
        query_profile: HeliumQueryProfile | None,
    ) -> ProcessorOutput:
        graph, inputs = compiled_graph.graph, compiled_graph.inputs
        graph_dict = graph.as_dict()
        dependencies = graph.dependencies()  # Mapping from nodes to dependent nodes
        # Ignore CacheFetchOp in dependency tracking
        dependencies = {
            op: {dep for dep in deps if not isinstance(dep, CacheFetchOp)}
            for op, deps in dependencies.items()
        }
        input_tracker: dict[str, int] = {}
        op_inputs: dict[str, dict[str, WorkerArg]] = {}
        can_execute: list[Op] = []
        output_tasks: dict[str, asyncio.Task[Data | None]] = {}
        prereg_task_ids: dict[str, str] = {}
        llm_dispatch_map: dict[Worker, list[LLMWorkerInput]] = {}

        for op_id, op in graph_dict.items():
            n = 0 if isinstance(op, CacheFetchOp) else len(op.inputs)
            input_tracker[op_id] = n
            if n == 0:
                can_execute.append(op)

        for name, value in inputs.items():
            op_id = graph.input_ops[name].id
            op_inputs[op_id] = {
                "input": WorkerArg(op_id=op_id, data=Data.text(value, indices=None))
            }

        pending_tasks: set[asyncio.Task[dict[str, WorkerArg]]] = set()
        while can_execute or pending_tasks:
            pending_tasks.update(
                asyncio.create_task(
                    self._dispatch_sequential(
                        request_id,
                        worker_assignment,
                        [
                            OpMeta(
                                op=op,
                                op_inputs=op_inputs.get(op.id, {}),
                                ref_count=(
                                    1
                                    if isinstance(op, OutputOp)
                                    else len(dependencies.get(op, ""))
                                ),
                                input_slice=input_slices[op.id],
                                is_eager=op in eager_ops,
                            )
                            for op in ops
                        ],
                        prereg_task_ids,
                        llm_dispatch_map,
                    )
                )
                for ops in self._get_ops_to_dispatch(can_execute)
            )
            finished_tasks, pending_tasks = await asyncio.wait(
                pending_tasks, return_when=asyncio.FIRST_COMPLETED
            )

            for task in finished_tasks:
                op_outputs: dict[str, WorkerArg] = task.result()
                for op_id, op_out in op_outputs.items():
                    op = graph_dict[op_id]
                    if op not in dependencies:
                        if isinstance(op, OutputOp):
                            worker_name = worker_assignment[op_id]
                            worker = (
                                self._executor
                                if worker_name is None
                                else self._all_workers[worker_name]
                            )
                            output_tasks[op.name] = asyncio.create_task(
                                op_out.resolve(worker, "output", looping=False)
                            )
                        elif isinstance(op, FutureOp):
                            pass
                        else:
                            raise ValueError(f"Unused op: {op.serialize()}")
                        continue
                    for dep in dependencies[op]:
                        dep_id = dep.id
                        if dep_id not in op_inputs:
                            op_inputs[dep_id] = {}
                        # Ensure that op executions do not share input objects.
                        op_inputs[dep_id][op_id] = op_out.copy()
                        input_tracker[dep_id] -= 1
                        if input_tracker[dep_id] == 0:
                            can_execute.append(dep)

        if llm_dispatch_map:
            if enable_cache_aware_scheduling:
                # Build radix trees for cache-aware scheduling
                scheduling_method = LLMSchedulingMethod.CAS
                radix_tree = graph.build_radix_tree(worker_assignment, sliced_op_map)
            else:
                # Default to batch-wise scheduling
                scheduling_method = LLMSchedulingMethod.BATCH_WISE
                radix_tree = None
            schedules = await self._schedule_llm_ops(
                scheduling_method, llm_dispatch_map, radix_tree, query_profile
            )

            # Determine dispatch mode
            if scheduling_method is LLMSchedulingMethod.OP_WISE_NO_WAIT:
                dispatch_mode = LLMDispatchMode.OP_WISE_NO_WAIT
            elif (
                scheduling_method is LLMSchedulingMethod.BATCH_WISE
                or enable_runtime_adjustment
            ):
                dispatch_mode = LLMDispatchMode.RUNTIME_ADJUSTMENT
            else:
                dispatch_mode = LLMDispatchMode.NO_RUNTIME_ADJUSTMENT

            # Dispatch LLM tasks
            await self._dispatch_llm_worker_input_batch(
                llm_dispatch_map,
                prereg_task_ids,
                schedules,
                dispatch_mode,
                profiling,
            )

        # Pull outputs
        # output name -> output data
        outputs: dict[str, list[str] | list[list[dict[str, str]]] | None] = {}
        has_error: bool = False
        for name, output_task in output_tasks.items():
            task_out = await output_task
            if task_out is None:
                outputs[name] = None
                has_error = True
            elif task_out.is_text():
                outputs[name] = task_out.sort().as_text()
            else:
                outputs[name] = [
                    [msg.to_dict() for msg in msgs]
                    for msgs in task_out.sort().as_message()
                ]

        # Fetch errors, if any
        errors = (
            await self._manager.dump_job_state(self._executor.name, request_id, "error")
            if has_error
            else None
        )
        processor_output = ProcessorOutput(outputs, errors)

        return processor_output

    def _get_ops_to_dispatch(self, ops: list[Op]) -> list[list[Op]]:
        # TODO: Implement a scheduling algorithm
        ret = [ops.copy()]
        ops.clear()
        return ret

    def _group_ops(
        self, op_metas: list[OpMeta[Op]], worker_assignment: dict[str, str | None]
    ) -> list[OpGroupMeta[Op]]:
        type_map: dict[tuple[type[Op], Worker | None], list[OpMeta[Op]]] = {}
        for op_meta in op_metas:
            op_type = type(op_meta.op)
            worker_name = worker_assignment[op_meta.op_id]
            worker = None if worker_name is None else self._all_workers[worker_name]
            key = (op_type, worker)
            if key in type_map:
                type_map[key].append(op_meta)
            else:
                type_map[key] = [op_meta]
        return [
            OpGroupMeta(op_type, worker, grouped_op_metas)
            for (op_type, worker), grouped_op_metas in type_map.items()
        ]

    async def _dispatch_sequential(
        self,
        request_id: str,
        worker_assignment: dict[str, str | None],
        op_metas: list[OpMeta[Op]],
        prereg_task_ids: dict[str, str],
        llm_dispatch_map: dict[Worker, list[LLMWorkerInput]],
    ) -> dict[str, WorkerArg]:
        op_outputs: dict[str, WorkerArg] = {}
        for op_group_meta in self._group_ops(op_metas, worker_assignment):
            op_out = await self._dispatch_op(
                request_id, op_group_meta, prereg_task_ids, llm_dispatch_map
            )
            for op_meta, out in zip(op_group_meta.op_metas, op_out):
                op_outputs[op_meta.op_id] = out
        return op_outputs

    async def _dispatch_op(
        self,
        request_id: str,
        op_group_meta: OpGroupMeta[Op],
        prereg_task_ids: dict[str, str],
        llm_dispatch_map: dict[Worker, list[LLMWorkerInput]],
    ) -> list[WorkerArg]:
        op_type = op_group_meta.op_type
        if op_type in self._dispatch_map:
            return await self._dispatch_map[op_type](
                request_id, op_group_meta, prereg_task_ids, llm_dispatch_map
            )
        raise NotImplementedError(
            f"Unsupported op: {op_group_meta.op_metas[0].op.serialize()}"
        )

    def _resolve_llm_service(self, config: GenerationConfig) -> str:
        llm_service = config.llm_service or self._default_llm_service
        if llm_service is None:
            raise ValueError("No LLM service available.")
        return LLMWorker.service_name_with_model(llm_service, config.model)

    def _get_worker_input_type(self, op: Op | None) -> type[WorkerInput]:
        assert op is not None
        assert not isinstance(op, FutureOp)
        op_type = type(op)
        if op_type not in self._worker_input_map:
            raise ValueError(f"No worker input for op: {op.serialize()}")
        return self._worker_input_map[op_type]

    async def _add_task(
        self,
        worker: Worker,
        worker_inputs: WorkerInputBatch | FnInputBatch,
        prereg_task_ids: dict[str, str],
    ) -> list[WorkerArg]:
        for worker_inp in worker_inputs:
            task_id = prereg_task_ids.pop(worker_inp.op_id, None)
            if task_id is not None:
                worker_inp.task_id = task_id
        return await worker.add_task(worker_inputs)

    def _preregister_op(
        self, worker: Worker, op: Op, prereg_task_ids: dict[str, str]
    ) -> WorkerArg:
        input_type = self._get_worker_input_type(op)
        task_id = worker.preregister(input_type)
        if op.id in prereg_task_ids:
            raise ValueError(f"Op '{op.serialize()}' has already been preregistered.")
        prereg_task_ids[op.id] = task_id
        return WorkerArg(op_id=op.id, src_worker=worker.name, src_task_id=task_id)

    async def _dispatch_data_op(
        self,
        request_id: str,
        op_group_meta: OpGroupMeta[DataOp],
        prereg_task_ids: dict[str, str],
        llm_dispatch_map: dict[Worker, list[LLMWorkerInput]],
    ) -> list[WorkerArg]:
        # Synchronous operation
        self.logger.debug("Dispatch data op.")
        outputs = [
            WorkerArg(op_id=op_meta.op_id, data=data)
            for op_meta in op_group_meta.op_metas
            async for data in (
                DataFnInput(
                    request_id,
                    op_meta.op_id,
                    op_meta.is_eager,
                    op_meta.ref_count,
                    op_meta.op.data,
                ).run(self)
            )
        ]
        return outputs

    async def _dispatch_input_op(
        self,
        request_id: str,
        op_group_meta: OpGroupMeta[InputOp],
        prereg_task_ids: dict[str, str],
        llm_dispatch_map: dict[Worker, list[LLMWorkerInput]],
    ) -> list[WorkerArg]:
        # Synchronous operation
        self.logger.debug("Dispatch input op.")
        outputs = [
            WorkerArg(op_id=op_meta.op_id, data=data)
            for op_meta in op_group_meta.op_metas
            async for data in (
                InputFnInput(
                    request_id,
                    op_meta.op_id,
                    op_meta.is_eager,
                    op_meta.ref_count,
                    op_meta.op_inputs["input"].data.as_text(),
                ).run(self)
            )
        ]
        return outputs

    async def _dispatch_output_op(
        self,
        request_id: str,
        op_group_meta: OpGroupMeta[OutputOp],
        prereg_task_ids: dict[str, str],
        llm_dispatch_map: dict[Worker, list[LLMWorkerInput]],
    ) -> list[WorkerArg]:
        self.logger.debug("Dispatch output op.")
        executor_inputs = [
            OutputFnInput(
                request_id,
                op_meta.op_id,
                op_meta.is_eager,
                op_meta.ref_count,
                op_meta.op_inputs[op_meta.op.inputs[0].id],
            )
            for op_meta in op_group_meta.op_metas
        ]
        return await self._add_task(
            op_group_meta.worker, FnInputBatch(executor_inputs), prereg_task_ids
        )

    async def _dispatch_message_op(
        self,
        request_id: str,
        op_group_meta: OpGroupMeta[MessageOp],
        prereg_task_ids: dict[str, str],
        llm_dispatch_map: dict[Worker, list[LLMWorkerInput]],
    ) -> list[WorkerArg]:
        self.logger.debug("Dispatch message op.")
        worker_inputs = []
        for op_meta in op_group_meta.op_metas:
            op = op_meta.op
            op_inputs = op_meta.op_inputs
            worker_inputs.append(
                MessageFnInput(
                    request_id=request_id,
                    op_id=op_meta.op_id,
                    is_eager=op_meta.is_eager,
                    ref_count=op_meta.ref_count,
                    max_iter=op.max_iter,
                    roles=op.roles,
                    message_refs=op.message_refs,
                    inputs=[op_inputs[inp_op.id] for inp_op in op.inputs],
                )
            )
        return await self._add_task(
            op_group_meta.worker, FnInputBatch(worker_inputs), prereg_task_ids
        )

    async def _dispatch_append_message_op(
        self,
        request_id: str,
        op_group_meta: OpGroupMeta[AppendMessageOp],
        prereg_task_ids: dict[str, str],
        llm_dispatch_map: dict[Worker, list[LLMWorkerInput]],
    ) -> list[WorkerArg]:
        def get_content(
            op: AppendMessageOp, op_inputs: dict[str, WorkerArg]
        ) -> str | WorkerArg:
            op_content = op.content
            if isinstance(op_content, str):
                return op_content
            return op_inputs[op_content.id]

        self.logger.debug("Dispatch append-message op.")
        worker_inputs = []
        for op_meta in op_group_meta.op_metas:
            op = op_meta.op
            op_inputs = op_meta.op_inputs
            worker_inputs.append(
                AppendMessageFnInput(
                    request_id=request_id,
                    op_id=op_meta.op_id,
                    is_eager=op_meta.is_eager,
                    ref_count=op_meta.ref_count,
                    max_iter=op.max_iter,
                    messages=op_inputs[op.messages.id],
                    content=get_content(op, op_inputs),
                    role=op.role,
                )
            )
        return await self._add_task(
            op_group_meta.worker, FnInputBatch(worker_inputs), prereg_task_ids
        )

    async def _dispatch_last_message_op(
        self,
        request_id: str,
        op_group_meta: OpGroupMeta[LastMessageOp],
        prereg_task_ids: dict[str, str],
        llm_dispatch_map: dict[Worker, list[LLMWorkerInput]],
    ) -> list[WorkerArg]:
        self.logger.debug("Dispatch append-message op.")
        worker_inputs = [
            LastMessageFnInput(
                request_id=request_id,
                op_id=op_meta.op_id,
                is_eager=op_meta.is_eager,
                ref_count=op_meta.ref_count,
                max_iter=op_meta.op.max_iter,
                messages=op_meta.op_inputs[op_meta.op.messages.id],
            )
            for op_meta in op_group_meta.op_metas
        ]
        return await self._add_task(
            op_group_meta.worker, FnInputBatch(worker_inputs), prereg_task_ids
        )

    async def _dispatch_format_op(
        self,
        request_id: str,
        op_group_meta: OpGroupMeta[FormatOp],
        prereg_task_ids: dict[str, str],
        llm_dispatch_map: dict[Worker, list[LLMWorkerInput]],
    ) -> list[WorkerArg]:
        self.logger.debug("Dispatch format op.")
        worker_inputs = []
        for op_meta in op_group_meta.op_metas:
            op = op_meta.op
            op_inputs = op_meta.op_inputs
            worker_inputs.append(
                FormatFnInput(
                    request_id=request_id,
                    op_id=op_meta.op_id,
                    is_eager=op_meta.is_eager,
                    ref_count=op_meta.ref_count,
                    max_iter=op.max_iter,
                    template=op.template,
                    format_args=[op_inputs[arg.id] for arg in op.format_args],
                    format_kwargs={
                        k: op_inputs[v.id] for k, v in op.format_kwargs.items()
                    },
                )
            )
        return await self._add_task(
            op_group_meta.worker, FnInputBatch(worker_inputs), prereg_task_ids
        )

    async def _dispatch_future_op(
        self,
        request_id: str,
        op_group_meta: OpGroupMeta[FutureOp],
        prereg_task_ids: dict[str, str],
        llm_dispatch_map: dict[Worker, list[LLMWorkerInput]],
    ) -> list[WorkerArg]:
        self.logger.debug("Dispatch future op.")
        worker = op_group_meta.worker
        out = [
            self._preregister_op(worker, op_meta.op.op, prereg_task_ids)
            for op_meta in op_group_meta.op_metas
        ]
        return out

    async def _dispatch_lambda_op(
        self,
        request_id: str,
        op_group_meta: OpGroupMeta[LambdaOp],
        prereg_task_ids: dict[str, str],
        llm_dispatch_map: dict[Worker, list[LLMWorkerInput]],
    ) -> list[WorkerArg]:
        self.logger.debug("Dispatch lambda op.")
        worker_inputs = []
        for op_meta in op_group_meta.op_metas:
            op = op_meta.op
            op_inputs = op_meta.op_inputs
            worker_inputs.append(
                LambdaFnInput(
                    request_id=request_id,
                    op_id=op_meta.op_id,
                    is_eager=op_meta.is_eager,
                    ref_count=op_meta.ref_count,
                    max_iter=op.max_iter,
                    inputs=[op_inputs[op_inp.id] for op_inp in op.inputs],
                    fn=op.fn,
                )
            )
        return await self._add_task(
            op_group_meta.worker, FnInputBatch(worker_inputs), prereg_task_ids
        )

    async def _dispatch_slice_op(
        self,
        request_id: str,
        op_group_meta: OpGroupMeta[SliceOp],
        prereg_task_ids: dict[str, str],
        llm_dispatch_map: dict[Worker, list[LLMWorkerInput]],
    ) -> list[WorkerArg]:
        self.logger.debug("Dispatch slice op.")
        worker_inputs = []
        for op_meta in op_group_meta.op_metas:
            op = op_meta.op
            op_inputs = op_meta.op_inputs
            worker_inputs.append(
                SliceFnInput(
                    request_id=request_id,
                    op_id=op_meta.op_id,
                    is_eager=op_meta.is_eager,
                    ref_count=op_meta.ref_count,
                    max_iter=op.max_iter,
                    inp=op_inputs[op.inp.id],
                    indices=op.indices,
                )
            )
        return await self._add_task(
            op_group_meta.worker, FnInputBatch(worker_inputs), prereg_task_ids
        )

    async def _dispatch_concat_op(
        self,
        request_id: str,
        op_group_meta: OpGroupMeta[ConcatOp],
        prereg_task_ids: dict[str, str],
        llm_dispatch_map: dict[Worker, list[LLMWorkerInput]],
    ) -> list[WorkerArg]:
        self.logger.debug("Dispatch concat op.")
        worker_inputs = []
        for op_meta in op_group_meta.op_metas:
            op = op_meta.op
            op_inputs = op_meta.op_inputs
            worker_inputs.append(
                ConcatFnInput(
                    request_id=request_id,
                    op_id=op_meta.op_id,
                    is_eager=op_meta.is_eager,
                    ref_count=op_meta.ref_count,
                    max_iter=op.max_iter,
                    inputs=[op_inputs[op_inp.id] for op_inp in op.inputs],
                )
            )
        return await self._add_task(
            op_group_meta.worker, FnInputBatch(worker_inputs), prereg_task_ids
        )

    async def _dispatch_enter_op(
        self,
        request_id: str,
        op_group_meta: OpGroupMeta[EnterOp],
        prereg_task_ids: dict[str, str],
        llm_dispatch_map: dict[Worker, list[LLMWorkerInput]],
    ) -> list[WorkerArg]:
        self.logger.debug("Dispatch enter op.")
        worker_inputs = []
        for op_meta in op_group_meta.op_metas:
            op = op_meta.op
            op_inputs = op_meta.op_inputs
            worker_inputs.append(
                EnterFnInput(
                    request_id=request_id,
                    op_id=op_meta.op_id,
                    is_eager=op_meta.is_eager,
                    ref_count=op_meta.ref_count,
                    max_iter=op.max_iter,
                    init_inp=op_inputs[op.init_op.id],
                    future_inp=op_inputs[op.future_op.id],
                )
            )
        return await self._add_task(
            op_group_meta.worker, FnInputBatch(worker_inputs), prereg_task_ids
        )

    async def _dispatch_exit_op(
        self,
        request_id: str,
        op_group_meta: OpGroupMeta[ExitOp],
        prereg_task_ids: dict[str, str],
        llm_dispatch_map: dict[Worker, list[LLMWorkerInput]],
    ) -> list[WorkerArg]:
        self.logger.debug("Dispatch exit op.")
        worker_inputs = []
        for op_meta in op_group_meta.op_metas:
            op = op_meta.op
            op_inputs = op_meta.op_inputs
            worker_inputs.append(
                ExitFnInput(
                    request_id=request_id,
                    op_id=op_meta.op_id,
                    is_eager=op_meta.is_eager,
                    ref_count=op_meta.ref_count,
                    inp=op_inputs[op.op.id],
                )
            )
        return await self._add_task(
            op_group_meta.worker, FnInputBatch(worker_inputs), prereg_task_ids
        )

    async def _dispatch_switch_op(
        self,
        request_id: str,
        op_group_meta: OpGroupMeta[SwitchOp],
        prereg_task_ids: dict[str, str],
        llm_dispatch_map: dict[Worker, list[LLMWorkerInput]],
    ) -> list[WorkerArg]:
        self.logger.debug("Dispatch switch op.")
        worker_inputs = []
        for op_meta in op_group_meta.op_metas:
            op = op_meta.op
            op_inputs = op_meta.op_inputs
            worker_inputs.append(
                SwitchFnInput(
                    request_id=request_id,
                    op_id=op_meta.op_id,
                    is_eager=op_meta.is_eager,
                    ref_count=op_meta.ref_count,
                    max_iter=op.max_iter,
                    inp=op_inputs[op.input_op.id],
                    cond_args=[op_inputs[op_inp.id] for op_inp in op.cond_ops],
                    pred=op.pred,
                    branch=op.branch,
                    dead_on_empty=op.dead_on_empty,
                )
            )
        return await self._add_task(
            op_group_meta.worker, FnInputBatch(worker_inputs), prereg_task_ids
        )

    async def _dispatch_merge_op(
        self,
        request_id: str,
        op_group_meta: OpGroupMeta[MergeOp],
        prereg_task_ids: dict[str, str],
        llm_dispatch_map: dict[Worker, list[LLMWorkerInput]],
    ) -> list[WorkerArg]:
        self.logger.debug("Dispatch merge op.")
        worker_inputs = []
        for op_meta in op_group_meta.op_metas:
            op = op_meta.op
            op_inputs = op_meta.op_inputs
            worker_inputs.append(
                MergeFnInput(
                    request_id=request_id,
                    op_id=op_meta.op_id,
                    is_eager=op_meta.is_eager,
                    ref_count=op_meta.ref_count,
                    max_iter=op.max_iter,
                    inputs=[op_inputs[op_inp.id] for op_inp in op.inputs],
                )
            )
        return await self._add_task(
            op_group_meta.worker, FnInputBatch(worker_inputs), prereg_task_ids
        )

    async def _dispatch_llm_completion_op(
        self,
        request_id: str,
        op_group_meta: OpGroupMeta[LLMCompletionOp],
        prereg_task_ids: dict[str, str],
        llm_dispatch_map: dict[Worker, list[LLMWorkerInput]],
    ) -> list[WorkerArg]:
        self.logger.debug("Dispatch LLM completion op.")
        worker = op_group_meta.worker
        outputs: list[WorkerArg] = []
        for op_meta in op_group_meta.op_metas:
            op = op_meta.op
            op_inputs = op_meta.op_inputs
            cur_inp = LLMCompletionWorkerInput(
                request_id=request_id,
                op_id=op_meta.op_id,
                is_eager=op_meta.is_eager,
                ref_count=op_meta.ref_count,
                max_iter=op.max_iter,
                input_slice=op_meta.input_slice,
                generation_config=op.config,
                prompts=op_inputs[op.prompt.id],
                echo=op.echo,
                cacheable=op.cacheable,
            )
            # Pre-register the task
            outputs.append(self._preregister_op(worker, op, prereg_task_ids))
            # Add to LLM dispatch map
            if worker not in llm_dispatch_map:
                llm_dispatch_map[worker] = []
            llm_dispatch_map[worker].append(cur_inp)
        return outputs

    async def _dispatch_llm_chat_op(
        self,
        request_id: str,
        op_group_meta: OpGroupMeta[LLMChatOp],
        prereg_task_ids: dict[str, str],
        llm_dispatch_map: dict[Worker, list[LLMWorkerInput]],
    ) -> list[WorkerArg]:
        self.logger.debug("Dispatch LLM chat op.")
        worker = op_group_meta.worker
        outputs: list[WorkerArg] = []
        for op_meta in op_group_meta.op_metas:
            op = op_meta.op
            op_inputs = op_meta.op_inputs
            cur_inp = LLMChatWorkerInput(
                request_id=request_id,
                op_id=op.id,
                is_eager=op_meta.is_eager,
                ref_count=op_meta.ref_count,
                max_iter=op.max_iter,
                input_slice=op_meta.input_slice,
                generation_config=op.config,
                messages=op_inputs[op.messages.id],
                return_history=op.return_history,
                cacheable=op.cacheable,
            )
            # Pre-register the task
            outputs.append(self._preregister_op(worker, op, prereg_task_ids))
            # Add to LLM dispatch map
            if worker not in llm_dispatch_map:
                llm_dispatch_map[worker] = []
            llm_dispatch_map[worker].append(cur_inp)
        return outputs

    async def _dispatch_cache_fetch_op(
        self,
        request_id: str,
        op_group_meta: OpGroupMeta[CacheFetchOp],
        prereg_task_ids: dict[str, str],
        llm_dispatch_map: dict[Worker, list[LLMWorkerInput]],
    ) -> list[WorkerArg]:
        self.logger.debug("Dispatch cache fetch op.")
        worker_inputs = []
        for op_meta in op_group_meta.op_metas:
            op = op_meta.op
            worker_inputs.append(
                CacheFetchFnInput(
                    request_id=request_id,
                    op_id=op_meta.op_id,
                    is_eager=op_meta.is_eager,
                    ref_count=op_meta.ref_count,
                    max_iter=op.max_iter,
                    cached_data=op.cached_data,
                )
            )
        return await self._add_task(
            op_group_meta.worker, FnInputBatch(worker_inputs), prereg_task_ids
        )

    async def _dispatch_llm_worker_input_batch(
        self,
        dispatch_map: dict[Worker, list[LLMWorkerInput]],
        prereg_task_ids: dict[str, str],
        schedules: dict[Worker, Iterable[list[str]]],
        dispatch_mode: LLMDispatchMode,
        profiling: bool,
    ) -> None:
        self.logger.debug("Dispatch LLM input batch.")
        for worker, inputs in dispatch_map.items():
            schedule = schedules[worker]
            input_batch = LLMWorkerInputBatch(
                inputs, schedule, dispatch_mode, profiling
            )
            await self._add_task(worker, input_batch, prereg_task_ids)

    async def _start_llm_benchmark(
        self, system_profiling_config: SystemProfilingConfig
    ) -> None:
        self.logger.debug("Start LLM benchmarking.")
        llm_service_info = system_profiling_config.llm_service_info
        for llm_service, workers in self._llm_services.items():
            if llm_service_info is not None and llm_service in llm_service_info:
                for api_key, base_url in set(llm_service_info[llm_service]):
                    for worker in workers:
                        await worker.start_benchmark(api_key, base_url)
            else:
                for worker in workers:
                    await worker.start_benchmark(None, None)

    async def _stop_llm_benchmark(
        self, system_profiling_config: SystemProfilingConfig
    ) -> dict[str, dict[str, dict[str, Any]]]:
        self.logger.debug("Stop LLM benchmarking.")
        llm_service_info = system_profiling_config.llm_service_info
        llm_benchmark_results: dict[str, dict[str, dict[str, Any]]] = {}
        for llm_service, workers in self._llm_services.items():
            for worker in workers:
                llm_worker_results: dict[str, dict[str, Any]] = {}
                if llm_service_info is not None and llm_service in llm_service_info:
                    for api_key, base_url in set(llm_service_info[llm_service]):
                        llm_worker_results[base_url or "none"] = (
                            await worker.stop_benchmark(api_key, base_url)
                        )
                else:
                    llm_worker_results["none"] = await worker.stop_benchmark(None, None)
                llm_benchmark_results[worker.name] = llm_worker_results
        return llm_benchmark_results

    async def _schedule_llm_ops(
        self,
        method: LLMSchedulingMethod,
        dispatch_map: dict[Worker, list[LLMWorkerInput]],
        radix_tree: TemplatedRadixTree | None,
        query_profile: HeliumQueryProfile | None,
    ) -> dict[Worker, Iterable[list[str]]]:
        schedules: dict[Worker, Iterable[list[str]]]
        profiling_range = f"llm_schedule:{self.name}"
        self.start_profiling_range(self.name, profiling_range)
        try:
            match method:
                case LLMSchedulingMethod.CAS:
                    assert radix_tree is not None
                    profiling_info_map: dict[str, LLMProfilingInfo]
                    all_inputs = list(itertools.chain(*dispatch_map.values()))
                    input_slice_map = {inp.op_id: inp.input_slice for inp in all_inputs}
                    if query_profile is None:
                        profiling_info_map = {
                            inp.op_id: LLMProfilingInfo.default() for inp in all_inputs
                        }
                    else:
                        profiling_info_map = {
                            op_id: (
                                LLMProfilingInfo.default()
                                if profiling_info is None
                                else profiling_info
                            )
                            for op_id, profiling_info in query_profile.llm_profiling_info.items()
                        }
                    cas_schedule = cache_aware_schedule(
                        radix_tree,
                        input_slice_map,
                        profiling_info_map,
                        self._llm_worker_info,
                    )
                    schedules = {
                        self._llm_workers[worker_name]: schedule
                        for worker_name, schedule in cas_schedule.items()
                    }
                case LLMSchedulingMethod.BATCH_WISE:
                    schedules = {}
                    for worker, inputs in dispatch_map.items():
                        schedules[worker] = batch_wise_schedule(
                            [inp.op_id for inp in inputs]
                        )
                case LLMSchedulingMethod.OP_WISE:
                    schedules = {}
                    for worker, inputs in dispatch_map.items():
                        schedules[worker] = op_wise_schedule(
                            [inp.op_id for inp in inputs]
                        )
                case LLMSchedulingMethod.OP_WISE_NO_WAIT:
                    schedules = {}
                    for worker, inputs in dispatch_map.items():
                        schedules[worker] = [[inp.op_id for inp in inputs]]
        finally:
            await self.stop_profiling_range(self.name, profiling_range)

        return schedules


def _remove_redundant_slicing(compiled_graph: CompiledGraph) -> CompiledGraph:
    # Implements lightweight graph rewrites to reduce redundant slicing and
    # flatten concat chains. This performs a few local, semantics-preserving
    # transformations and recompiles the graph to drop any now-unreferenced
    # nodes.
    #
    # Rules (informal):
    # - Flatten Concat children that are single-consumer Concats.
    # - Fuse Slice over Slice via absolute-index intersection.
    # - Push Slice through generic ops (row-wise 1:1) by inserting per-input
    #   slices; mutate the parent in place if the slice is its sole consumer,
    #   otherwise clone to keep the rewrite edge-local.
    # - Distribute Slice over Concat similarly; mutate in place when safe.
    #
    # Note: We avoid pushing through FutureOp. Removal of nodes relies on a
    # final recompile, which rebuilds the reachable graph from outputs.
    graph = compiled_graph.graph
    for op in graph.iter_ops(CacheFetchOp):
        op.resolve()

    def _merge_concats() -> bool:
        changed = False
        deps = graph.dependencies()  # op -> dependents set
        for op in graph.iter_ops():
            if not isinstance(op, ConcatOp):
                continue
            # Try to flatten child concats that have only this parent.
            flat_inputs: list[Op] = []
            did_flatten = False
            for child in op.inputs:
                if isinstance(child, ConcatOp) and len(deps.get(child, set())) == 1:
                    flat_inputs.extend(child.inputs)
                    did_flatten = True
                else:
                    flat_inputs.append(child)
            if did_flatten:
                op.inputs = flat_inputs
                changed = True
        return changed

    def _fuse_adjacent_slices() -> bool:
        changed = False
        # We do a single sweep; additional fusions will be caught by the outer loop.
        for s in graph.iter_ops(SliceOp):
            parent = s.inp
            if isinstance(parent, SliceOp):
                # Fuse via absolute intersection
                new_indices = parent.indices.intersect(s.indices)
                if new_indices != s.indices or parent.inp is not s.inp:
                    s.indices = new_indices
                    s.inputs[0] = parent.inp
                    changed = True
        return changed

    def _pushdown_slices() -> bool:
        changed_any = False
        # Work to fixed point within this call by sweeping until no changes.
        while True:
            changed = False
            deps = graph.dependencies()
            for s in graph.iter_ops(SliceOp):
                parent = s.inp
                consumers = deps.get(s, set())
                if not consumers:
                    # Orphan slice (no dependents) — skip; will be pruned on recompile
                    continue

                # Case 1: Slice over Slice -> fuse
                if isinstance(parent, SliceOp):
                    new_idx = parent.indices.intersect(s.indices)
                    if new_idx != s.indices or parent.inp is not s.inp:
                        s.indices = new_idx
                        s.inputs[0] = parent.inp
                        changed = True
                    continue

                # Case 2: Slice over generic op (row-wise 1:1) => push to all inputs
                if not isinstance(
                    parent, (CacheFetchOp, ConcatOp, DataOp, InputOp, FutureOp)
                ):
                    parent_consumers = deps.get(parent, set())
                    if len(parent_consumers) == 1 and next(iter(parent_consumers)) is s:
                        # Mutate in place (safe)
                        parent.inputs = [
                            (
                                pin
                                if (
                                    isinstance(pin, SliceOp)
                                    and pin.indices == s.indices
                                )
                                else SliceOp(pin, indices=s.indices)
                            )
                            for pin in parent.inputs
                        ]
                        for consumer in consumers:
                            consumer.replace_input(s, parent)
                        changed = True
                    continue

                # Case 3: Slice over Concat => distribute to each child then bypass
                if isinstance(parent, ConcatOp):
                    parent_consumers = deps.get(parent, set())
                    if len(parent_consumers) == 1 and next(iter(parent_consumers)) is s:
                        # Skip mutation if concat has no children to avoid zero-input concat
                        if len(parent.inputs) == 0:
                            continue
                        parent.inputs = [
                            (
                                child
                                if (
                                    isinstance(child, SliceOp)
                                    and child.indices == s.indices
                                )
                                else SliceOp(child, indices=s.indices)
                            )
                            for child in parent.inputs
                        ]
                        for consumer in consumers:
                            consumer.replace_input(s, parent)
                        changed = True
                    continue

                # Otherwise, cannot push further (sources / FutureOp / CacheFetchOp)
            changed_any = changed_any or changed
            if not changed:
                break
        return changed_any

    # Run optimizer to a small fixed-point.
    # A couple of iterations are typically enough.
    for _ in range(3):  # safeguard to avoid pathological loops
        changed = False
        if _merge_concats():
            changed = True
        if _pushdown_slices():
            changed = True
        if _merge_concats():  # flatten newly exposed concats
            changed = True
        if _fuse_adjacent_slices():
            changed = True
        if not changed:
            break

    # Recompile to prune unreachable nodes and recompute input slices.
    for op in compiled_graph.iter_ops(CacheFetchOp):
        op.unresolve()
    return compiled_graph.recompile(check_inputs=False)
