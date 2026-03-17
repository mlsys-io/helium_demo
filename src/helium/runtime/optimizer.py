from collections import defaultdict
from dataclasses import dataclass
from typing import Hashable, TypeVar

from helium.common import Slice
from helium.graphs import CompiledGraph, Graph
from helium.ops import (
    AppendMessageOp,
    CacheFetchOp,
    ConcatOp,
    DataOp,
    FormatOp,
    FunctionalOp,
    InputOp,
    LambdaOp,
    LastMessageOp,
    LLMChatOp,
    LLMCompletionOp,
    LLMOp,
    MessageOp,
    Op,
    OutputOp,
    SliceOp,
)
from helium.runtime.data import Data
from helium.runtime.functional import (
    AppendMessageFnInput,
    CacheResolveFnInput,
    ConcatFnInput,
    DataFnInput,
    FnInput,
    FormatFnInput,
    InputFnInput,
    LambdaFnInput,
    LastMessageFnInput,
    MessageFnInput,
    OutputFnInput,
    SliceFnInput,
)
from helium.runtime.llm import LLMProfilingInfo
from helium.runtime.profiler import RequestProfiler
from helium.runtime.protocol import HeliumQueryProfile
from helium.runtime.request import RequestInfo
from helium.runtime.utils.logger import Logger, LogLevel, init_child_logger
from helium.runtime.worker.llm_worker import (
    LLMChatWorkerInput,
    LLMCompletionWorkerInput,
)
from helium.runtime.worker.worker_input import ResultPuller, WorkerArg
from helium.utils import unique_id

T = TypeVar("T")


@dataclass
class OptimizerInfo:
    original_graph_name: str | None
    """Name of the original graph, if the query contains a single graph"""
    merged_query_profile: HeliumQueryProfile | None
    """Merged query profile, if available"""
    output_mapping: dict[str, tuple[str, str]] | None
    """Mapping from output op ID to (graph name, output name)"""
    op_mapping: dict[str, list[str]] | None
    """Mapping from graph name to set of op IDs in the corresponding graph"""


class CacheResolver:
    def __init__(self, puller: ResultPuller) -> None:
        self._puller = puller

    async def resolve(
        self, request_id: str, compiled_graph: CompiledGraph
    ) -> dict[str, Data]:
        """Resolves the cache keys for the cacheable ops (e.g., LLM ops) in the compute
        graph"""
        puller = self._puller
        op_outputs: dict[str, WorkerArg] = {
            op_id: WorkerArg(op_id=op_id, data=Data.text(inp, None))
            for op_id, inp in compiled_graph.inputs.items()
        }
        input_slices = compiled_graph.input_slices
        # op ID -> cached data
        cached_outputs: dict[str, Data] = {}
        for op in compiled_graph.iter_ops():
            # Create WorkerInput
            worker_input = self._create_worker_input(
                request_id, op, op_outputs, input_slices[op.id]
            )
            counter = 0
            # Execute the WorkerInput
            async for data in worker_input.run(puller):
                if counter > 0:
                    raise ValueError(f"Op {op.id} returned multiple outputs.")
                if data is None:
                    raise ValueError(f"Op {op.id} returned no output.")
                if isinstance(worker_input, CacheResolveFnInput):
                    cached_data = worker_input.resolved_data
                    if cached_data is not None:
                        cached_outputs[op.id] = cached_data
                op_outputs[op.id] = WorkerArg(op_id=op.id, data=data)
                counter += 1
        return cached_outputs

    def _create_worker_input(
        self,
        request_id: str,
        op: Op,
        op_outputs: dict[str, WorkerArg],
        input_slice: Slice,
    ) -> FnInput:
        match op:
            case DataOp():
                return DataFnInput(
                    request_id=request_id,
                    op_id=op.id,
                    is_eager=True,
                    ref_count=1,
                    data=op.data,
                )
            case InputOp():
                return InputFnInput(
                    request_id=request_id,
                    op_id=op.id,
                    is_eager=True,
                    ref_count=1,
                    inputs=op_outputs[op.name].data.as_text(),
                )
            case OutputOp():
                return OutputFnInput(
                    request_id=request_id,
                    op_id=op.id,
                    is_eager=True,
                    ref_count=1,
                    output=op_outputs[op.inputs[0].id],
                )
            case MessageOp():
                return MessageFnInput(
                    request_id=request_id,
                    op_id=op.id,
                    is_eager=True,
                    ref_count=1,
                    max_iter=op.max_iter,
                    roles=op.roles,
                    message_refs=op.message_refs,
                    inputs=[op_outputs[input_op.id] for input_op in op.inputs],
                )
            case AppendMessageOp():
                op_content = op.content
                content = (
                    op_content
                    if isinstance(op_content, str)
                    else op_outputs[op_content.id]
                )
                return AppendMessageFnInput(
                    request_id=request_id,
                    op_id=op.id,
                    is_eager=True,
                    ref_count=1,
                    max_iter=op.max_iter,
                    messages=op_outputs[op.messages.id],
                    content=content,
                    role=op.role,
                )
            case LastMessageOp():
                return LastMessageFnInput(
                    request_id=request_id,
                    op_id=op.id,
                    is_eager=True,
                    ref_count=1,
                    max_iter=op.max_iter,
                    messages=op_outputs[op.messages.id],
                )
            case FormatOp():
                return FormatFnInput(
                    request_id=request_id,
                    op_id=op.id,
                    is_eager=True,
                    ref_count=1,
                    max_iter=op.max_iter,
                    template=op.template,
                    format_args=[op_outputs[arg.id] for arg in op.format_args],
                    format_kwargs={
                        k: op_outputs[v.id] for k, v in op.format_kwargs.items()
                    },
                )
            case LambdaOp():
                return LambdaFnInput(
                    request_id=request_id,
                    op_id=op.id,
                    is_eager=True,
                    ref_count=1,
                    max_iter=op.max_iter,
                    inputs=[op_outputs[input_op.id] for input_op in op.inputs],
                    fn=op.fn,
                )
            case SliceOp():
                return SliceFnInput(
                    request_id=request_id,
                    op_id=op.id,
                    is_eager=True,
                    ref_count=1,
                    max_iter=op.max_iter,
                    inp=op_outputs[op.inp.id],
                    indices=op.indices,
                )
            case ConcatOp():
                return ConcatFnInput(
                    request_id=request_id,
                    op_id=op.id,
                    is_eager=True,
                    ref_count=1,
                    max_iter=op.max_iter,
                    inputs=[op_outputs[input_op.id] for input_op in op.inputs],
                )
            case LLMCompletionOp():
                return CacheResolveFnInput.wrap(
                    LLMCompletionWorkerInput(
                        request_id=request_id,
                        op_id=op.id,
                        is_eager=True,
                        ref_count=1,
                        max_iter=op.max_iter,
                        input_slice=input_slice,
                        generation_config=op.config,
                        prompts=op_outputs[op.prompt.id],
                        echo=op.echo,
                        cacheable=False,
                    )
                )
            case LLMChatOp():
                return CacheResolveFnInput.wrap(
                    LLMChatWorkerInput(
                        request_id=request_id,
                        op_id=op.id,
                        is_eager=True,
                        ref_count=1,
                        max_iter=op.max_iter,
                        input_slice=input_slice,
                        generation_config=op.config,
                        messages=op_outputs[op.messages.id],
                        return_history=op.return_history,
                        cacheable=False,
                    )
                )
            case _:
                raise ValueError(f"Cache resolution does not support {type(op)}.")


class HeliumOptimizer:
    def __init__(
        self,
        puller: ResultPuller,
        request_profiler: RequestProfiler,
        logger: Logger | None = None,
        log_level: LogLevel | None = None,
    ) -> None:
        self.logger: Logger = init_child_logger("Optimizer", logger, log_level)
        self._cache_resolver = CacheResolver(puller)
        self._request_profiler = request_profiler

    def initial_rewrite(self, request_info: RequestInfo) -> OptimizerInfo:
        self.logger.info("Performing initial query plan rewrite.")

        # 1. Merge graphs
        merged_graph, info = self._merge_graphs(request_info)
        # 2. Merge nodes
        merged_graph = self._merge_nodes(merged_graph)
        # 3. Partition disjoint graphs
        disjoint_graphs = merged_graph.copy(new_ids=False).disjoint_subgraphs()

        request_info.compiled_graph = merged_graph
        request_info.disjoint_graphs = disjoint_graphs
        request_info.query_profile = info.merged_query_profile

        return info

    async def cache_aware_optimize(self, request_info: RequestInfo) -> None:
        request_id = request_info.request_id
        cache_resolver = self._cache_resolver

        async def optimize_graph(compiled_graph: CompiledGraph) -> CompiledGraph:
            self.logger.info("Performing cache resolution.")
            cached_outputs = await cache_resolver.resolve(request_id, compiled_graph)

            if not cached_outputs:
                return compiled_graph

            # Rebuild the compute graph with cached results resolved.
            input_slices = compiled_graph.input_slices
            dependent_ops = compiled_graph.dependencies()
            for op in compiled_graph.iter_ops(LLMOp):
                if op.id not in cached_outputs:
                    continue
                cached_data = cached_outputs[op.id]

                remaining_slice = input_slices[op.id].difference(
                    Slice(cached_data.indices)
                )
                if remaining_slice.length == 0:
                    # Replace the current op by the CacheFetch op.
                    # Use ConcatOp as the dummy op.
                    cache_fetch_op = CacheFetchOp(ConcatOp(), cached_data)
                    cache_fetch_op.resolve()
                    for dependent_op in dependent_ops[op]:
                        dependent_op.replace_input(op, cache_fetch_op)
                else:
                    # Insert a slice op to extract the remaining inputs.
                    slice_op = SliceOp(op.inputs[0], indices=remaining_slice)
                    op.inputs[0] = slice_op
                    # Insert a CacheFetchOp to fetch the cached output.
                    cache_fetch_op = CacheFetchOp(op, cached_data)
                    # Insert a ConcatOp to combine the cached output and the remaining outputs.
                    concat_op = ConcatOp([op, cache_fetch_op])
                    for dependent_op in dependent_ops[op]:
                        dependent_op.replace_input(op, concat_op)

            return compiled_graph.recompile(check_inputs=False)

        self.logger.info("Performing cache-aware logical plan optimization.")
        request_info.disjoint_graphs = [
            await optimize_graph(compiled_graph)
            for compiled_graph in request_info.disjoint_graphs
        ]

    def remap_output(
        self, output: dict[str, T], info: OptimizerInfo
    ) -> dict[str, dict[str, T]]:
        new_output: dict[str, dict[str, T]] = {}

        # 1. Remap for merged graph
        if info.output_mapping is not None:
            for output_id, value in output.items():
                graph_name, output_name = info.output_mapping[output_id]
                if graph_name not in new_output:
                    new_output[graph_name] = {}
                new_output[graph_name][output_name] = value
        else:
            assert info.original_graph_name is not None
            new_output[info.original_graph_name] = output

        return new_output

    def remap_ops(
        self, output: dict[str, T], info: OptimizerInfo
    ) -> dict[str, dict[str, T]]:
        if info.op_mapping is None:
            assert info.original_graph_name is not None
            return {info.original_graph_name: output}

        op_mapping = info.op_mapping
        new_output: dict[str, dict[str, T]] = {}
        for graph_name, op_ids in op_mapping.items():
            if graph_name not in new_output:
                new_output[graph_name] = {}
            cur_output = new_output[graph_name]
            for op_id in op_ids:
                if op_id in output:
                    cur_output[op_id] = output[op_id]

        return new_output

    def _merge_graphs(
        self, request_info: RequestInfo
    ) -> tuple[CompiledGraph, OptimizerInfo]:
        # Get query profile if available
        if (
            request_info.query_profiling_config is None
            or request_info.query_profiling_config.query_profile_map is None
        ):
            query_profile_map = {}
            has_query_profile = False
        else:
            query_profile_map = request_info.query_profiling_config.query_profile_map
            has_query_profile = True

        compiled_graphs = request_info.query_graphs
        if len(compiled_graphs) == 1:
            graph_name, compiled_graph = next(iter(compiled_graphs.items()))
            info = OptimizerInfo(
                original_graph_name=graph_name,
                merged_query_profile=query_profile_map.get(graph_name),
                output_mapping=None,
                op_mapping=None,
            )
            return compiled_graph, info

        output_ops: list[OutputOp] = []
        # (llm op id) -> LLMProfilingInfo
        merged_llm_profiling_info: dict[str, LLMProfilingInfo | None] = {}
        # (new input name) -> input values
        merged_inputs: dict[str, list[str]] = {}
        # (new output name) -> (graph name, output name)
        output_mapping: dict[str, tuple[str, str]] = {}
        # (graph name) -> set of op IDs in the graph
        op_mapping: dict[str, list[str]] = {}

        for name, compiled_graph in compiled_graphs.items():
            graph = compiled_graph.graph
            inputs = compiled_graph.inputs
            query_profile = query_profile_map.get(name)

            # Must perform in place as other nodes are referring to them.
            for input_name, input_op in graph.input_ops.items():
                new_input_name = unique_id()
                input_op.name = new_input_name
                merged_inputs[new_input_name] = inputs[input_name]
            for output_name, output_op in graph.output_ops.items():
                new_output_name = unique_id()
                output_op.name = new_output_name
                output_mapping[new_output_name] = (name, output_name)
                output_ops.append(output_op)

            # Merge LLM profiling info
            if query_profile is not None:
                merged_llm_profiling_info.update(query_profile.llm_profiling_info)

            # Collect op IDs for the graph
            op_mapping[name] = [op.id for op in graph.iter_ops()]

        info = OptimizerInfo(
            original_graph_name=None,
            merged_query_profile=(
                HeliumQueryProfile(llm_profiling_info=merged_llm_profiling_info)
                if has_query_profile
                else None
            ),
            output_mapping=output_mapping,
            op_mapping=op_mapping,
        )

        # Single compute graph
        compiled_graph = Graph.from_ops(output_ops).compile(**merged_inputs)

        return compiled_graph, info

    def _merge_nodes(self, compiled_graph: CompiledGraph) -> CompiledGraph:
        graph = compiled_graph.graph
        dependencies = graph.dependencies()

        # Group ops with the same signature
        op_groups: dict[Hashable, set[FunctionalOp]] = defaultdict(set)
        for op in graph.as_dict().values():
            # Merge only functional ops
            if isinstance(op, FunctionalOp):
                op_groups[(op.__class__, op.signature())].add(op)

        for ops in op_groups.values():
            if len(ops) == 1:
                continue

            merged_op = next(iter(ops))
            # Remove redundant ops
            for op in ops:
                if op == merged_op:
                    continue
                for dep in dependencies[op]:
                    dep.inputs = [merged_op if inp == op else inp for inp in dep.inputs]
                    dependencies[merged_op].add(dep)
                del dependencies[op]
        return compiled_graph.recompile()
