from helium.common import Slice
from helium.graphs.graph import CompiledGraph
from helium.ops import ConcatOp, Op, SliceOp
from helium.runtime.protocol import QueryProfilingConfig


def sample_disjoint_graph(
    disjoint_graph: CompiledGraph, query_profiling_config: QueryProfilingConfig
) -> tuple[CompiledGraph | None, CompiledGraph]:
    """Samples a disjoint graph based on the given sampling ratio and size constraints

    Returns
    -------
    CompiledGraph
        The sampled disjoint graph.
    CompiledGraph
        The remaining disjoint graph after sampling.
    """
    data_size = disjoint_graph.data_size
    sampling_size = min(
        int(data_size * query_profiling_config.sampling_ratio),
        query_profiling_config.max_sampling_size,
    )
    if (
        sampling_size < query_profiling_config.min_sampling_size
        or sampling_size > data_size
    ):
        # Don't sample if the sampling size is too small or too large
        return None, disjoint_graph
    sampled_graph = disjoint_graph.copy(new_ids=False).apply_to_data(
        lambda d: d[:sampling_size]
    )
    remaining_graph = disjoint_graph.copy(new_ids=False).apply_to_data(
        lambda d: d[sampling_size:]
    )
    return sampled_graph, remaining_graph


def partition_op(
    op: Op, input_slice: Slice, partition_count: int, dependencies: dict[Op, set[Op]]
) -> list[Op]:
    """Partitions an Op into multiple smaller Ops

    Returns
    -------
    list[Op]
        A list of partitioned Ops.
    """
    if partition_count <= 1:
        return [op]

    # Replace the LLMOp with a ConcatOp in the graph
    concat_op = ConcatOp()
    deps = dependencies.pop(op)
    for dep in deps:
        dep.replace_input(op, concat_op)
    dependencies[concat_op] = deps

    # Create partitioned Ops
    partitioned_ops: list[Op] = []
    for part_slice in input_slice.partition(partition_count):
        # Create a new Op and slice the inputs
        new_op = op.copy()
        concat_op.add_op(new_op)
        dependencies[new_op] = {concat_op}
        for i in range(len(new_op.inputs)):
            # Insert a SliceOp before the new Op
            inp_op = new_op.inputs[i]
            slice_op = SliceOp(inp_op, part_slice)
            new_op.inputs[i] = slice_op
            # Add the SliceOp to dependencies
            dependencies[slice_op] = {new_op}
            # Modify the dependencies to point to the SliceOp
            deps = dependencies[inp_op]
            deps.discard(op)
            deps.add(slice_op)
        partitioned_ops.append(new_op)

    return partitioned_ops
