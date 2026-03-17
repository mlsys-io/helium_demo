from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal, TypeVar, overload

import dill
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.axes import Axes

from helium.common import Slice
from helium.ops import CacheFetchOp, DataOp, FutureOp, InputOp, LLMOp, Op, OutputOp
from helium.utils.graph import detect_wccs, topological_sort
from helium.utils.prefix.radix_tree import (
    PrefixType,
    TemplatedNodeDependency,
    TemplatedRadixTree,
)

OpType = TypeVar("OpType", bound=Op)


class CompiledGraph:
    def __init__(self, graph: "Graph", inputs: dict[str, list[str]]) -> None:
        self.graph = graph
        self.inputs = inputs

        self._data_size: int | None = None
        self._input_slices: dict[str, Slice] | None = None

    def copy(self, new_ids: bool = True) -> "CompiledGraph":
        """Creates a deep copy of the compiled graph"""
        return CompiledGraph(self.graph.copy(new_ids), self.inputs.copy())

    @classmethod
    def merge(cls, disjoint_graphs: list["CompiledGraph"]) -> "CompiledGraph":
        output_ops: list[OutputOp] = []
        inputs: dict[str, list[str]] = {}
        input_slices: dict[str, Slice] = {}
        for disjoint_graph in disjoint_graphs:
            output_ops.extend(disjoint_graph.graph.output_ops.values())
            inputs.update(disjoint_graph.inputs)
            input_slices.update(disjoint_graph.input_slices)
        merged_graph = Graph.from_ops(output_ops).compile(**inputs)
        # Update input slices
        merged_graph._input_slices = input_slices
        return merged_graph

    @property
    def data_size(self) -> int:
        if self._data_size is not None:
            return self._data_size

        data_size = None
        for op in self.iter_ops(DataOp):
            op_data_size = len(op.data)
            if data_size is None:
                data_size = op_data_size
            elif op_data_size != data_size:
                raise ValueError(
                    f"Data size mismatch: {data_size} != {op_data_size} for op {op.id}"
                )
        for input_name, input_data in self.inputs.items():
            input_data_size = len(input_data)
            if data_size is None:
                data_size = input_data_size
            elif input_data_size != data_size:
                raise ValueError(
                    f"Input data size mismatch: {data_size} != {input_data_size} "
                    f"for input {input_name}"
                )
        if data_size is None:
            # Check CacheFetchOp
            for cache_op in self.iter_ops(CacheFetchOp):
                op_data_size = len(cache_op.cached_data)
                if data_size is None:
                    data_size = op_data_size
                elif op_data_size != data_size:
                    raise ValueError(
                        f"Data size mismatch: {data_size} != {op_data_size} for op {cache_op.id}"
                    )
            if data_size is None:
                raise ValueError("No data found in the graph or inputs.")
        return data_size

    @property
    def input_slices(self) -> dict[str, Slice]:
        if self._input_slices is not None:
            return self._input_slices

        data_size = self.data_size
        input_slices: dict[str, Slice] = self.graph.build_input_slices(data_size)
        self._input_slices = input_slices
        return input_slices

    def serialize(self) -> dict[str, Any]:
        return {
            "graph": self.graph.serialize(),
            "inputs": self.inputs,
        }

    def recompile(self, check_inputs: bool = True) -> "CompiledGraph":
        graph = Graph.from_ops(list(self.graph.output_ops.values()))
        if check_inputs:
            return graph.compile(**self.inputs)
        input_ops = graph.input_ops
        inputs = {name: data for name, data in self.inputs.items() if name in input_ops}
        return graph.compile(**inputs)

    @overload
    def iter_ops(self, op_type: None = None) -> Iterable[Op]: ...

    @overload
    def iter_ops(self, op_type: type[OpType]) -> Iterable[OpType]: ...

    def iter_ops(self, op_type: type[Op] | None = None) -> Iterable[Op]:
        return self.graph.iter_ops(op_type)

    def apply_to_data(self, func: Callable[[list[str]], list[str]]) -> "CompiledGraph":
        self.graph.apply_to_data(func)
        self.inputs = {name: func(data) for name, data in self.inputs.items()}
        return self

    def dependencies(self, include_loop: bool = True) -> dict[Op, set[Op]]:
        return self.graph.dependencies(include_loop)

    def build_llm_dependencies(self) -> dict[str, set[LLMOp]]:
        return self.graph.build_llm_dependencies()

    def build_radix_tree(
        self,
        worker_assignment: dict[str, str | None],
        sliced_op_map: dict[str, str] | None = None,
    ) -> TemplatedRadixTree:
        return self.graph.build_radix_tree(worker_assignment, sliced_op_map)

    def disjoint_subgraphs(self) -> list["CompiledGraph"]:
        """Partitions the graph into disjoint subgraphs based on weakly connected
        components (WCCs)"""
        dependencies = {op: deps for op, deps in self.dependencies().items()}
        wccs = detect_wccs(dependencies)

        graphs: list[CompiledGraph] = []
        for wcc in wccs:
            inputs: dict[str, list[str]] = {}
            output_ops: list[OutputOp] = []
            for op in wcc:
                if isinstance(op, InputOp):
                    inputs[op.name] = self.inputs[op.name]
                elif isinstance(op, OutputOp):
                    output_ops.append(op)
            graphs.append(Graph.from_ops(output_ops).compile(**inputs))
        return graphs


class Graph:
    def __init__(self, graph: dict[str, Op]) -> None:
        self._graph = graph

        # Build input and output op mappings.
        self._input_ops: dict[str, InputOp] = {}
        self._output_ops: dict[str, OutputOp] = {}
        for op in self._graph.values():
            if isinstance(op, InputOp):
                if op.name in self._input_ops:
                    raise ValueError(f"Duplicate input name: {op.name}")
                self._input_ops[op.name] = op
            elif isinstance(op, OutputOp):
                if op.name in self._output_ops:
                    raise ValueError(f"Duplicate output name: {op.name}")
                self._output_ops[op.name] = op

        # Resolve FutureOps. This is needed to
        # 1. Remove the wrapped op from its dependencies after graph traversal to
        # allow scheduling.
        # 2. set the op attribute to point to the wrapped op.
        for op in graph.values():
            if isinstance(op, FutureOp):
                op.resolve(graph)

        # Sort the graph in topological order.
        self.topological_sort()

        # Validate the correctness of the graph.
        self.validate()

    def as_dict(self) -> dict[str, Op]:
        """Mapping from op ID to op"""
        return self._graph

    @property
    def input_ops(self) -> dict[str, InputOp]:
        """Mapping from input name to input op"""
        return self._input_ops

    @property
    def output_ops(self) -> dict[str, OutputOp]:
        """Mapping from output name to output op"""
        return self._output_ops

    @overload
    def iter_ops(self, op_type: None = None) -> Iterable[Op]: ...

    @overload
    def iter_ops(self, op_type: type[OpType]) -> Iterable[OpType]: ...

    def iter_ops(self, op_type: type[Op] | None = None) -> Iterable[Op]:
        """Iterates over all ops in the graph in topological order"""
        if op_type is None:
            return self._graph.values()
        return (op for op in self._graph.values() if isinstance(op, op_type))

    @property
    def node_count(self) -> int:
        return len(self._graph)

    def compile(self, *_, **inputs: list[str]) -> CompiledGraph:
        # Check input length
        if len(inputs) != len(self._input_ops):
            raise ValueError(
                f"Expected {len(self._input_ops)} inputs but got {len(inputs)}"
            )

        # Check all inputs are present
        for name in self._input_ops:
            if name not in inputs:
                raise ValueError(f"Missing input: {name}")

        # Check all inputs are lists of strings
        for name, value in inputs.items():
            if not isinstance(value, list):
                raise ValueError(f"Input {name} is not a list")
            for v in value:
                if not isinstance(v, str):
                    raise ValueError(f"Input {name} contains a non-string value")

        return CompiledGraph(self, inputs)

    def serialize(self) -> dict[str, Any]:
        serialized_graph = {op_id: op.serialize() for op_id, op in self._graph.items()}
        return serialized_graph

    @classmethod
    def from_ops(cls, ops: list[OutputOp]) -> "Graph":
        graph: dict[str, Op] = {}
        for op in ops:
            op.traverse_graph(graph)
        return cls(graph)

    @classmethod
    def from_json(cls, json_graph: dict[str, Any]) -> "Graph":
        graph: dict[str, Op] = {}
        for op_id, op_data in json_graph.items():
            op_cls = Op.registry.get_class(op_data["_op"])
            op = op_cls.from_json(op_data, graph)
            graph[op_id] = op
        return cls(graph)

    def copy(self, new_ids: bool = True) -> "Graph":
        node_map = {op: op.copy(new_ids) for op in self.iter_ops()}

        # Update op dependencies
        output_ops: list[OutputOp] = []
        for new_op in node_map.values():
            new_op.inputs = [node_map[inp] for inp in new_op.inputs]
            if isinstance(new_op, FutureOp):
                new_op.update(node_map[new_op.op])
            elif isinstance(new_op, OutputOp):
                output_ops.append(new_op)

        return Graph.from_ops(output_ops)

    def apply_to_data(self, func: Callable[[list[str]], list[str]]) -> "Graph":
        for op in self.iter_ops(DataOp):
            op.data = func(op.data)
        return self

    def validate(self) -> None:
        # TODO: Implement this
        pass

    def dependencies(self, include_loop: bool = True) -> dict[Op, set[Op]]:
        """Returns a dependency mapping from op to a set of dependent ops"""
        dependencies: defaultdict[Op, set[Op]] = defaultdict(set)
        for op in self._graph.values():
            for dep in op.inputs:
                dependencies[dep].add(op)
                if isinstance(dep, FutureOp) and include_loop:
                    # Also add dependent ops to the wrapped ops.
                    dep_op = dep.op
                    assert dep_op is not None
                    dependencies[dep_op].add(op)
        return dict(dependencies)

    def visualize(
        self,
        save_to: Path | None = None,
        ax: Axes | None = None,
        layout: Literal["spring", "planar", "circular", "kamada_kawai"] | None = None,
    ) -> nx.Graph:
        # op_id -> op name
        op_name_map = {
            op: f"{op.__class__.__name__}\n({op.id})" for op in self.iter_ops()
        }

        graph_dict: dict[str, list[str]] = {}
        node_color: list[str] = []
        dependencies = self.dependencies()
        for op in self.iter_ops():
            op_name = op_name_map[op]
            if op not in dependencies:
                graph_dict[op_name] = []
            else:
                graph_dict[op_name] = [op_name_map[dep] for dep in dependencies[op]]

            match op:
                case InputOp():
                    node_color.append("#ff7f0e")
                case DataOp():
                    node_color.append("#2ca02c")
                case OutputOp():
                    node_color.append("#d62728")
                case LLMOp():
                    node_color.append("#9467bd")
                case CacheFetchOp():
                    node_color.append("#8c564b")
                case _:
                    node_color.append("#1f77b4")

        nxgraph = nx.DiGraph(graph_dict)  # type: ignore[arg-type]
        nx.set_node_attributes(
            nxgraph, {u: color for u, color in zip(nxgraph.nodes, node_color)}, "color"
        )
        if layout is None:
            pos = None
        elif layout == "spring":
            pos = nx.spring_layout(nxgraph)
        elif layout == "planar":
            pos = nx.planar_layout(nxgraph)
        elif layout == "circular":
            pos = nx.circular_layout(nxgraph)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(nxgraph)
        else:
            raise ValueError("Invalid layout type.")
        nx.draw_networkx(nxgraph, pos=pos, node_color=node_color, ax=ax)
        if save_to is not None:
            plt.savefig(save_to)
        return nxgraph

    def _get_llm_dependencies(
        self, op: Op, llm_dependencies: dict[Op, set[LLMOp]]
    ) -> set[LLMOp]:
        """Identifies LLM dependencies for the given op

        Returns
        -------
        set[LLMOp]
            A set of LLM ops depended on.
        """
        llm_deps: set[LLMOp] = set()
        for op_inp in op.inputs:
            if isinstance(op_inp, LLMOp):
                # Add the current LLM op as a dependency
                llm_deps.add(op_inp)
            else:
                # Inherit LLM dependencies from the input op
                llm_deps.update(llm_dependencies[op_inp])
        return llm_deps

    def build_input_slices(self, data_size: int) -> dict[str, Slice]:
        input_slices: dict[str, Slice] = {}
        future_ops: list[FutureOp] = []
        for op_id, op in self._graph.items():
            if isinstance(op, FutureOp):
                future_ops.append(op)
            else:
                input_slices[op_id] = op.get_input_slice(data_size, input_slices)
        # Resolve FutureOps
        for op in future_ops:
            input_slices[op.id] = op.get_input_slice(data_size, input_slices)
        return input_slices

    def build_llm_dependencies(self) -> dict[str, set[LLMOp]]:
        """Builds the LLM dependencies for LLM ops in the graph

        Returns
        -------
        dict[str, set[LLMOp]]
            A mapping from op ID to a set of LLM ops depended on.
        """
        # Get the LLM ops in topological order (input -> output)
        llm_dependencies: dict[Op, set[LLMOp]] = {}
        for op in self.iter_ops():
            llm_dependencies[op] = self._get_llm_dependencies(op, llm_dependencies)
        return {
            op.id: deps
            for op, deps in llm_dependencies.items()
            if isinstance(op, LLMOp)
        }

    def build_prefix_templates(
        self, sliced_op_map: dict[str, str] | None = None
    ) -> dict[str, tuple[PrefixType, set[LLMOp]]]:
        """Builds the prefix templates and LLM dependencies for LLM ops in the graph

        Returns
        -------
        dict[str, tuple[PrefixType, set[LLMOp]]]
            A mapping from op ID to a tuple of prefix template and LLM dependencies
            (LLM ops depended on).
        """
        if sliced_op_map is None:
            sliced_op_map = {}
        prefix_templates: dict[str, tuple[PrefixType, set[LLMOp]]] = {}

        input_templates: dict[str, PrefixType] = {}
        llm_dependencies: dict[Op, set[LLMOp]] = {}
        for op in self.iter_ops():
            prefix_template = op.get_prefix_template(input_templates, sliced_op_map)
            llm_deps = self._get_llm_dependencies(op, llm_dependencies)
            input_templates[op.id] = prefix_template
            llm_dependencies[op] = llm_deps
            if isinstance(op, LLMOp):
                prefix_templates[op.id] = (
                    op.get_input_prefix_template(input_templates),
                    llm_deps,
                )
        return prefix_templates

    def build_radix_tree(
        self,
        worker_assignment: dict[str, str | None],
        sliced_op_map: dict[str, str] | None = None,
    ) -> TemplatedRadixTree:
        """Builds a templated radix tree from the graph"""
        prefix_templates = self.build_prefix_templates(sliced_op_map)
        tree = TemplatedRadixTree()
        node_dict: dict[str, TemplatedNodeDependency] = {}
        for op_id, (template, dependencies) in prefix_templates.items():
            worker = worker_assignment[op_id]
            assert worker is not None
            node_deps = {node_dict[dep.id] for dep in dependencies}
            new_node = tree.add(template, worker, op_id, node_deps)
            node_dict[op_id] = new_node
        return tree

    def topological_sort(self) -> None:
        dependencies: defaultdict[Op, set[Op]] = defaultdict(set)
        for op in self._graph.values():
            for dep in op.inputs:
                dependencies[dep].add(op)
        sorted_ops = topological_sort(dict(dependencies))
        self._graph = {op.id: op for op in sorted_ops}

    def save(
        self,
        path: str | Path,
        query_profiling_info: dict[str, dict[str, Any]] | None,
    ) -> None:
        obj = {"graph": self, "query_profiling_info": query_profiling_info}
        with open(path, "wb") as f:
            dill.dump(obj, f)

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> tuple["Graph", dict[str, dict[str, Any]] | None]:
        with open(path, "rb") as f:
            obj = dill.load(f)
        if not isinstance(obj, dict) or "graph" not in obj:
            raise ValueError("Invalid file format")
        return obj["graph"], obj.get("query_profiling_info", None)


def from_ops(ops: list[OutputOp]) -> Graph:
    return Graph.from_ops(ops)


def from_json(json_graph: dict[str, Any]) -> Graph:
    return Graph.from_json(json_graph)


def save(
    graph: Graph,
    path: str | Path,
    query_profiling_info: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Saves the graph to a file"""
    graph.save(path, query_profiling_info)


def load(
    graph: Graph, path: str | Path
) -> tuple[Graph, dict[str, dict[str, Any]] | None]:
    """Loads the graph from a file"""
    return graph.load(path)
