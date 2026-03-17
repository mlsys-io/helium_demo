from collections.abc import Callable, Sequence
from typing import Any, Hashable

from helium.common import Slice
from helium.ops.data_ops import DataOp
from helium.ops.ops import FunctionalOp, Op
from helium.ops.util_ops import FutureOp
from helium.utils.prefix.radix_tree import Placeholder, PrefixType

Predicate = Callable[..., bool] | str | list[str]
"""
A type alias for predicates used with SwitchOp. It can be:
    - A callable that takes multiple string arguments when not looping, and an integer 
    followed by multiple string arguments when looping.
    - A string, interpreted as a regex pattern for the stop condition. All conditional 
    inputs must match the pattern for the loop to stop.
    - A list of strings, where each element is matched with the corresponding 
    conditional input.
"""


@Op.registry.register("SwitchOp")
class SwitchOp(FunctionalOp):
    pred: Predicate | None
    branch: bool
    dead_on_empty: bool

    def __init__(
        self,
        input_op: list[str] | Op,
        cond_ops: Sequence[list[str] | Op],
        pred: Predicate | None,
        branch: bool,
        dead_on_empty: bool,
    ) -> None:
        input_op = input_op if isinstance(input_op, Op) else DataOp(input_op)
        inputs: list[Op] = [input_op]
        cond_op_ids: list[int] = []
        for op in cond_ops:
            op = op if isinstance(op, Op) else DataOp(op)
            try:
                i = inputs.index(op)
                cond_op_ids.append(i)
            except ValueError:
                cond_op_ids.append(len(inputs))
                inputs.append(op)
        super().__init__(inputs)
        self.pred = pred
        self.branch = branch
        self.dead_on_empty = dead_on_empty
        self._input_op = 0
        self._cond_ops = cond_op_ids

    @property
    def input_op(self) -> Op:
        return self.inputs[self._input_op]

    @property
    def cond_ops(self) -> list[Op]:
        return [self.inputs[i] for i in self._cond_ops]

    def _serialize(self) -> dict[str, Any]:
        if not (self.pred is None or isinstance(self.pred, str)):
            raise ValueError("Callable predicates are not serializable")
        return dict(
            input_op=self.input_op.id,
            cond_ops=[op.id for op in self.cond_ops],
            pred=self.pred,
            branch=self.branch,
            dead_on_empty=self.dead_on_empty,
        )

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "SwitchOp":
        input_op_id = data["input_op"]
        cond_op_ids = data["cond_ops"]
        if len(data["_inputs"]) != len(set([input_op_id, *cond_op_ids])):
            raise ValueError(
                "SwitchOp must have an equal number of inputs and cond_ops"
            )
        return cls(
            input_op=other_ops[input_op_id],
            cond_ops=[other_ops[op_id] for op_id in cond_op_ids],
            pred=data["pred"],
            branch=data["branch"],
            dead_on_empty=data["dead_on_empty"],
        )

    def _input_signature(self) -> Hashable | None:
        return None  # Implied in state signature

    def _state_signature(self) -> Hashable | None:
        if callable(self.pred):
            return self.id  # SwitchOps with callable predicates are not functional
        pred = tuple(self.pred) if isinstance(self.pred, list) else self.pred
        return (
            self.input_op.id,
            tuple(op.id for op in self.cond_ops),
            pred,
            self.branch,
            self.dead_on_empty,
        )

    def get_prefix_template(
        self, input_templates: dict[str, PrefixType], sliced_op_map: dict[str, str]
    ) -> PrefixType:
        return input_templates[self.input_op.id]

    def get_input_slice(self, data_size: int, input_slices: dict[str, Slice]) -> Slice:
        input_slice = input_slices[self.inputs[0].id]
        for inp in self.inputs[1:]:
            if input_slices[inp.id] != input_slice:
                raise ValueError("Different input slices found.")
        return input_slice


def switch(
    input_op: list[str] | Op,
    cond_ops: Sequence[list[str] | Op],
    pred: Predicate | None = None,
    branch: bool = True,
) -> SwitchOp:
    return SwitchOp(input_op, cond_ops, pred, branch, dead_on_empty=False)


def cond(
    input_op: list[str] | Op,
    cond_ops: Sequence[list[str] | Op],
    pred: Predicate | None = None,
) -> tuple[SwitchOp, SwitchOp]:
    return (
        SwitchOp(input_op, cond_ops, pred, branch=True, dead_on_empty=False),
        SwitchOp(input_op, cond_ops, pred, branch=False, dead_on_empty=False),
    )


@Op.registry.register("MergeOp")
class MergeOp(FunctionalOp):
    def __init__(self, *inputs: Op) -> None:
        super().__init__(list(inputs))

    def _serialize(self) -> dict[str, Any]:
        return {}

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "MergeOp":
        input_ids = data["_inputs"]
        return cls(*[other_ops[op_id] for op_id in input_ids])

    def _state_signature(self) -> Hashable | None:
        return None  # Implied in input signature

    def get_prefix_template(
        self, input_templates: dict[str, PrefixType], sliced_op_map: dict[str, str]
    ) -> PrefixType:
        op_id = self.id
        op_id = sliced_op_map.get(op_id, op_id)
        return (Placeholder(op_id),)

    def get_input_slice(self, data_size: int, input_slices: dict[str, Slice]) -> Slice:
        input_slice = input_slices[self.inputs[0].id]
        for inp in self.inputs[1:]:
            if input_slices[inp.id] != input_slice:
                raise ValueError("Different input slices found.")
        return input_slice


def loop_merge(*inputs: Op) -> MergeOp:
    return MergeOp(*inputs)


@Op.registry.register("EnterOp")
class EnterOp(FunctionalOp):
    def __init__(self, init_op: list[str] | Op, future_op: Op) -> None:
        init_op = init_op if isinstance(init_op, Op) else DataOp(init_op)
        super().__init__([init_op, future_op])

    @property
    def init_op(self) -> Op:
        return self.inputs[0]

    @property
    def future_op(self) -> Op:
        return self.inputs[1]

    def _serialize(self) -> dict[str, Any]:
        return dict(init_op=self.init_op.id, future_op=self.future_op.id)

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "EnterOp":
        input_ids = data["_inputs"]
        if len(input_ids) != 2:
            raise ValueError("EnterOp must have two inputs")
        return cls(
            init_op=other_ops[data["init_op"]], future_op=other_ops[data["future_op"]]
        )

    def _state_signature(self) -> Hashable | None:
        return None  # Implied in input signature

    def get_prefix_template(
        self, input_templates: dict[str, PrefixType], sliced_op_map: dict[str, str]
    ) -> PrefixType:
        # TODO: handle prefix template for looping
        raise NotImplementedError("EnterOp currently does not support prefix template.")

    def get_input_slice(self, data_size: int, input_slices: dict[str, Slice]) -> Slice:
        return input_slices[self.init_op.id]


def loop_enter(init_op: list[str] | Op, future_op: Op) -> EnterOp:
    return EnterOp(init_op, future_op)


@Op.registry.register("ExitOp")
class ExitOp(FunctionalOp):
    def __init__(self, op: Op) -> None:
        super().__init__([op])

    @property
    def op(self) -> Op:
        return self.inputs[0]

    def _serialize(self) -> dict[str, Any]:
        return {}

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "ExitOp":
        input_ids = data["_inputs"]
        if len(input_ids) != 1:
            raise ValueError("ExitOp must have exactly one input")
        return cls(op=other_ops[input_ids[0]])

    def _state_signature(self) -> Hashable | None:
        return None  # Implied in input signature

    def get_prefix_template(
        self, input_templates: dict[str, PrefixType], sliced_op_map: dict[str, str]
    ) -> PrefixType:
        return input_templates[self.op.id]

    def get_input_slice(self, data_size: int, input_slices: dict[str, Slice]) -> Slice:
        return input_slices[self.op.id]


def loop_exit(op: Op) -> ExitOp:
    return ExitOp(op)


class _SubGraph:
    def __init__(self, input_ops: list[Op], output_ops: list[Op]) -> None:
        self.input_ops = input_ops
        self.output_ops = output_ops
        self.graph: dict[str, Op]
        """op_id -> Op"""
        self.dependencies: dict[Op, set[Op]]
        """op -> dependent ops"""
        self.external_dependencies: dict[Op, set[Op]]
        """op -> external dependencies (ops depended on)"""
        self._build_subgraph()

    def _traverse_graph(
        self,
        op: Op,
        end_ops: set[Op],
        graph: dict[str, Op],
        dependencies: dict[Op, set[Op]],
    ) -> None:
        assert op.id not in graph

        graph[op.id] = op

        if op in end_ops:
            return
        for input_op in op.inputs:
            # Add the dependencies of this op.
            if input_op in dependencies:
                dependencies[input_op].add(op)
            else:
                dependencies[input_op] = {op}

            # Recursively traverse the graph.
            if input_op.id not in graph:
                self._traverse_graph(input_op, end_ops, graph, dependencies)

    def _prune_subgraph(
        self,
        input_ops: list[Op],
        graph: dict[str, Op],
        dependencies: dict[Op, set[Op]],
        external_dependencies: dict[Op, set[Op]],
    ) -> None:
        """Prune the subgraph to include only the ops that are reachable from the
        input ops and minimal set of external dependencies."""

        dependent_ops: set[Op] = set()

        def traverse_down(op: Op) -> None:
            """Traverse down the graph to capture all dependent ops."""
            cur_ops = dependencies.get(op)
            if cur_ops is None:
                return
            for cur_op in cur_ops:
                if cur_op in dependent_ops:
                    continue
                # All related ops must have been captured in the graph.
                if cur_op.id not in graph:
                    raise ValueError(
                        f"External Op '{cur_op.__class__.__name__}(id={cur_op.id})' "
                        "accessing internal context is found."
                    )
                dependent_ops.add(cur_op)
                traverse_down(cur_op)

        # Get all dependent ops.
        for input_op in input_ops:
            traverse_down(input_op)

        internal_ops = set(input_ops) | dependent_ops
        external_ops: set[Op] = set()
        for op in dependent_ops:
            for input_op in op.inputs:
                # External dependencies are non-dependent ops that are depended
                # on by dependent ops.
                if input_op not in internal_ops:
                    if op in external_dependencies:
                        external_dependencies[op].add(input_op)
                    else:
                        external_dependencies[op] = {input_op}
                    external_ops.add(input_op)

        # Prune the graph in place.
        required_ops = internal_ops | external_ops
        for op_id, op in list(graph.items()):
            if op not in required_ops:
                graph.pop(op_id, None)
                dependencies.pop(op, None)

    def _build_subgraph(self) -> None:
        graph: dict[str, Op] = {}
        dependencies: dict[Op, set[Op]] = {}
        input_ops = set(self.input_ops)

        # Build the subgraph starting from the output ops.
        for op in self.output_ops:
            self._traverse_graph(op, input_ops, graph, dependencies)

        # Prune the subgraph to include only the dependent ops and a minimal set
        # of external dependencies.
        external_dependencies: dict[Op, set[Op]] = {}
        self._prune_subgraph(self.input_ops, graph, dependencies, external_dependencies)

        self.graph = graph
        self.dependencies = dependencies
        self.external_dependencies = external_dependencies

    def insert_after(self, op: Op, new_op: Op) -> None:
        """Insert a new op after an existing op in the subgraph."""
        if op.id not in self.graph:
            raise ValueError("Op to insert after is not in the subgraph")

        # Insert to the actual graph.
        dep_ops = self.dependencies[op]
        for dep_op in dep_ops:
            dep_op.replace_input(op, new_op)

        # Update the subgraph.
        self.graph[new_op.id] = new_op

        # Update the dependencies.
        if new_op in self.dependencies:
            self.dependencies[new_op].update(dep_ops)
        else:
            self.dependencies[new_op] = dep_ops
        self.dependencies[op] = {new_op}

    @property
    def external_ops(self) -> list[Op]:
        return list({op for ops in self.external_dependencies.values() for op in ops})

    def add(self, op: Op) -> None:
        """Add an op to the subgraph."""
        self.graph[op.id] = op
        for input_op in op.inputs:
            if input_op.id not in self.graph:
                raise ValueError("Input of the given op is not in the subgraph")
            if input_op in self.dependencies:
                self.dependencies[input_op].add(op)
            else:
                self.dependencies[input_op] = {op}

    def duplicate(self) -> "_SubGraph":
        """Duplicates the subgraph, excluding input ops"""
        # Duplicate all internal nodes except input nodes and external dependencies.
        external_dependencies = {
            op for ops in self.external_dependencies.values() for op in ops
        }
        # op -> new_op
        internal_nodes: dict[Op, Op] = {
            op: op.copy()
            for op in self.graph.values()
            if not (op in self.input_ops or op in external_dependencies)
        }

        # Update the dependencies of the new internal ops.
        new_output_ops: list[Op] = []
        dependencies = self.dependencies
        for op, new_op in internal_nodes.items():
            if op in dependencies:
                for depending_op in dependencies[op]:
                    internal_nodes[depending_op].replace_input(op, new_op)
            # Identify new output ops.
            if op in self.output_ops:
                new_output_ops.append(new_op)

        return self.__class__(self.input_ops, new_output_ops)

    def replace_inputs(self, new_input_map: dict[Op, Op]) -> None:
        """Replaces the inputs of the subgraph with new inputs

        Parameters
        ----------
        new_input_map : dict[Op, Op]
            A mapping from old input ops to new input ops.
            The old input ops must be in the subgraph.
        """
        new_inputs: list[Op] = []
        for old_input, new_input in new_input_map.items():
            if old_input not in self.input_ops:
                raise ValueError(f"Old input op {old_input.id} is not in the subgraph")
            if new_input.id in self.graph:
                raise ValueError(
                    f"New input op {new_input.id} already exists in the subgraph"
                )
            # Update the dependencies of ops depending on the old input.
            for depending_op in self.dependencies.get(old_input, set()):
                depending_op.replace_input(old_input, new_input)
            new_inputs.append(new_input)
            # Update the graph
            del self.graph[old_input.id]
            self.graph[new_input.id] = new_input
        # Update the input ops
        self.input_ops = new_inputs


def branch(
    input_ops: list[list[str] | Op],
    cond_ops: list[list[str] | Op],
    pred: Predicate,
) -> tuple[list[SwitchOp], list[SwitchOp]]:
    true_ops: list[SwitchOp] = []
    false_ops: list[SwitchOp] = []
    for inp_op in input_ops:
        true_ops.append(
            SwitchOp(
                input_op=inp_op,
                cond_ops=cond_ops,
                pred=pred,
                branch=True,
                dead_on_empty=True,
            )
        )
        false_ops.append(
            SwitchOp(
                input_op=inp_op,
                cond_ops=cond_ops,
                pred=pred,
                branch=False,
                dead_on_empty=True,
            )
        )
    return true_ops, false_ops


def while_loop(
    input_ops: list[Op],
    output_ops: list[Op | None],
    pred: Predicate | None,
    max_iter: int | None = 10,
) -> list[ExitOp]:
    # Validate the inputs.
    if len(input_ops) != len(output_ops):
        raise ValueError("Input and output ops must have the same length")

    subgraph = _SubGraph(input_ops, [op for op in output_ops if op is not None])

    # Control ops that need to run one iteration more than the loop body.
    control_ops: set[Op] = set()

    # Insert Switch ops.
    loop_ops: list[Op] = []
    loop_output_ops: list[SwitchOp] = []
    for input_op, output_op in zip(input_ops, output_ops):
        if output_op is None:
            # Case 1: No output ops. => Loop input ops.
            true_op = SwitchOp(
                input_op=input_op,
                cond_ops=input_ops,
                pred=pred,
                branch=True,
                dead_on_empty=True,
            )
            subgraph.insert_after(input_op, true_op)
            loop_ops.append(true_op)
            control_ops.add(true_op)
        else:
            # Case 2: With output ops.
            true_op = SwitchOp(
                input_op=input_op,
                cond_ops=input_ops,
                pred=pred,
                branch=True,
                dead_on_empty=True,
            )
            false_op = SwitchOp(
                input_op=input_op,
                cond_ops=input_ops,
                pred=pred,
                branch=False,
                dead_on_empty=False,
            )
            subgraph.insert_after(input_op, true_op)
            control_ops.add(true_op)
            subgraph.add(false_op)
            control_ops.add(false_op)
            loop_ops.append(output_op)
            loop_output_ops.append(false_op)

    # Insert Enter ops.
    for input_op, loop_op in zip(input_ops, loop_ops):
        enter_op = EnterOp(init_op=input_op, future_op=FutureOp.wrap(loop_op))
        subgraph.insert_after(input_op, enter_op)
        control_ops.add(enter_op)

    # Handle external dependencies.
    external_ops = subgraph.external_ops
    for op in external_ops:
        # Insert Switch ops.
        true_op = SwitchOp(
            input_op=op, cond_ops=input_ops, pred=pred, branch=True, dead_on_empty=True
        )
        subgraph.insert_after(op, true_op)
        control_ops.add(true_op)
        # Insert Enter ops.
        enter_op = EnterOp(init_op=op, future_op=FutureOp.wrap(true_op))
        subgraph.insert_after(op, enter_op)
        control_ops.add(enter_op)

    # Mark the ops that are part of the loop.
    non_looping_op_ids = [op.id for op in (input_ops + external_ops)]
    for op_id, op in subgraph.graph.items():
        if op_id not in non_looping_op_ids:
            if op in control_ops:
                op.set_looping(
                    max_iter=max_iter if max_iter is None else (max_iter + 1)
                )
            else:
                op.set_looping(max_iter=max_iter)

    exit_ops = [ExitOp(op) for op in loop_output_ops]

    return exit_ops


def loop(
    input_ops: Sequence[Op | None], output_ops: Sequence[Op | None], num_iter: int
) -> list[Op]:
    # Validate the inputs.
    if len(input_ops) != len(output_ops):
        raise ValueError("Input and output ops must have the same length")
    if num_iter < 1:
        raise ValueError("Number of iterations must be at least 1")

    new_input_ops: list[Op] = []
    new_output_ops: list[Op] = []
    # Track which output ops will be used for subsequent iterations.
    output_mask: list[bool] = []
    for input_op, output_op in zip(input_ops, output_ops):
        if output_op is not None:
            new_output_ops.append(output_op)
            if input_op is None:
                output_mask.append(False)
            else:
                new_input_ops.append(input_op)
                output_mask.append(True)

    subgraph = _SubGraph(new_input_ops, new_output_ops)
    for _ in range(num_iter - 1):
        new_output_ops = [op for m, op in zip(output_mask, new_output_ops) if m]
        new_input_map = dict(zip(new_input_ops, new_output_ops))
        subgraph = subgraph.duplicate()
        subgraph.replace_inputs(new_input_map)
        new_input_ops, new_output_ops = subgraph.input_ops, subgraph.output_ops

    return new_output_ops
