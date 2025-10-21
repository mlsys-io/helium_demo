import re
from collections.abc import Callable, Sequence
from typing import Any, Hashable

from helium.common import Slice
from helium.ops.data_ops import DataOp
from helium.ops.ops import FunctionalOp, Op, SingleDtype
from helium.utils.prefix.radix_tree import (
    Placeholder,
    PrefixType,
    TextKeyItem,
    TextPrefixType,
    to_text_prefix,
)


@Op.registry.register("FormatOp")
class FormatOp(FunctionalOp):
    template: str

    def __init__(
        self, template: str, *args: list[str] | Op, **kwargs: list[str] | Op
    ) -> None:
        format_args: list[int] = []
        format_kwargs: dict[str, int] = {}
        inputs: list[Op] = []

        for arg in args:
            arg = arg if isinstance(arg, Op) else DataOp(arg)
            try:
                i = inputs.index(arg)
                format_args.append(i)
            except ValueError:
                format_args.append(len(inputs))
                inputs.append(arg)

        for k, v in kwargs.items():
            v = v if isinstance(v, Op) else DataOp(v)
            try:
                i = inputs.index(v)
                format_kwargs[k] = i
            except ValueError:
                format_kwargs[k] = len(inputs)
                inputs.append(v)

        super().__init__(inputs)
        self.template = template
        self._format_args = format_args
        self._format_kwargs = format_kwargs

    @property
    def format_args(self) -> list[Op]:
        return [self.inputs[i] for i in self._format_args]

    @property
    def format_kwargs(self) -> dict[str, Op]:
        return {k: self.inputs[i] for k, i in self._format_kwargs.items()}

    def _serialize(self) -> dict[str, Any]:
        format_args = [arg.id for arg in self.format_args]
        format_kwargs = {k: v.id for k, v in self.format_kwargs.items()}
        return dict(
            template=self.template, format_args=format_args, format_kwargs=format_kwargs
        )

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "FormatOp":
        format_args = [other_ops[arg] for arg in data["format_args"]]
        format_kwargs = {k: other_ops[v] for k, v in data["format_kwargs"].items()}
        return cls(data["template"], *format_args, **format_kwargs)

    def _input_signature(self) -> Hashable | None:
        return None  # Implied in state signature

    def _state_signature(self) -> Hashable | None:
        return (
            self.template,
            tuple(arg.id for arg in self.format_args),
            tuple((k, v.id) for k, v in self.format_kwargs.items()),
        )

    def get_prefix_template(
        self, input_templates: dict[str, PrefixType], sliced_op_map: dict[str, str]
    ) -> TextPrefixType:
        formatted = self.template.format(
            *[f"<|op_id={v.id}|>" for v in self.format_args],
            **{k: f"<|op_id={v.id}|>" for k, v in self.format_kwargs.items()},
        )
        op_ids = re.findall(r"<\|op_id=(\w+?)\|>", formatted)
        split = re.split(r"<\|op_id=\w+?\|>", formatted)
        template: list[TextKeyItem] = []
        for s, op_id in zip(split, op_ids):
            if s:
                template.append(s)
            op_template = to_text_prefix(input_templates[op_id])
            template.extend(op_template)
        s = split[-1]
        if s:
            template.append(s)
        return tuple(template)

    def get_input_slice(self, data_size: int, input_slices: dict[str, Slice]) -> Slice:
        if len(self.inputs) == 0:
            raise ValueError("No inputs found.")
        input_slice = input_slices[self.inputs[0].id]
        for inp in self.inputs[1:]:
            if input_slices[inp.id] != input_slice:
                raise ValueError("Different input slices found.")
        return input_slice


def format_op(template: str, *args: list[str] | Op, **kwargs: list[str] | Op) -> Op:
    return FormatOp(template, *args, **kwargs)


@Op.registry.register("FutureOp")
class FutureOp(FunctionalOp):
    op_id: str

    def __init__(self, op_id: str, op: Op | None = None) -> None:
        inputs = [] if op is None else [op]
        super().__init__(inputs)
        self.op_id = op_id
        self._op: Op | None = None

    def _serialize(self) -> dict[str, Any]:
        return dict(op_id=self.op_id)

    @property
    def op(self) -> Op:
        if self._op is None:
            raise ValueError("FutureOp not resolved")
        return self._op

    @classmethod
    def wrap(cls, op: Op) -> "FutureOp":
        return cls(op.id, op)

    def resolve(self, graph: dict[str, "Op"]) -> None:
        if self._op is not None:
            return
        if self.inputs:
            op = self.inputs.pop()
            self.op_id = op.id
            assert not self.inputs
        else:
            op = graph[self.op_id]
        self._op = op

    def update(self, op: Op) -> None:
        if self._op is None:
            raise ValueError("FutureOp not resolved")
        self._op = op

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "FutureOp":
        return cls(data["op_id"])

    def _traverse_graph(self, graph: dict[str, "Op"]) -> None:
        assert self.id not in graph

        graph[self.id] = self
        op = self.inputs[0] if self.inputs else self.op
        if op is not None and op.id not in graph:
            op._traverse_graph(graph)

    def _input_signature(self) -> Hashable | None:
        return None  # Implied in state signature

    def _state_signature(self) -> Hashable | None:
        return self.op_id

    def get_prefix_template(
        self, input_templates: dict[str, PrefixType], sliced_op_map: dict[str, str]
    ) -> PrefixType:
        # TODO: handle prefix template for looping
        raise NotImplementedError(
            "FutureOp currently does not support prefix template."
        )

    def get_input_slice(self, data_size: int, input_slices: dict[str, Slice]) -> Slice:
        return input_slices[self.op_id]


def loop_future(op: Op) -> FutureOp:
    return FutureOp.wrap(op)


@Op.registry.register("LambdaOp")
class LambdaOp(Op):
    fn: Callable[[tuple[SingleDtype, ...]], str]

    def __init__(
        self,
        inputs: list[list[str] | Op],
        fn: Callable[[tuple[SingleDtype, ...]], str],
    ) -> None:
        input_ops: list[Op] = []
        for inp in inputs:
            if isinstance(inp, Op):
                input_ops.append(inp)
            else:
                input_ops.append(DataOp(inp))
        super().__init__(input_ops)
        self.fn = fn

    def _serialize(self) -> dict[str, Any]:
        raise ValueError("LambdaOp is not serializable")

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "LambdaOp":
        raise ValueError("LambdaOp is not serializable")

    def _state_signature(self) -> Hashable | None:
        return self.id  # Merging LambdaOps is not supported

    def get_prefix_template(
        self, input_templates: dict[str, PrefixType], sliced_op_map: dict[str, str]
    ) -> PrefixType:
        op_id = self.id
        op_id = sliced_op_map.get(op_id, op_id)
        return (Placeholder(op_id),)

    def get_input_slice(self, data_size: int, input_slices: dict[str, Slice]) -> Slice:
        if len(self.inputs) == 0:
            raise ValueError("No inputs found.")
        input_slice = input_slices[self.inputs[0].id]
        for inp in self.inputs[1:]:
            if input_slices[inp.id] != input_slice:
                raise ValueError("Different input slices found.")
        return input_slice


def lambda_op(
    inputs: list[list[str] | Op], fn: Callable[[tuple[SingleDtype, ...]], str]
) -> LambdaOp:
    return LambdaOp(inputs, fn)  # type: ignore


@Op.registry.register("SliceOp")
class SliceOp(FunctionalOp):
    indices: Slice

    def __init__(self, op: list[str] | Op, indices: Slice) -> None:
        inp = op if isinstance(op, Op) else DataOp(op)
        super().__init__([inp])
        self.indices = indices

    @property
    def inp(self) -> Op:
        return self.inputs[0]

    def _serialize(self) -> dict[str, Any]:
        return self.indices.serialize()

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "SliceOp":
        indices = Slice.deserialize(data)
        input_ids = data["_inputs"]
        if len(input_ids) != 1:
            raise ValueError("SliceOp must have exactly one input")
        return cls(op=other_ops[input_ids[0]], indices=indices)

    def _state_signature(self) -> Hashable | None:
        return self.indices.to_hashable()

    def get_prefix_template(
        self, input_templates: dict[str, PrefixType], sliced_op_map: dict[str, str]
    ) -> PrefixType:
        return input_templates[self.inp.id]

    def get_input_slice(self, data_size: int, input_slices: dict[str, Slice]) -> Slice:
        input_slice = input_slices[self.inp.id]
        if self.indices not in input_slice:
            raise ValueError("Slice indices not contained in input slice.")
        return self.indices


def slice_op(op: list[str] | Op, indices: Slice) -> SliceOp:
    return SliceOp(op, indices)


@Op.registry.register("ConcatOp")
class ConcatOp(FunctionalOp):
    def __init__(self, ops: Sequence[list[str] | Op] | None = None) -> None:
        inputs = (
            []
            if ops is None
            else [op if isinstance(op, Op) else DataOp(op) for op in ops]
        )
        super().__init__(inputs)

    def add_op(self, op: Op) -> None:
        self.inputs.append(op)

    def _serialize(self) -> dict[str, Any]:
        return {}

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "ConcatOp":
        inputs = data["_inputs"]
        return cls([other_ops[i] for i in inputs])

    def _state_signature(self) -> Hashable | None:
        return None  # Implied in input signature

    def get_prefix_template(
        self, input_templates: dict[str, PrefixType], sliced_op_map: dict[str, str]
    ) -> PrefixType:
        if len(self.inputs) == 0:
            raise ValueError("No inputs found.")
        prefix_template = input_templates[self.inputs[0].id]
        for inp in self.inputs[1:]:
            if input_templates[inp.id] != prefix_template:
                op_id = self.id
                op_id = sliced_op_map.get(op_id, op_id)
                prefix_template = (Placeholder(op_id),)
                break
        return prefix_template

    def get_input_slice(self, data_size: int, input_slices: dict[str, Slice]) -> Slice:
        if len(self.inputs) == 0:
            raise ValueError("No inputs found.")
        return Slice.merge([input_slices[inp.id] for inp in self.inputs])


def concat(ops: Sequence[list[str] | Op]) -> ConcatOp:
    return ConcatOp(ops)
