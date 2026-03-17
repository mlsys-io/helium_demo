from typing import Any, Hashable

from helium.common import Slice
from helium.ops.ops import Op
from helium.ops.util_ops import ConcatOp
from helium.utils.prefix.radix_tree import Placeholder, PrefixType


@Op.registry.register("CacheFetchOp")
class CacheFetchOp(Op):
    def __init__(self, cached_op: Op, cached_data: Any):
        super().__init__([cached_op])
        self._cached_op: Op | None = None
        self.cached_data = cached_data

    @property
    def cached_op(self) -> Op:
        if self._cached_op is not None:
            return self._cached_op
        return self.inputs[0]

    def resolve(self) -> None:
        if self._cached_op is None:
            self._cached_op = self.inputs[0]
            self.inputs = []

    def unresolve(self) -> None:
        if self._cached_op is not None and not (
            isinstance(self._cached_op, ConcatOp) and len(self._cached_op.inputs) == 0
        ):
            self.inputs = [self._cached_op]
            self._cached_op = None

    def _serialize(self) -> dict[str, Any]:
        raise RuntimeError("CacheFetchOp cannot be serialized")

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "Op":
        raise RuntimeError("CacheFetchOp cannot be deserialized")

    def _state_signature(self) -> Hashable | None:
        return self.id

    def get_prefix_template(
        self, input_templates: dict[str, PrefixType], sliced_op_map: dict[str, str]
    ) -> PrefixType:
        op_id = self.cached_op.id
        op_id = sliced_op_map.get(op_id, op_id)
        if op_id in input_templates:
            return input_templates[op_id]
        return (Placeholder(op_id),)

    def get_input_slice(self, data_size: int, input_slices: dict[str, Slice]) -> Slice:
        cached_data = self.cached_data
        if isinstance(cached_data, dict):
            return Slice(list(cached_data))
        return Slice(cached_data.indices)
