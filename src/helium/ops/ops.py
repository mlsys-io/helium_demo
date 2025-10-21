import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Hashable, Self, TypeVar

from class_registry import ClassRegistry

from helium.common import Message, Slice
from helium.utils import unique_id
from helium.utils.prefix.radix_tree import PrefixType

T = TypeVar("T")
SingleDtype = str | list[Message]


class OpRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs) -> Callable[[T], T]:
        return cls.registry.register(*args, **kwargs)

    @classmethod
    def keys(cls) -> list[str]:
        return cls.registry.keys()

    @classmethod
    def items(cls) -> dict[str, "Op"]:
        return dict(cls.registry.items())

    @classmethod
    def get(cls, key: Any) -> "Op":
        return cls.registry.get(key)

    @classmethod
    def get_class(cls, key: Any) -> "Op":
        return cls.registry.get_class(key)


class Op(ABC):
    id: str
    inputs: list["Op"]
    registry: OpRegistry = OpRegistry()

    def __init__(self, inputs: list["Op"] | None = None) -> None:
        self.id = unique_id()
        self.inputs = [] if inputs is None else inputs
        self._max_iter: int | None = None

    @property
    def looping(self) -> bool:
        return self._max_iter is not None

    @property
    def max_iter(self) -> int | None:
        """
        Maximum number of iterations. There are three cases:
        - None: No looping.
        - -1: Infinite looping.
        - n: Looping for n times.
        """
        return self._max_iter

    def copy(self, new_id: bool = True) -> Self:
        new_op = copy.copy(self)
        # Create new unique attributes
        if new_id:
            new_op.id = unique_id()
        new_op.inputs = new_op.inputs.copy()
        return new_op

    def set_looping(self, looping: bool = True, max_iter: int | None = None) -> None:
        """
        Set the looping parameters.

        Parameters
        ----------
        looping : bool
            Whether to enable looping.
        max_iter : int | None
            Maximum number of iterations. There are three cases:
            - None: No looping.
            - -1: Infinite looping.
            - n: Looping for n times.
        """
        if looping:
            self._max_iter = -1 if max_iter is None else max_iter
        else:
            self._max_iter = None

    def _traverse_graph(self, graph: dict[str, "Op"]):
        assert self.id not in graph

        graph[self.id] = self
        for op in self.inputs:
            if op.id not in graph:
                op._traverse_graph(graph)

    def traverse_graph(self, graph: dict[str, "Op"] | None = None) -> dict[str, "Op"]:
        graph = {} if graph is None else graph
        self._traverse_graph(graph)
        return graph

    def replace_input(self, old_op: "Op", new_op: "Op") -> None:
        for i, op in enumerate(self.inputs):
            if op.id == old_op.id:
                self.inputs[i] = new_op

    @abstractmethod
    def _serialize(self) -> dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "Op":
        pass

    def serialize(self) -> dict[str, Any]:
        serialized = self._serialize()
        serialized["_id"] = self.id
        serialized["_op"] = self.__class__.__name__
        serialized["_max_iter"] = self._max_iter
        serialized["_inputs"] = [op.id for op in self.inputs]
        return serialized

    @classmethod
    def from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "Op":
        op = cls._from_json(data, other_ops)
        op.id = data["_id"]
        max_iter = data["_max_iter"]
        op.set_looping(max_iter is not None, max_iter)
        return op

    def __hash__(self) -> int:
        return hash(self.id)

    def signature(self) -> Hashable:
        state_signature = self._state_signature()
        if state_signature == self.id:
            return self.id  # Guaranteed to be unique
        input_signature = self._input_signature()
        if state_signature is None:
            return (self.__class__, input_signature)
        if input_signature is None:
            return (self.__class__, state_signature)
        return (self.__class__, input_signature, state_signature)

    def _input_signature(self) -> Hashable | None:
        return tuple(op.id for op in self.inputs)

    @abstractmethod
    def _state_signature(self) -> Hashable | None:
        pass

    @abstractmethod
    def get_prefix_template(
        self, input_templates: dict[str, PrefixType], sliced_op_map: dict[str, str]
    ) -> PrefixType:
        pass

    @abstractmethod
    def get_input_slice(self, data_size: int, input_slices: dict[str, Slice]) -> Slice:
        pass


class FunctionalOp(Op):
    pass
