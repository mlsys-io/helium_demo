from collections.abc import Sequence
from typing import Any, Hashable, cast

from helium.common import Message, Slice
from helium.ops.ops import FunctionalOp, Op
from helium.utils import check_and_cast_list
from helium.utils.prefix.radix_tree import (
    MessageKeyItem,
    MessagePrefixType,
    Placeholder,
    PrefixType,
    TextPrefixType,
    prefix_message,
    to_text_prefix,
)


class OpMessage:
    role: str
    content: str | Op

    def __init__(self, role: str, content: str | list[str] | Op) -> None:
        self.role = role
        self.content = DataOp(data=content) if isinstance(content, list) else content

    def content_or_id(self) -> str:
        return self.content.id if isinstance(self.content, Op) else self.content


@Op.registry.register("DataOp")
class DataOp(FunctionalOp):
    data: list[str]

    def __init__(self, data: list[str]) -> None:
        super().__init__()
        self.data = data

    def _serialize(self) -> dict[str, Any]:
        return dict(data=self.data)

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "DataOp":
        return cls(data=data["data"])

    def _state_signature(self) -> Hashable | None:
        return self.id  # Merging DataOps is not supported

    def get_prefix_template(
        self, input_templates: dict[str, PrefixType], sliced_op_map: dict[str, str]
    ) -> TextPrefixType:
        op_id = self.id
        op_id = sliced_op_map.get(op_id, op_id)
        return (Placeholder(op_id),)

    def get_input_slice(self, data_size: int, input_slices: dict[str, Slice]) -> Slice:
        return Slice((0, data_size))


@Op.registry.register("MessageOp")
class MessageOp(FunctionalOp):
    def __init__(self, messages: list[OpMessage]) -> None:
        inputs: list[Op] = []
        message_refs: list[int | str] = []
        for msg in messages:
            if isinstance(msg.content, str):
                message_refs.append(msg.content)
            else:
                try:
                    i = inputs.index(msg.content)
                    message_refs.append(i)
                except ValueError:
                    message_refs.append(len(inputs))
                    inputs.append(msg.content)
        super().__init__(inputs)
        self._message_refs = message_refs
        self._roles = [msg.role for msg in messages]

    @property
    def messages(self) -> list[OpMessage]:
        return [
            OpMessage(
                role=role, content=self.inputs[ref] if isinstance(ref, int) else ref
            )
            for role, ref in zip(self._roles, self._message_refs)
        ]

    @property
    def roles(self) -> list[str]:
        return self._roles

    @property
    def message_refs(self) -> list[int | str]:
        return self._message_refs

    def _serialize(self) -> dict[str, Any]:
        return dict(
            messages=[
                {"role": message.role, "content": message.content_or_id()}
                for message in self.messages
            ]
        )

    @classmethod
    def _from_json(
        cls, data: dict[str, Any], other_ops: dict[str, "Op"]
    ) -> "MessageOp":
        def get_content(content_or_id: str) -> str | Op:
            if content_or_id in other_ops:
                return other_ops[content_or_id]
            return content_or_id

        messages = [
            OpMessage(role=message["role"], content=get_content(message["content"]))
            for message in data["messages"]
        ]
        return cls(messages)

    def _input_signature(self) -> Hashable | None:
        return None  # Implied in state signature

    def _state_signature(self) -> Hashable | None:
        return tuple((msg.role, msg.content_or_id()) for msg in self.messages)

    def get_prefix_template(
        self, input_templates: dict[str, PrefixType], sliced_op_map: dict[str, str]
    ) -> MessagePrefixType:
        template: list[MessageKeyItem] = []
        for message in self.messages:
            message_template = (
                (message.content,)
                if isinstance(message.content, str)
                else input_templates[message.content.id]
            )
            template.extend(
                prefix_message(message.role, to_text_prefix(message_template))
            )
        return tuple(template)

    def get_input_slice(self, data_size: int, input_slices: dict[str, Slice]) -> Slice:
        if len(self.inputs) == 0:
            raise ValueError("No inputs found.")
        input_slice = input_slices[self.inputs[0].id]
        for inp in self.inputs[1:]:
            if input_slices[inp.id] != input_slice:
                raise ValueError("Different input slices found.")
        return input_slice


def message_data(data: Sequence[Message | OpMessage]) -> MessageOp:
    return MessageOp(
        [
            (
                msg
                if isinstance(msg, OpMessage)
                else OpMessage(role="user", content=msg.content)
            )
            for msg in data
        ]
    )


@Op.registry.register("InputOp")
class InputOp(FunctionalOp):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def _serialize(self) -> dict[str, Any]:
        return dict(name=self.name)

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "InputOp":
        return cls(name=data["name"])

    def _state_signature(self) -> Hashable | None:
        return self.id  # Merging InputOps is not supported

    def get_prefix_template(
        self, input_templates: dict[str, PrefixType], sliced_op_map: dict[str, str]
    ) -> TextPrefixType:
        op_id = self.id
        op_id = sliced_op_map.get(op_id, op_id)
        return (Placeholder(op_id),)

    def get_input_slice(self, data_size: int, input_slices: dict[str, Slice]) -> Slice:
        return Slice((0, data_size))


def input_placeholder(name: str) -> InputOp:
    return InputOp(name)


@Op.registry.register("OutputOp")
class OutputOp(FunctionalOp):
    name: str

    def __init__(self, name: str, output: list[str] | Op) -> None:
        inp = output if isinstance(output, Op) else DataOp(output)
        super().__init__([inp])
        self.name = name

    @property
    def op(self) -> Op:
        return self.inputs[0]

    def _serialize(self) -> dict[str, Any]:
        return dict(name=self.name)

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "OutputOp":
        input_ids = data["_inputs"]
        if len(input_ids) != 1:
            raise ValueError("OutputOp must have exactly one input")
        return cls(name=data["name"], output=other_ops[input_ids[0]])

    def _state_signature(self) -> Hashable | None:
        return self.id  # Merging OutputOps is not supported

    def get_prefix_template(
        self, input_templates: dict[str, PrefixType], sliced_op_map: dict[str, str]
    ) -> PrefixType:
        return input_templates[self.op.id]

    def get_input_slice(self, data_size: int, input_slices: dict[str, Slice]) -> Slice:
        return input_slices[self.op.id]


def as_output(name: str, output: list[str] | Op) -> OutputOp:
    return OutputOp(name, output)


def data(
    data: str | list[str] | list[Message | OpMessage],
) -> DataOp | MessageOp:
    if isinstance(data, str):
        return DataOp(data=[data])
    if len(data) == 0:
        return DataOp(data=cast(list, data))
    if isinstance(data[0], str):
        return DataOp(data=check_and_cast_list(str, data))
    return message_data(cast(list[Message | OpMessage], data))
