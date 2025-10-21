from typing import Any, Hashable

from helium.common import Slice
from helium.ops.data_ops import DataOp, MessageOp, OpMessage
from helium.ops.ops import FunctionalOp, Op
from helium.utils.prefix.radix_tree import (
    MessageDelimiter,
    MessagePrefixType,
    Placeholder,
    PrefixType,
    TextKeyItem,
    TextPrefixType,
    prefix_message,
    to_message_prefix,
    to_text_prefix,
)


@Op.registry.register("LastMessageOp")
class LastMessageOp(FunctionalOp):
    def __init__(self, messages: list[OpMessage] | Op) -> None:
        messages = messages if isinstance(messages, Op) else MessageOp(messages)
        super().__init__([messages])

    @property
    def messages(self) -> Op:
        return self.inputs[0]

    def _serialize(self) -> dict[str, Any]:
        return {}

    @classmethod
    def _from_json(
        cls, data: dict[str, Any], other_ops: dict[str, "Op"]
    ) -> "LastMessageOp":
        inputs = data["_inputs"]
        if len(inputs) != 1:
            raise ValueError("LastMessageOp must have exactly one input")
        return cls(other_ops[inputs[0]])

    def _state_signature(self) -> Hashable | None:
        return None  # Implied in input signature

    def get_prefix_template(
        self, input_templates: dict[str, PrefixType], sliced_op_map: dict[str, str]
    ) -> TextPrefixType:
        parent_template = to_message_prefix(input_templates[self.messages.id])
        if not parent_template:
            raise ValueError("LastMessageOp got empty message list")

        last_message: list[TextKeyItem] = []
        start = False
        for item in reversed(parent_template):
            match item:
                case MessageDelimiter.ROLE:
                    assert False
                case MessageDelimiter.CONTENT:
                    break
                case MessageDelimiter.END:
                    assert not start
                    start = True
                case _:
                    if start:
                        last_message.append(item)
                    else:
                        assert isinstance(item, Placeholder)
                        last_message.append(item)
                        break
        last_message.reverse()

        return tuple(last_message)

    def get_input_slice(self, data_size: int, input_slices: dict[str, Slice]) -> Slice:
        return input_slices[self.messages.id]


def get_last_message(messages: list[OpMessage] | Op) -> LastMessageOp:
    return LastMessageOp(messages)


@Op.registry.register("AppendMessageOp")
class AppendMessageOp(FunctionalOp):
    def __init__(
        self,
        messages: list[OpMessage] | Op,
        content: Op | OpMessage | str | list[str],
        role: str = "user",
    ):
        messages = messages if isinstance(messages, Op) else MessageOp(messages)
        inputs: list[Op] = [messages]
        content_str: str | None = None

        if isinstance(content, OpMessage):
            role = content.role if role is None else role
            content = content.content
            if isinstance(content, Op):
                inputs.append(content)
            else:
                content_str = content
        elif isinstance(content, list):
            inputs.append(DataOp(data=content))
        elif isinstance(content, Op):
            inputs.append(content)
        else:
            content_str = content

        super().__init__(inputs)
        self.role = role
        self.content_str = content_str

    @property
    def messages(self) -> Op:
        return self.inputs[0]

    @property
    def content(self) -> str | Op:
        return self.inputs[1] if self.content_str is None else self.content_str

    def _serialize(self) -> dict[str, Any]:
        return dict(role=self.role, content_str=self.content_str)

    @classmethod
    def _from_json(
        cls, data: dict[str, Any], other_ops: dict[str, "Op"]
    ) -> "AppendMessageOp":
        inputs = data["_inputs"]
        if len(inputs) != 2:
            raise ValueError("AppendMessageOp must have exactly two inputs")
        content_str = data["content_str"]
        content = other_ops[inputs[1]] if content_str is None else content_str
        return cls(other_ops[inputs[0]], content, data["role"])

    def _state_signature(self) -> Hashable | None:
        return self.role, self.content_str

    def get_prefix_template(
        self, input_templates: dict[str, PrefixType], sliced_op_map: dict[str, str]
    ) -> MessagePrefixType:
        messages_template = to_message_prefix(input_templates[self.messages.id])
        content = self.content
        content_template = to_text_prefix(
            input_templates[content.id] if isinstance(content, Op) else (content,)
        )
        return messages_template + prefix_message(self.role, content_template)

    def get_input_slice(self, data_size: int, input_slices: dict[str, Slice]) -> Slice:
        input_slice = input_slices[self.inputs[0].id]
        for inp in self.inputs[1:]:
            if input_slices[inp.id] != input_slice:
                raise ValueError("Different input slices found.")
        return input_slice


def append_message(
    messages: Op | list[OpMessage],
    content: Op | OpMessage | list[str],
    role: str = "user",
) -> AppendMessageOp:
    return AppendMessageOp(messages, content, role)
