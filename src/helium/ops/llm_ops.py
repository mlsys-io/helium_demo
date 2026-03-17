from typing import Any, Hashable

from helium.common import GenerationConfig, Slice
from helium.ops.data_ops import DataOp, MessageOp, OpMessage
from helium.ops.ops import Op
from helium.utils.prefix.radix_tree import (
    MessagePrefixType,
    Placeholder,
    PrefixType,
    TextPrefixType,
    to_message_prefix,
    to_text_prefix,
)


class LLMOp(Op):
    config: GenerationConfig
    cacheable: bool

    def __init__(
        self,
        inputs: list["Op"] | None,
        config: GenerationConfig | None,
        cacheable: bool,
    ) -> None:
        super().__init__(inputs)
        self.config = config or GenerationConfig.from_env()
        self.cacheable = cacheable

    @property
    def llm_service(self) -> str | None:
        return self.config.llm_service

    def _serialize(self) -> dict[str, Any]:
        raise NotImplementedError()

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "LLMOp":
        raise NotImplementedError()

    def _state_signature(self) -> Hashable | None:
        raise NotImplementedError()

    def get_prefix_template(
        self, input_templates: dict[str, PrefixType], sliced_op_map: dict[str, str]
    ) -> PrefixType:
        raise NotImplementedError()

    def get_input_prefix_template(
        self, input_templates: dict[str, PrefixType]
    ) -> PrefixType:
        raise NotImplementedError()

    def get_input_slice(self, data_size: int, input_slices: dict[str, Slice]) -> Slice:
        raise NotImplementedError()


@Op.registry.register("LLMCompletionOp")
class LLMCompletionOp(LLMOp):
    echo: bool

    def __init__(
        self,
        prompt: list[str] | Op,
        config: GenerationConfig | None = None,
        echo: bool = False,
        cacheable: bool = False,
    ) -> None:
        prompt = prompt if isinstance(prompt, Op) else DataOp(data=prompt)
        super().__init__([prompt], config, cacheable)
        self.echo = echo

    @property
    def prompt(self) -> Op:
        return self.inputs[0]

    def _serialize(self) -> dict[str, Any]:
        return dict(
            prompt=self.prompt.id,
            config=self.config.to_dict(),
            echo=self.echo,
            cacheable=self.cacheable,
        )

    @classmethod
    def _from_json(
        cls, data: dict[str, Any], other_ops: dict[str, "Op"]
    ) -> "LLMCompletionOp":
        config = GenerationConfig(**data["config"])
        return cls(other_ops[data["prompt"]], config, data["echo"], data["cacheable"])

    def _state_signature(self) -> Hashable | None:
        return (tuple(self.config.to_dict().items()), self.echo, self.cacheable)

    def get_prefix_template(
        self, input_templates: dict[str, PrefixType], sliced_op_map: dict[str, str]
    ) -> TextPrefixType:
        op_id = self.id
        op_id = sliced_op_map.get(op_id, op_id)
        history = self.get_input_prefix_template(input_templates) if self.echo else ()
        return history + (Placeholder(op_id),)

    def get_input_prefix_template(
        self, input_templates: dict[str, PrefixType]
    ) -> TextPrefixType:
        return to_text_prefix(input_templates[self.prompt.id])

    def get_input_slice(self, data_size: int, input_slices: dict[str, Slice]) -> Slice:
        return input_slices[self.prompt.id]


def llm_completion(
    prompt: list[str] | Op,
    config: GenerationConfig | None = None,
    echo: bool = False,
    cacheable: bool = False,
) -> LLMCompletionOp:
    return LLMCompletionOp(prompt, config, echo, cacheable)


@Op.registry.register("LLMChatOp")
class LLMChatOp(LLMOp):
    config: GenerationConfig
    return_history: bool

    def __init__(
        self,
        messages: list[OpMessage] | Op,
        config: GenerationConfig | None = None,
        return_history: bool = False,
        cacheable: bool = False,
    ) -> None:
        messages = messages if isinstance(messages, Op) else MessageOp(messages)
        super().__init__([messages], config, cacheable)
        self.return_history = return_history

    @property
    def messages(self) -> Op:
        return self.inputs[0]

    def _serialize(self) -> dict[str, Any]:
        return dict(
            messages=self.messages.id,
            config=self.config.to_dict(),
            return_history=self.return_history,
            cacheable=self.cacheable,
        )

    @classmethod
    def _from_json(
        cls, data: dict[str, Any], other_ops: dict[str, "Op"]
    ) -> "LLMChatOp":
        message = other_ops[data["messages"]]
        config = GenerationConfig(**data["config"])
        return_history = data["return_history"]
        cacheable = data["cacheable"]
        return cls(message, config, return_history, cacheable)

    def _state_signature(self) -> Hashable | None:
        return (
            tuple(self.config.to_dict().items()),
            self.return_history,
            self.cacheable,
        )

    def get_prefix_template(
        self, input_templates: dict[str, PrefixType], sliced_op_map: dict[str, str]
    ) -> PrefixType:
        op_id = self.id
        op_id = sliced_op_map.get(op_id, op_id)
        history = (
            self.get_input_prefix_template(input_templates)
            if self.return_history
            else ()
        )
        return history + (Placeholder(op_id),)

    def get_input_prefix_template(
        self, input_templates: dict[str, PrefixType]
    ) -> MessagePrefixType:
        return to_message_prefix(input_templates[self.messages.id])

    def get_input_slice(self, data_size: int, input_slices: dict[str, Slice]) -> Slice:
        return input_slices[self.messages.id]


def llm_chat(
    messages: list[OpMessage] | Op,
    config: GenerationConfig | None = None,
    return_history: bool = False,
    cacheable: bool = False,
) -> LLMChatOp:
    return LLMChatOp(messages, config, return_history, cacheable)
