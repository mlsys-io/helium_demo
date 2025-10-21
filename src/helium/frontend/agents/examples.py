from helium import ops
from helium.common import GenerationConfig
from helium.frontend.agents import Agent
from helium.ops import InputOp, OpMessage, OutputOp
from helium.runtime import HeliumServerConfig
from helium.runtime.protocol import HeliumRequestConfig, HeliumResponse


class CompletionAgent(Agent):
    def __init__(
        self,
        input_op: InputOp | None = None,
        output_name: str = "response",
        name: str | None = None,
        server_config: HeliumServerConfig | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        if input_op is None:
            input_op = ops.input_placeholder(name="user_inputs")
        self.input_name = input_op.name
        super().__init__(
            name=name,
            server_config=server_config,
            input_op=input_op,
            output_name=output_name,
            generation_config=generation_config,
        )

    def build_ops(
        self,
        input_op: InputOp,
        output_name: str,
        generation_config: GenerationConfig | None = None,
    ) -> list[OutputOp]:
        llm_completion = ops.llm_completion(prompt=input_op, config=generation_config)
        output = ops.as_output(name=output_name, output=llm_completion)
        return [output]

    def run(
        self,
        config: HeliumRequestConfig | None = None,
        *args,
        inputs: list[str],
        **kwargs,
    ) -> HeliumResponse:
        if args or kwargs:
            raise ValueError(f"Unexpected arguments: args={args}, kwargs={kwargs}")
        return super().run(config, **{self.input_name: inputs})

    async def run_async(
        self,
        config: HeliumRequestConfig | None = None,
        *args,
        inputs: list[str],
        **kwargs,
    ) -> HeliumResponse:
        if args or kwargs:
            raise ValueError(f"Unexpected arguments: args={args}, kwargs={kwargs}")
        return await super().run_async(config, **{self.input_name: inputs})


class ChatAgent(Agent):
    def __init__(
        self,
        input_op: InputOp | None = None,
        messages: list[OpMessage] | None = None,
        output_name: str = "response",
        name: str | None = None,
        system_prompt: str | None = None,
        generation_config: GenerationConfig | None = None,
        server_config: HeliumServerConfig | None = None,
    ) -> None:
        if input_op is None:
            input_op = ops.input_placeholder(name="user_inputs")
        self.input_name = input_op.name

        if messages is None:
            messages = self.create_messages(input_op, system_prompt)

        super().__init__(
            name=name,
            server_config=server_config,
            messages=messages,
            output_name=output_name,
            generation_config=generation_config,
        )
        self.system_prompt = system_prompt

    @staticmethod
    def create_messages(
        input_op: InputOp, system_prompt: str | None = None
    ) -> list[OpMessage]:
        messages = []
        if system_prompt is not None:
            messages.append(OpMessage(role="system", content=system_prompt))
        messages.append(OpMessage(role="user", content=input_op))
        return messages

    def build_ops(
        self,
        messages: list[OpMessage],
        output_name: str,
        generation_config: GenerationConfig | None = None,
    ) -> list[OutputOp]:
        message_op = ops.message_data(messages)
        llm_chat = ops.llm_chat(
            messages=message_op, config=generation_config, return_history=False
        )
        output = ops.as_output(name=output_name, output=llm_chat)
        return [output]

    def run(
        self,
        config: HeliumRequestConfig | None = None,
        *args,
        inputs: list[str],
        **kwargs,
    ) -> HeliumResponse:
        if args or kwargs:
            raise ValueError(f"Unexpected arguments: args={args}, kwargs={kwargs}")
        return super().run(config, **{self.input_name: inputs})

    async def run_async(
        self,
        config: HeliumRequestConfig | None = None,
        *args,
        inputs: list[str],
        **kwargs,
    ) -> HeliumResponse:
        if args or kwargs:
            raise ValueError(f"Unexpected arguments: args={args}, kwargs={kwargs}")
        return await super().run_async(config, **{self.input_name: inputs})
