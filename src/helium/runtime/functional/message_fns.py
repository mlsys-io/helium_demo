from collections.abc import AsyncGenerator

from helium.runtime.data import Data, DataType, MessageData, MessageList
from helium.runtime.functional.fns import FnInput
from helium.runtime.worker.worker_input import ResultPuller, WorkerArg


class MessageFnInput(FnInput):
    NAME: str = "message"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        max_iter: int | None,
        roles: list[str],
        message_refs: list[int | str],
        inputs: list[WorkerArg],
    ):
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=max_iter,
            args=inputs,
        )
        self.roles = roles
        self.message_refs = message_refs

    @property
    def output_type(self) -> DataType:
        return DataType.MESSAGE

    async def _run(self, puller: ResultPuller) -> AsyncGenerator[Data | None, None]:
        if len(self.roles) != len(self.message_refs):
            raise ValueError("Inconsistent number of roles and messages")

        # Find reference args and message data to broadcast.
        messages: list[MessageData] = []
        to_broadcast: list[MessageData] = []
        ref_num_contents, ref_arg = 0, None
        for role, ref in zip(self.roles, self.message_refs):
            if isinstance(ref, int):
                arg = self.args[ref]
                message_content = arg.data.as_text()
                if ref_arg is None:
                    # Set reference arg.
                    ref_num_contents = len(message_content)
                    ref_arg = arg
                elif len(message_content) != ref_num_contents:
                    # Consistency check.
                    raise ValueError("Inconsistent number of message contents")
                message = MessageData(role=role, content=message_content)
            else:
                # Broadcastable message data.
                message = MessageData(role=role, content=[ref])
                to_broadcast.append(message)
            messages.append(message)

        if ref_arg is None:
            raise ValueError("There must be at least one message with data.")

        ref_indices = ref_arg.data.indices
        # Broadcast the message contents of broadcastable messages.
        for message in to_broadcast:
            message.content *= ref_num_contents
        # Check consistency of indices.
        for arg in self.args:
            if arg.data.indices != ref_indices:
                raise ValueError("Inconsistent data ordering")
        yield Data.message(MessageList(messages), ref_indices)


class AppendMessageFnInput(FnInput):
    NAME: str = "append-message"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        max_iter: int | None,
        messages: WorkerArg,
        content: str | WorkerArg,
        role: str,
    ) -> None:
        self.role = role
        self.content_str: str | None
        args = [messages]
        if isinstance(content, str):
            self.content_str = content
        else:
            self.content_str = None
            args.append(content)
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=max_iter,
            args=args,
        )

    @property
    def output_type(self) -> DataType:
        return DataType.MESSAGE

    async def _run(self, puller: ResultPuller) -> AsyncGenerator[Data | None, None]:
        content: str | WorkerArg
        if self.content_str is None:
            messages_arg, content = self.args
        else:
            messages_arg = self.args[0]
            content = self.content_str

        messages = messages_arg.data.as_message().copy()

        if isinstance(content, str):
            role = self.role
            new_message_content = [content]
        else:
            content_data = content.data
            if content_data.is_text():
                role = self.role
                new_message_content = content_data.as_text()
            else:
                assert content_data.is_message()
                if len(content_data) != 1:
                    raise ValueError("Content must be a single message.")
                new_message = content_data.as_message().get(0)
                assert new_message is not None
                role = new_message.role if self.role is None else self.role
                new_message_content = new_message.content

        # This broadcasts the message content internally.
        messages.append(MessageData(role=role, content=new_message_content))

        yield messages_arg.data.into_message(messages)


class LastMessageFnInput(FnInput):
    NAME: str = "last-message"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        max_iter: int | None,
        messages: WorkerArg,
    ) -> None:
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=max_iter,
            args=[messages],
        )

    @property
    def output_type(self) -> DataType:
        return DataType.TEXT

    async def _run(self, puller: ResultPuller) -> AsyncGenerator[Data | None, None]:
        arg = self.args[0]
        data = arg.data
        messages = data.as_message()
        last_message = messages.get(-1)
        if last_message is None:
            # TODO: Consider this case.
            yield data.into_empty(DataType.TEXT)
        else:
            yield data.into_text(last_message.content)
