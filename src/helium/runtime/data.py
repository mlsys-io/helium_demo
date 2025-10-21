import enum
from collections.abc import Callable, Iterable, Iterator

from helium.common import Message


class DataType(enum.Enum):
    TEXT = enum.auto()
    MESSAGE = enum.auto()


class MessageData:
    __slots__ = ("_role", "content")

    content: list[str]

    def __init__(self, role: str | None, content: list[str]) -> None:
        self._role = role
        self.content = content

    @classmethod
    def empty(cls) -> "MessageData":
        return cls(None, [])

    @property
    def role(self) -> str:
        if self._role is None:
            raise ValueError("Role has not been set")
        return self._role

    def __len__(self) -> int:
        return len(self.content)

    def _get_role(self, other: "MessageData") -> str | None:
        if other._role is None:
            return self._role
        if self._role is None:
            return other._role
        if self._role == other._role:
            return self._role
        raise ValueError("Inconsistent roles")

    def __add__(self, other: "MessageData") -> "MessageData":
        return self.__class__(self._get_role(other), self.content + other.content)

    def __iadd__(self, other: "MessageData") -> "MessageData":
        self._role = self._get_role(other)
        self.content += other.content
        return self


class MessageList:
    __slots__ = ("_messages",)

    def __init__(self, messages: list[MessageData]) -> None:
        # Validate messages.
        if len(messages) > 0:
            num_contents = len(messages[0])
            if any(len(m) != num_contents for m in messages):
                raise ValueError("Inconsistent number of messages")
        self._messages = messages

    @classmethod
    def from_messages(cls, messages: list[list[Message]]) -> "MessageList":
        if len(messages) == 0:
            return cls([])
        num_contents = len(messages[0])
        message_data_list: list[MessageData] = []
        for msg in messages:
            if len(msg) != num_contents:
                raise ValueError("Inconsistent number of messages")
        for i in range(num_contents):
            role = None
            content: list[str] = []
            for msgs in messages:
                m = msgs[i]
                if role is None:
                    role = m.role
                elif role != m.role:
                    raise ValueError("Inconsistent roles")
                content.append(m.content)
            message_data_list.append(MessageData(role, content))
        return cls(message_data_list)

    def get(self, index: int) -> MessageData | None:
        if index >= len(self._messages):
            return None
        return self._messages[index]

    def __add__(self, other: "MessageList") -> "MessageList":
        if len(self._messages) != len(other._messages):
            raise ValueError("Inconsistent number of messages")
        messages = [(m1 + m2) for m1, m2 in zip(self._messages, other._messages)]
        return self.__class__(messages)

    def __iadd__(self, other: "MessageList") -> "MessageList":
        if len(self._messages) != len(other._messages):
            raise ValueError("Inconsistent number of messages")
        for m1, m2 in zip(self._messages, other._messages):
            m1 += m2
        return self

    def __len__(self) -> int:
        if len(self._messages) == 0:
            return 0
        return len(self._messages[0])

    @property
    def num_messages(self) -> int:
        return len(self._messages)

    def copy(self) -> "MessageList":
        return self.__class__(
            [MessageData(msg.role, msg.content.copy()) for msg in self._messages]
        )

    def append(self, message: MessageData) -> None:
        if len(self._messages) > 0:
            num_contents = len(self._messages[0])
            new_num_contents = len(message)
            if new_num_contents != num_contents:
                if new_num_contents == 1:
                    message = MessageData(message.role, message.content * num_contents)
                else:
                    raise ValueError("Inconsistent number of messages")
        self._messages.append(message)

    def into_empty(self) -> "MessageList":
        return self.__class__([MessageData(msg.role, []) for msg in self._messages])

    def filter(self, preds: list[bool]) -> "MessageList":
        if self.num_messages == 0:
            return self.__class__([])
        if len(preds) != len(self):
            raise ValueError("Inconsistent number of predicates")
        messages = [
            MessageData(msg.role, [c for c, p in zip(msg.content, preds) if p])
            for msg in self._messages
        ]
        return self.__class__(messages)

    def get_by_indices(self, indices: list[int]) -> "MessageList":
        if self.num_messages == 0:
            return self.__class__([])
        messages = []
        for msg_data in self._messages:
            content = [msg_data.content[i] for i in indices]
            messages.append(MessageData(msg_data.role, content))
        return self.__class__(messages)

    def pop_by_indices(self, indices: list[int]) -> "MessageList":
        if self.num_messages == 0:
            return self.__class__([])
        index_map = {j: i for i, j in enumerate(indices)}
        cur_messages = []
        ret_messages = []
        for msg_data in self._messages:
            cur_msg_content = []
            ret_msg_content = [""] * len(indices)
            for i, content in enumerate(msg_data.content):
                if i in index_map:
                    ret_msg_content[index_map[i]] = content
                else:
                    cur_msg_content.append(content)
            cur_messages.append(MessageData(msg_data.role, cur_msg_content))
            ret_messages.append(MessageData(msg_data.role, ret_msg_content))

        self._messages = cur_messages
        return self.__class__(ret_messages)

    def sort(self, indices: list[int]) -> "MessageList":
        if self.num_messages == 0:
            return self.__class__([])
        if len(indices) != len(self):
            raise ValueError("Inconsistent number of indices")
        sorted_indices = sorted(range(len(indices)), key=lambda i: indices[i])
        return self.__class__(
            [
                MessageData(msg.role, [msg.content[i] for i in sorted_indices])
                for msg in self._messages
            ]
        )

    def __iter__(self) -> Iterator[list[Message]]:
        if self.num_messages == 0:
            return
        for i in range(len(self)):
            yield [Message(msg.role, msg.content[i]) for msg in self._messages]


class Data:
    __slots__ = ("dtype", "data", "_indices")

    dtype: DataType
    data: list[str] | MessageList

    def __init__(
        self,
        dtype: DataType,
        data: list[str] | MessageList,
        indices: list[int] | None,
    ) -> None:
        self._validate_inputs(dtype, data, indices)
        self.dtype = dtype
        self.data = data
        self._indices = indices

    @property
    def indices(self) -> list[int]:
        if self._indices is None:
            raise ValueError("Indices have not been set")
        return self._indices

    @indices.setter
    def indices(self, indices: list[int]) -> None:
        self._indices = indices

    @staticmethod
    def _validate_inputs(
        dtype: DataType,
        data: list[str] | MessageList,
        indices: list[int] | None,
    ) -> None:
        if dtype == DataType.TEXT:
            Data._validate_text_data(data, indices)
        elif dtype == DataType.MESSAGE:
            Data._validate_message_data(data, indices)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    @staticmethod
    def _validate_text_data(
        data: list[str] | MessageList, indices: list[int] | None
    ) -> None:
        if not isinstance(data, list):
            raise ValueError("Data type is TEXT, but data is not a list")
        if indices is None:
            raise ValueError("Indices must be provided for TEXT data")
        if len(data) != len(indices):
            raise ValueError("TEXT data and indices must have the same length")

    @staticmethod
    def _validate_message_data(
        data: list[str] | MessageList, indices: list[int] | None
    ) -> None:
        if not isinstance(data, MessageList):
            raise ValueError("Data type is MESSAGE, but data is not a MessageList")
        if data.num_messages > 0:
            if indices is None:
                raise ValueError("Indices must be provided for non-empty MESSAGE data")
            if len(data) != len(indices):
                raise ValueError("MESSAGE data and indices must have the same length")

    def into_empty(self, dtype: DataType | None) -> "Data":
        dtype = dtype or self.dtype
        if dtype == DataType.TEXT:
            return self.__class__.text([], [])
        if self.is_text():
            return self.__class__.message(MessageList([]), [])
        assert isinstance(self.data, MessageList)
        return self.__class__.message(self.data.into_empty(), [])

    @classmethod
    def text(cls, data: list[str], indices: list[int] | None) -> "Data":
        if indices is None:
            indices = list(range(len(data)))
        return cls(DataType.TEXT, data, indices)

    @classmethod
    def message(cls, data: MessageList, indices: list[int] | None) -> "Data":
        if data.num_messages > 0 and indices is None:
            indices = list(range(len(data)))
        return cls(DataType.MESSAGE, data, indices)

    def copy(self) -> "Data":
        return self.__class__(self.dtype, self.data.copy(), self.indices.copy())

    def is_text(self) -> bool:
        return self.dtype == DataType.TEXT

    def is_message(self) -> bool:
        return self.dtype == DataType.MESSAGE

    def is_empty(self) -> bool:
        return len(self) == 0

    def as_text(self) -> list[str]:
        assert self.dtype == DataType.TEXT
        return self.data  # type: ignore

    def as_message(self) -> MessageList:
        assert self.dtype == DataType.MESSAGE
        return self.data  # type: ignore

    def into_text(self, data: list[str]) -> "Data":
        return self.__class__.text(data=data, indices=self.indices)

    def into_message(self, data: MessageList) -> "Data":
        return self.__class__.message(data=data, indices=self.indices)

    def convert_text(
        self, func: Callable[[list[str]], Iterable[list[Message]]]
    ) -> "Data":
        if not self.is_text():
            raise ValueError("Data is not text")
        msgs: list[Message]
        message_data_list: list[MessageData] = []
        for msgs in func(self.as_text()):
            role = None
            content: list[str] = []
            for msg in msgs:
                if role is None:
                    role = msg.role
                elif role != msg.role:
                    raise ValueError("Inconsistent roles")
                content.append(msg.content)
            message_data_list.append(MessageData(role, content))
        return self.into_message(MessageList(message_data_list))

    def convert_message(
        self, func: Callable[[list[list[Message]]], list[str]]
    ) -> "Data":
        if not self.is_message():
            raise ValueError("Data is not message")
        msgs = [msg for _, msg in self.iter_message()]
        return self.into_text(func(msgs))

    def filter(self, preds: list[bool]) -> "Data":
        if len(preds) != len(self.indices):
            raise ValueError("Inconsistent number of predicates")
        if isinstance(self.data, MessageList):
            return self.__class__(
                self.dtype,
                self.data.filter(preds),
                [i for i, p in zip(self.indices, preds) if p],
            )
        new_data = []
        indices = []
        for d, i, p in zip(self.data, self.indices, preds):
            if p:
                new_data.append(d)
                indices.append(i)
        return self.__class__(self.dtype, new_data, indices)

    def _get_message_content_indices(self, indices: list[int]) -> list[int]:
        # This also checks the presence of indices in self.indices
        index_map = {j: i for i, j in enumerate(self.indices)}
        return [index_map[i] for i in indices]

    def get_by_indices(self, indices: list[int], uncheck: bool = False) -> "Data":
        if uncheck:
            indices = [i for i in indices if i in self.indices]

        if indices == self.indices:
            return self.copy()

        msg_indices = self._get_message_content_indices(indices)
        if isinstance(self.data, MessageList):
            new_msg_data = self.data.get_by_indices(msg_indices)
            return self.__class__(self.dtype, new_msg_data, indices)
        new_txt_data = [self.data[i] for i in msg_indices]
        return self.__class__(self.dtype, new_txt_data, indices)

    def pop_by_indices(self, indices: list[int]) -> "Data":
        if indices == self.indices:
            ret = self.copy()
            self.data = [] if isinstance(self.data, list) else self.data.into_empty()
            self.indices = []
            return ret

        msg_indices = self._get_message_content_indices(indices)
        if isinstance(self.data, MessageList):
            # Also modify self.data in place
            new_msg_data = self.data.pop_by_indices(msg_indices)
            # Set to remaining indices
            self.indices = [i for i in self.indices if i not in indices]
            return self.__class__(self.dtype, new_msg_data, indices)
        index_map = {j: i for i, j in enumerate(indices)}
        new_txt_data = []
        new_indices = []
        ret_data = [""] * len(indices)
        for d, i in zip(self.data, self.indices):
            if i in index_map:
                ret_data[index_map[i]] = d
            else:
                new_txt_data.append(d)
                new_indices.append(i)
        self.data = new_txt_data
        self.indices = new_indices
        return self.__class__(self.dtype, ret_data, indices)

    def sort(self) -> "Data":
        data: list[str] | MessageList
        if self.is_text():
            indices = []
            data = []
            for i, d in sorted(zip(self.indices, self.as_text())):
                indices.append(i)
                data.append(d)
        else:
            assert isinstance(self.data, MessageList)
            indices = self.indices
            data = self.data.sort(indices)
            indices = list(range(len(self.indices)))
        return self.__class__(self.dtype, data, indices)

    def __add__(self, other: "Data") -> "Data":
        if self.dtype != other.dtype:
            raise ValueError("Data types must match")
        if any(i in other.indices for i in self.indices):
            raise ValueError("Overlapping indices found")
        return self.__class__(
            self.dtype, self.data + other.data, self.indices + other.indices  # type: ignore
        )

    def __radd__(self, other: "Data") -> "Data":
        return other + self

    def __len__(self) -> int:
        return len(self.data)

    def iter_text(self) -> Iterable[tuple[int, str]]:
        for i, d in zip(self.indices, self.as_text()):
            yield i, d

    def iter_message(self) -> Iterable[tuple[int, list[Message]]]:
        for i, d in zip(self.indices, self.as_message()):
            yield i, d

    def iter_content(
        self,
    ) -> Iterable[tuple[int, str]] | Iterable[tuple[int, list[Message]]]:
        return self.iter_text() if self.is_text() else self.iter_message()

    def get_text(self, index: int) -> str:
        for i, d in zip(self.indices, self.as_text()):
            if i == index:
                return d
        raise ValueError(f"Index {index} not found")

    def get_message(self, index: int) -> list[Message]:
        for i, d in zip(self.indices, self.as_message()):
            if i == index:
                return d
        raise ValueError(f"Index {index} not found")
