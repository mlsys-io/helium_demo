import asyncio
import multiprocessing as mp
import queue
from abc import ABC, abstractmethod
from queue import Empty, Full
from typing import Generic, TypeVar

from helium.runtime.utils.logger import get_default_logger

logger = get_default_logger()

T = TypeVar("T")


class CloseSignal:
    pass


_CLOSE_SIGNAL = CloseSignal()
_DELAY: float = 0


class AsyncQueue(ABC, Generic[T]):
    @abstractmethod
    async def get(self) -> T:
        pass

    @abstractmethod
    def get_nowait(self) -> T:
        pass

    @abstractmethod
    async def get_all(self) -> list[T]:
        pass

    @abstractmethod
    async def put(self, obj: T) -> None:
        pass

    @abstractmethod
    def put_nowait(self, obj: T) -> None:
        pass

    @abstractmethod
    def empty(self) -> bool:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class AIOQueue(AsyncQueue, Generic[T]):
    """
    Asynchronous wrapper for asyncio queue.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[T | CloseSignal] = asyncio.Queue()
        self._is_closed: bool = False

    def _unwrap(self, message: T | CloseSignal) -> T:
        if isinstance(message, CloseSignal):
            self._is_closed = True
            raise asyncio.CancelledError()
        return message

    async def get(self) -> T:
        return self._unwrap(await self._queue.get())

    def get_nowait(self) -> T:
        return self._unwrap(self._queue.get_nowait())

    async def get_all(self) -> list[T]:
        items: list[T] = [await self.get()]
        while not self._queue.empty():
            item = self._queue.get_nowait()
            if isinstance(item, CloseSignal):
                self._is_closed = True
                break
            items.append(item)
        return items

    async def put(self, obj: T) -> None:
        if self._is_closed:
            raise asyncio.CancelledError()
        await self._queue.put(obj)

    def put_nowait(self, obj: T) -> None:
        if self._is_closed:
            raise asyncio.CancelledError()
        self._queue.put_nowait(obj)

    def empty(self) -> bool:
        return self._queue.empty()

    def close(self) -> None:
        if not self._is_closed:
            self._is_closed = True
            self._queue.put_nowait(_CLOSE_SIGNAL)


class TSQueue(AsyncQueue, Generic[T]):
    """
    Asynchronous wrapper for thread-safe queue.
    """

    def __init__(self, delay: float = _DELAY) -> None:
        self._queue: queue.Queue[T | CloseSignal] = queue.Queue()
        self._delay: float = delay
        self._is_closed: bool = False

    def _unwrap(self, message: T | CloseSignal) -> T:
        if isinstance(message, CloseSignal):
            self._is_closed = True
            raise asyncio.CancelledError()
        return message

    async def get(self, delay: float | None = None) -> T:
        delay = self._delay if delay is None else delay
        while True:
            try:
                return self.get_nowait()
            except Empty:
                await asyncio.sleep(delay)

    def get_nowait(self) -> T:
        return self._unwrap(self._queue.get_nowait())

    async def get_all(self) -> list[T]:
        items: list[T] = [await self.get()]
        while not self._queue.empty():
            item = self._queue.get_nowait()
            if isinstance(item, CloseSignal):
                self._is_closed = True
                break
            items.append(item)
        return items

    async def put(self, obj: T, delay: float | None = None) -> None:
        delay = self._delay if delay is None else delay
        while True:
            try:
                self.put_nowait(obj)
                return
            except Full:
                await asyncio.sleep(delay)

    def put_nowait(self, obj: T) -> None:
        if self._is_closed:
            raise asyncio.CancelledError()
        self._queue.put_nowait(obj)

    def empty(self) -> bool:
        return self._queue.empty()

    def close(self) -> None:
        if not self._is_closed:
            self._is_closed = True
            self._queue.put_nowait(_CLOSE_SIGNAL)


class MPQueue(AsyncQueue, Generic[T]):
    """
    Asynchronous wrapper for multiprocessing queue.
    """

    def __init__(self, delay: float = _DELAY) -> None:
        self._queue: mp.Queue[T | CloseSignal] = mp.Queue()
        self._delay: float = delay
        self._is_closed: bool = False

    def _unwrap(self, message: T | CloseSignal) -> T:
        if isinstance(message, CloseSignal):
            self._is_closed = True
            raise asyncio.CancelledError()
        return message

    async def get(self, delay: float | None = None) -> T:
        delay = self._delay if delay is None else delay
        while True:
            try:
                return self.get_nowait()
            except Empty:
                await asyncio.sleep(delay)

    def get_nowait(self) -> T:
        return self._unwrap(self._queue.get_nowait())

    async def get_all(self) -> list[T]:
        items: list[T] = [await self.get()]
        while not self._queue.empty():
            item = self._queue.get_nowait()
            if isinstance(item, CloseSignal):
                self._is_closed = True
                break
            items.append(item)
        return items

    async def put(self, obj: T, delay: float | None = None) -> None:
        delay = self._delay if delay is None else delay
        while True:
            try:
                self.put_nowait(obj)
                return
            except Full:
                await asyncio.sleep(delay)

    def put_nowait(self, obj: T) -> None:
        if self._is_closed:
            raise asyncio.CancelledError()
        self._queue.put_nowait(obj)

    def empty(self) -> bool:
        return self._queue.empty()

    def close(self) -> None:
        if not self._is_closed:
            self._is_closed = True
            self._queue.put_nowait(_CLOSE_SIGNAL)
