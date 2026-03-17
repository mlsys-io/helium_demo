import asyncio
import multiprocessing as mp
from collections.abc import Callable, Iterable, Iterator, MutableMapping
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")

_DELAY: float = 0.01


class AsyncPool(Generic[K, V]):
    """
    An asynchronous data pool based on dictionary.
    """

    def __init__(self) -> None:
        self._pool: MutableMapping[K, V] = {}
        lock = asyncio.Lock()
        self._get_cond = asyncio.Condition(lock)
        self._put_cond = asyncio.Condition(lock)

    async def wait(self, key: K) -> None:
        """Waits until the key is removed from the pool"""
        async with self._put_cond:
            await self._put_cond.wait_for(lambda: key not in self._pool)

    async def get(self, key: K) -> V:
        async with self._get_cond:
            await self._get_cond.wait_for(lambda: key in self._pool)
            return self._pool[key]

    async def get_nowait(self, key: K, default: V | None = None) -> V | None:
        async with self._get_cond:
            return self._pool.get(key, default)

    async def get_all(self) -> dict[K, V]:
        async with self._get_cond:
            await self._get_cond.wait_for(lambda: len(self._pool) > 0)
            return dict(self._pool)

    async def pop(self, key: K) -> V:
        async with self._get_cond:
            await self._get_cond.wait_for(lambda: key in self._pool)
            ret = self._pool.pop(key)
            self._put_cond.notify_all()
            return ret

    async def pop_first(self) -> tuple[K, V]:
        async with self._get_cond:
            await self._get_cond.wait_for(lambda: len(self._pool) > 0)
            key = next(iter(self._pool))
            ret = self._pool.pop(key)
            self._put_cond.notify_all()
            return key, ret

    async def pop_nowait(self, key: K, default: V | None = None) -> V | None:
        async with self._put_cond:
            ret = self._pool.pop(key, default)
            self._put_cond.notify_all()
            return ret

    async def pop_all(self) -> dict[K, V]:
        async with self._get_cond:
            await self._get_cond.wait_for(lambda: len(self._pool) > 0)
            ret = dict(self._pool)
            self._pool.clear()
            self._put_cond.notify_all()
            return ret

    async def put(self, key: K, obj: V, exist_ok: bool = False) -> None:
        if not exist_ok:
            await self.put_nowait(key, obj, exist_ok)
            return

        async with self._put_cond:
            await self._put_cond.wait_for(lambda: key not in self._pool)
            self._pool[key] = obj
            self._get_cond.notify_all()

    async def put_nowait(self, key: K, obj: V, exist_ok: bool = False) -> None:
        if (not exist_ok) and (key in self._pool):
            raise ValueError("Key already exists")

        async with self._get_cond:
            self._pool[key] = obj
            self._get_cond.notify_all()

    async def put_all_no_wait(self, items: dict[K, V], exist_ok: bool = False) -> None:
        async with self._get_cond:
            for key, obj in items.items():
                if (not exist_ok) and (key in self._pool):
                    raise ValueError(f"Key {key} already exists")
                self._pool[key] = obj
            self._get_cond.notify_all()

    def empty(self) -> bool:
        return len(self._pool) == 0

    def dump(self) -> dict[K, V]:
        return dict(self._pool)


class MPAsyncPool(AsyncPool[K, V]):
    """
    An asynchronous data pool based on shared dictionary.
    """

    def __init__(self, delay: float = _DELAY) -> None:
        self._manager = mp.Manager()
        self._pool = self._manager.dict()
        self._delay: float = delay

    async def get(self, key: K, delay: float | None = None) -> V:
        delay = self._delay if delay is None else delay
        while True:
            if key in self._pool:
                return self._pool[key]
            await asyncio.sleep(delay)

    async def get_nowait(self, key: K, default: V | None = None) -> V | None:
        return self._pool.get(key, default)

    async def pop(self, key: K, delay: float | None = None) -> V:
        delay = self._delay if delay is None else delay
        while True:
            if key in self._pool:
                return self._pool.pop(key)
            await asyncio.sleep(delay)

    async def pop_nowait(self, key: K, default: V | None = None) -> V | None:
        return self._pool.pop(key, default)

    async def put(
        self, key: K, obj: V, exist_ok: bool = False, delay: float | None = None
    ) -> None:
        delay = self._delay if delay is None else delay
        while True:
            if key not in self._pool:
                self._pool[key] = obj
                return
            if not exist_ok:
                raise ValueError("Key already exists")
            await asyncio.sleep(delay)

    async def put_nowait(self, key: K, obj: V, exist_ok: bool = False) -> None:
        if (not exist_ok) and (key in self._pool):
            raise ValueError("Key already exists")
        self._pool[key] = obj

    def empty(self) -> bool:
        return len(self._pool) == 0

    def dump(self) -> dict[K, V]:
        return dict(self._pool)


class _ItemList(Generic[V]):
    def __init__(self, merge_func: Callable[[V, V], V] | None = None) -> None:
        self._item_list: list[V] = []
        self._item_counter: int = 0
        self._is_committed: bool = False
        self.merge_func = merge_func

    def add(self, obj: V) -> None:
        if self._is_committed:
            raise ValueError("Already committed")

        if self.merge_func is None or len(self._item_list) == 0:
            # Add new item if there is no item or the previous item has been read.
            self._item_list.append(obj)
        else:
            # Merge the new object with the previous object if it has not been read.
            self._item_list[-1] = self.merge_func(self._item_list[-1], obj)

    def pop(self, ticket: int) -> tuple[int | None, V] | None:
        if ticket - self._item_counter != 0:
            raise ValueError("Invalid ticket")
        if len(self._item_list) == 0:
            if self._is_committed:
                raise IndexError("No more items")
            return None

        obj = self._item_list[0]
        if self.merge_func is None:
            count = 1
        else:
            count = len(self._item_list)
            for o in self._item_list[1:]:
                obj = self.merge_func(obj, o)

        # Assign a new ticket
        if self._is_committed and count >= len(self._item_list):
            new_ticket = None  # No more items
        else:
            new_ticket = ticket + count

        # Remove popped items
        self._item_list = self._item_list[count:]
        self._item_counter += count

        return new_ticket, obj

    def is_empty(self) -> bool:
        return self._is_committed and len(self._item_list) == 0

    def commit(self) -> None:
        self._is_committed = True

    @property
    def is_committed(self) -> bool:
        return self._is_committed

    def __iter__(self) -> Iterator[V]:
        return iter(self._item_list)


class StreamingPool(Generic[K, V]):
    """An asynchronous streaming pool based on dictionary."""

    def __init__(self, merge_func: Callable[[V, V], V]) -> None:
        self._merge_func = merge_func
        self._pool: MutableMapping[K, _ItemList[V]] = {}
        lock = asyncio.Lock()
        self._get_cond = asyncio.Condition(lock)
        self._put_cond = asyncio.Condition(lock)

    async def pop_all(self, key: K) -> V:
        if self._merge_func is None:
            raise ValueError("Merge function is not defined")

        obj_list: list[V] = await self.pop_all_no_merge(key)
        ret = obj_list[0]  # Assume at least one object exist
        for obj in obj_list[1:]:
            ret = self._merge_func(ret, obj)
        return ret

    async def pop_all_no_merge(self, key: K) -> list[V]:
        ticket: int | None = 0
        obj_list: list[V] = []
        while ticket is not None:
            ticket, obj = await self.pop(key, ticket)
            obj_list.append(obj)
        return obj_list

    async def pop(self, key: K, ticket: int) -> tuple[int | None, V]:
        ret = None

        def try_pop() -> bool:
            nonlocal ret
            if key in self._pool:
                ret = self._pool[key].pop(ticket)
                return ret is not None
            return False

        async with self._get_cond:
            await self._get_cond.wait_for(try_pop)
            assert ret is not None
            if ret[0] is None:
                self._pool.pop(key)
            self._put_cond.notify_all()
            return ret

    async def pop_nowait(
        self, key: K, ticket: int, default: V | None = None
    ) -> tuple[int | None, V | None]:
        if key not in self._pool:
            return None, default
        async with self._put_cond:
            ret: tuple[int | None, V | None] | None = self._pool[key].pop(ticket)
            if ret is None:
                ret = None, default
            elif ret[0] is None:
                self._pool.pop(key)  # Remove the item list
            self._put_cond.notify_all()
            return ret

    async def put(self, key: K, obj: V, exist_ok: bool = False) -> None:
        def try_put() -> bool:
            if key not in self._pool:
                self._pool[key] = _ItemList(self._merge_func)
            item_list = self._pool[key]
            if not item_list.is_committed:
                self._pool[key].add(obj)
                return True
            return False

        if not exist_ok:
            await self.put_nowait(key, obj, exist_ok)
            return

        async with self._put_cond:
            await self._put_cond.wait_for(try_put)
            self._get_cond.notify_all()

    async def put_nowait(self, key: K, obj: V, exist_ok: bool = False) -> None:
        if key not in self._pool:
            self._pool[key] = _ItemList(self._merge_func)

        async with self._get_cond:
            item_list = self._pool[key]
            if item_list.is_committed:
                if not exist_ok:
                    raise ValueError("Key already exists")
                self._pool[key] = _ItemList(self._merge_func)  # Overwrite
            self._pool[key].add(obj)
            self._get_cond.notify_all()

    async def commit(self, key: K) -> None:
        async with self._put_cond:
            self._pool[key].commit()
            self._put_cond.notify_all()
            self._get_cond.notify_all()

    def empty(self) -> bool:
        return len(self._pool) == 0

    def dump(self) -> dict[K, list[V]]:
        return {
            key: [item for item in item_list] for key, item_list in self._pool.items()
        }


class _RcItem(Generic[V]):
    def __init__(self, obj: V, ref_count: int) -> None:
        self.obj = obj
        self.ref_count = ref_count

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"({', '.join(str(k) + '=' + str(v) for k, v in vars(self).items())})"
        )


class LateCommitException(Exception):
    pass


class _RcItemList(Generic[V]):
    def __init__(
        self, ref_count: int, merge_func: Callable[[V, V], V] | None = None
    ) -> None:
        self._item_list: list[_RcItem[V]] = []
        self._initial_rc = ref_count
        self._item_counter: int = 0
        self._is_committed: bool = False
        self._allocated_tickets: list[int] = []
        self.merge_func = merge_func

    @property
    def ref_count(self) -> int:
        return self._initial_rc

    def add(self, obj: V) -> None:
        if self._is_committed:
            raise ValueError("Already committed")

        if self._initial_rc == 0:
            # Ignore incoming objects if the reference count is zero
            return

        if (
            self.merge_func is None
            or len(self._item_list) == 0
            or self._item_list[-1].ref_count != self._initial_rc
        ):
            # Add new item if there is no item or the previous item has been read.
            self._item_list.append(_RcItem(obj, self._initial_rc))
        else:
            # Merge the new object with the previous object if it has not been read.
            self._item_list[-1].obj = self.merge_func(self._item_list[-1].obj, obj)

    def _get_item_index(self, ticket: int, remove_ticket: bool) -> int | None:
        """Gets the index of the item in the list

        Returns
        -------
        int | None
            A non-negative integer if the ticket is valid, i.e., it refers to an
            item in the list. Otherwise, returns None if the ticket is greater than
            the number of items in the list but the list has not been committed.

        Raises
        ------
        ValueError
            If the ticket is less than the freed item counter.
        LateCommitException
            If the ticket is greater than the number of items in the list and has
            been allocated but the list has been committed.
        """
        was_allocated = ticket in self._allocated_tickets
        item_index = ticket - self._item_counter
        if item_index < 0:
            raise ValueError("Invalid ticket")
        if item_index >= len(self._item_list):
            if not self._is_committed:
                if remove_ticket and was_allocated:
                    self._allocated_tickets.remove(ticket)
                return None
            if ticket == 0 or was_allocated:
                if was_allocated:
                    self._allocated_tickets.remove(ticket)
                raise LateCommitException()
            raise IndexError("No more items")
        if was_allocated:
            self._allocated_tickets.remove(ticket)
        return item_index

    def _pop_item(self, index: int, to_remove: set[int]) -> V:
        item = self._item_list[index]
        if item.ref_count <= 0:
            raise ValueError("Invalid reference count")
        item.ref_count -= 1
        if item.ref_count == 0:
            to_remove.add(index)
        return item.obj

    def _remove_items(self, indices: set[int]) -> None:
        self._item_list = [
            item for i, item in enumerate(self._item_list) if i not in indices
        ]
        self._item_counter += len(indices)

    def pop(self, ticket: int) -> tuple[int | None, V] | None:
        # get_debug_logger().debug(
        #     "Pop: ticket=%s, item_list=%s, item_counter=%s",
        #     ticket,
        #     self._item_list,
        #     self._item_counter,
        # )
        item_index = self._get_item_index(ticket, remove_ticket=False)
        if item_index is None:
            return None

        to_remove: set[int] = set()
        obj = self._pop_item(item_index, to_remove)
        if self.merge_func is None:
            count = 1
        else:
            num_items = len(self._item_list)
            count = num_items - item_index
            for i in range(item_index + 1, num_items):
                obj = self.merge_func(obj, self._pop_item(i, to_remove))

        # Assign a new ticket
        if self._is_committed and item_index + count >= len(self._item_list):
            new_ticket = None  # No more items
        else:
            new_ticket = ticket + count
            self._allocated_tickets.append(new_ticket)

        # Remove items with no remaining references
        # get_debug_logger().debug(
        #     "Pop: to_remove=%s, new_ticket=%s, obj=%s, item_list=%s",
        #     to_remove,
        #     new_ticket,
        #     obj,
        #     self._item_list,
        # )
        self._remove_items(to_remove)

        return new_ticket, obj

    def is_empty(self) -> bool:
        return (
            self._is_committed
            and len(self._item_list) == 0
            and len(self._allocated_tickets) == 0
        )

    def commit(self) -> None:
        self._is_committed = True

    def unsubscribe(self, ticket: int) -> None:
        item_index = self._get_item_index(ticket, remove_ticket=True)
        # Decrement the reference count as incoming objects will be ignored
        self._initial_rc -= 1
        if item_index is None:
            return

        to_remove: set[int] = set()
        for i in range(item_index, len(self._item_list)):
            self._pop_item(i, to_remove)
        self._remove_items(to_remove)

    @property
    def is_committed(self) -> bool:
        return self._is_committed

    def __iter__(self) -> Iterator[_RcItem[V]]:
        return iter(self._item_list)


class RcAsyncPool(Generic[K, V]):
    """
    An asynchronous data pool based on dictionary with reference counting.
    """

    def __init__(self) -> None:
        self._rc_pool: MutableMapping[K, _RcItem[V]] = {}
        lock = asyncio.Lock()
        self._get_cond = asyncio.Condition(lock)
        self._put_cond = asyncio.Condition(lock)

    async def get(self, key: K) -> V:
        async with self._get_cond:
            await self._get_cond.wait_for(lambda: key in self._rc_pool)
            return self._rc_pool[key].obj

    def get_nowait(self, key: K, default: V | None = None) -> V | None:
        if key in self._rc_pool:
            return self._rc_pool[key].obj
        return default

    async def pop(self, key: K) -> V:
        async with self._get_cond:
            await self._get_cond.wait_for(lambda: key in self._rc_pool)
            item = self._rc_pool[key]
            item.ref_count -= 1
            if item.ref_count <= 0:
                ret = self._rc_pool.pop(key).obj
            else:
                ret = item.obj
            self._put_cond.notify_all()
            return ret

    async def pop_nowait(self, key: K, default: V | None = None) -> V | None:
        if key not in self._rc_pool:
            return default
        async with self._put_cond:
            item = self._rc_pool[key]
            item.ref_count -= 1
            if item.ref_count <= 0:
                ret = self._rc_pool.pop(key).obj
            else:
                ret = item.obj
            self._put_cond.notify_all()
            return ret

    async def put(
        self, key: K, obj: V, ref_count: int = 1, exist_ok: bool = False
    ) -> None:
        if not exist_ok:
            await self.put_nowait(key, obj, ref_count, exist_ok)
            return

        async with self._put_cond:
            await self._put_cond.wait_for(lambda: key not in self._rc_pool)
            self._rc_pool[key] = _RcItem(obj, ref_count)
            self._get_cond.notify_all()

    async def put_nowait(
        self, key: K, obj: V, ref_count: int = 1, exist_ok: bool = False
    ) -> None:
        if (not exist_ok) and (key in self._rc_pool):
            raise ValueError("Key already exists")

        async with self._get_cond:
            self._rc_pool[key] = _RcItem(obj, ref_count)
            self._get_cond.notify_all()

    def empty(self) -> bool:
        return len(self._rc_pool) == 0

    def dump(self) -> dict[K, V]:
        return {key: item.obj for key, item in self._rc_pool.items()}


class RcStreamingPool(Generic[K, V]):
    """
    A streaming pool based on dictionary with reference counting.
    """

    def __init__(self, merge_func: Callable[[V, V], V] | None = None) -> None:
        self._merge_func = merge_func
        self._rc_pool: MutableMapping[K, _RcItemList[V]] = {}
        lock = asyncio.Lock()
        self._get_cond = asyncio.Condition(lock)
        self._put_cond = asyncio.Condition(lock)

    def keys(self) -> Iterable[K]:
        return self._rc_pool.keys()

    async def pop_all(self, key: K) -> V:
        if self._merge_func is None:
            raise ValueError("Merge function is not defined")

        obj_list: list[V] = await self.pop_all_no_merge(key)
        ret = obj_list[0]  # Assume at least one object exist
        for obj in obj_list[1:]:
            ret = self._merge_func(ret, obj)
        return ret

    async def pop_all_no_merge(self, key: K) -> list[V]:
        ticket: int | None = 0
        obj_list: list[V] = []
        while ticket is not None:
            popped = await self.pop(key, ticket)
            if popped is None:
                break
            ticket, obj = popped
            obj_list.append(obj)
        return obj_list

    async def pop(self, key: K, ticket: int) -> tuple[int | None, V] | None:
        """Pops an object associated with the given key from the pool

        It blocks until an object is available. Returns None if no more objects are
        available.

        Parameters
        ----------
        key : K
            The key associated with the object.
        ticket : int
            The ticket number used to track popped objects.

        Returns
        -------
        tuple[int | None, V] | None
            A tuple of the new ticket number and the object if available. Otherwise,
            returns None if no more objects are available.

        Raises
        ------
        ValueError
            If the ticket is invalid.
        IndexError
            If no more objects are available and the ticket is invalid.
        """
        ret = None

        def try_pop() -> bool:
            nonlocal ret
            if key in self._rc_pool:
                item_list = self._rc_pool[key]
                try:
                    ret = item_list.pop(ticket)
                except LateCommitException:
                    # Commit occurred after a new ticket was allocated
                    if item_list.is_empty():
                        self._rc_pool.pop(key)
                    ret = None
                    return True
                if ret is not None:
                    if ret[0] is None and item_list.is_empty():
                        self._rc_pool.pop(key)  # Remove the item list
                    return True
            return False

        async with self._get_cond:
            await self._get_cond.wait_for(try_pop)
            self._put_cond.notify_all()
            return ret

    async def pop_nowait(
        self, key: K, ticket: int, default: V | None = None
    ) -> tuple[int | None, V | None] | None:
        ret_default = None, default

        if key not in self._rc_pool:
            return ret_default

        async with self._put_cond:
            item_list = self._rc_pool[key]
            ret: tuple[int | None, V | None] | None
            try:
                ret = item_list.pop(ticket)
            except LateCommitException:
                # Commit occurred after a new ticket was allocated
                if item_list.is_empty():
                    self._rc_pool.pop(key)
                    self._put_cond.notify_all()
                return ret_default
            if ret is None:
                ret = ret_default
            elif ret[0] is None and item_list.is_empty():
                self._rc_pool.pop(key)  # Remove the item list
            self._put_cond.notify_all()
            return ret

    async def put(
        self, key: K, obj: V, ref_count: int = 1, exist_ok: bool = False
    ) -> None:
        if ref_count == 0:
            # Ignore if the reference count is zero
            return

        if not exist_ok:
            await self.put_nowait(key, obj, ref_count, exist_ok)
            return

        def try_put() -> bool:
            if key not in self._rc_pool:
                self._rc_pool[key] = _RcItemList(ref_count, self._merge_func)
            item_list = self._rc_pool[key]
            if not item_list.is_committed:
                if item_list.ref_count != ref_count:
                    raise ValueError(
                        f"Reference count mismatch: {item_list.ref_count} != {ref_count}"
                    )
                self._rc_pool[key].add(obj)
                return True
            return False

        async with self._put_cond:
            await self._put_cond.wait_for(try_put)
            self._get_cond.notify_all()

    async def put_nowait(
        self, key: K, obj: V, ref_count: int = 1, exist_ok: bool = False
    ) -> None:
        if key not in self._rc_pool:
            self._rc_pool[key] = _RcItemList(ref_count, self._merge_func)

        async with self._get_cond:
            item_list = self._rc_pool[key]
            if item_list.is_committed:
                if not exist_ok:
                    raise ValueError("Key already exists")
                self._rc_pool[key] = _RcItemList(
                    ref_count, self._merge_func
                )  # Overwrite
            elif item_list.ref_count != ref_count:
                raise ValueError(
                    f"Reference count mismatch: {item_list.ref_count} != {ref_count}"
                )
            self._rc_pool[key].add(obj)
            self._get_cond.notify_all()

    async def commit(self, key: K) -> bool:
        if key not in self._rc_pool:
            return False

        async with self._put_cond:
            item_list = self._rc_pool[key]
            item_list.commit()
            if item_list.is_empty():
                self._rc_pool.pop(key)
            self._put_cond.notify_all()
            self._get_cond.notify_all()
            return True

    async def unsubscribe(self, key: K, ticket: int) -> bool:
        if key not in self._rc_pool:
            return False

        async with self._put_cond:
            item_list = self._rc_pool[key]
            try:
                item_list.unsubscribe(ticket)
            except LateCommitException:
                pass
            if item_list.is_empty():
                self._rc_pool.pop(key)
            self._put_cond.notify_all()
            self._get_cond.notify_all()
            return True

    def empty(self) -> bool:
        return len(self._rc_pool) == 0

    def dump(self) -> dict[K, list[V]]:
        return {
            key: [item.obj for item in item_list]
            for key, item_list in self._rc_pool.items()
        }


class AsyncBucket(Generic[K, V]):
    """
    An asynchronous data bucket based on dictionary.
    """

    def __init__(self) -> None:
        self._pool: MutableMapping[K, list[V]] = {}
        lock = asyncio.Lock()
        self._get_cond = asyncio.Condition(lock)

    async def get(self, key: K) -> list[V]:
        async with self._get_cond:
            await self._get_cond.wait_for(lambda: key in self._pool)
            return self._pool[key]

    async def get_nowait(self, key: K) -> list[V] | None:
        async with self._get_cond:
            return self._pool.get(key)

    async def pop(self, key: K) -> list[V]:
        async with self._get_cond:
            await self._get_cond.wait_for(lambda: key in self._pool)
            ret = self._pool.pop(key)
            return ret

    async def pop_nowait(self, key: K) -> list[V] | None:
        async with self._get_cond:
            ret = self._pool.pop(key, None)
            return ret

    async def put(self, key: K, obj: V) -> None:
        async with self._get_cond:
            if key in self._pool:
                self._pool[key].append(obj)
            else:
                self._pool[key] = [obj]
            self._get_cond.notify_all()

    def empty(self) -> bool:
        return len(self._pool) == 0

    def dump(self) -> dict[K, list[V]]:
        return dict(self._pool)

    def dump_and_clear(self) -> dict[K, list[V]]:
        ret = dict(self._pool)
        self._pool.clear()
        return ret
