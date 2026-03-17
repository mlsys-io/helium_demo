import asyncio
from asyncio import Queue
from collections.abc import Awaitable, Callable
from typing import cast

import networkx as nx

from helium.runtime.utils.logger import log_on_exception_async


class _State:
    def __init__(self, index: int, data: dict) -> None:
        self.index = index
        self.data = data


async def execute_node(
    func: Callable[[dict], Awaitable[dict]],
    in_channel: Queue[_State | None],
    out_channel: Queue[_State | None],
):
    @log_on_exception_async(is_method=False)
    async def handler(state: _State) -> None:
        state.data = await func(state.data)
        await out_channel.put(state)

    handler_tasks = []
    while True:
        state = await in_channel.get()
        if state is None:
            break
        handler_tasks.append(asyncio.create_task(handler(state)))
    await asyncio.gather(*handler_tasks)
    await out_channel.put(None)


class WorkflowDAG:
    def __init__(self):
        self._graph = nx.DiGraph()

    def add_node(self, name: str, func: Callable[[dict], Awaitable[dict]]):
        self._graph.add_node(name, func=func)

    def add_edge(self, from_node: str, to_node: str):
        self._graph.add_edge(from_node, to_node)

    async def execute(self, inputs: list[dict]) -> list[dict]:
        g = self._graph
        if g.number_of_nodes() == 0:
            raise ValueError("The graph is empty. Please add nodes before executing.")
        # Order nodes and prepare channels
        nodes = list(nx.topological_sort(g))
        channels: dict[str, Queue[_State | None]] = {node: Queue() for node in nodes}
        last_channel: Queue[_State | None] = Queue()
        # Start tasks for each node
        tasks = []
        for node, next_node in zip(nodes, nodes[1:] + [None]):
            func = g.nodes[node]["func"]
            in_channel = channels[node]
            out_channel = last_channel if next_node is None else channels[next_node]
            task = asyncio.create_task(execute_node(func, in_channel, out_channel))
            tasks.append(task)
        # Feed inputs to the first node
        first_channel = channels[nodes[0]]
        state: _State | None
        for index, data in enumerate(inputs):
            state = _State(index, data)
            await first_channel.put(state)
        await first_channel.put(None)
        # Fetch results from the last node
        results: list[dict | None] = [None] * len(inputs)
        counter = 0
        while True:
            state = await last_channel.get()
            if state is None:
                break
            results[state.index] = state.data
            counter += 1
        assert counter == len(inputs)
        await asyncio.gather(*tasks)
        return cast(list[dict], results)
