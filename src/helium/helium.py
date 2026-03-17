from collections.abc import Coroutine, Generator
from contextlib import contextmanager
from typing import Any, TypeVar

from helium.graphs import CompiledGraph, Graph
from helium.ops import Op, OutputOp
from helium.runtime import HeliumServer
from helium.runtime.protocol import (
    HeliumQueryProfile,
    HeliumRequestConfig,
    HeliumResponse,
)
from helium.utils import run_coroutine_blocking, unique_id

T = TypeVar("T")


def get_instance(*args, **kwargs) -> HeliumServer:
    return HeliumServer.get_instance(*args, **kwargs)


def get_started_instance(*args, **kwargs) -> HeliumServer:
    return HeliumServer.get_started_instance(*args, **kwargs)


@contextmanager
def serve_instance(*args, **kwargs) -> Generator[HeliumServer, None, None]:
    with HeliumServer.serve_instance(*args, **kwargs) as server:
        yield server


def run_in_server_loop(coro: Coroutine[Any, Any, T]) -> T:
    _ = get_started_instance()
    return run_coroutine_blocking(coro)


def terminate() -> None:
    server = HeliumServer.get_instance()
    if server.is_started:
        server.close()


async def execute_all_async(
    graphs: dict[str, CompiledGraph], config: HeliumRequestConfig | None = None
) -> HeliumResponse:
    server = HeliumServer.get_started_instance()
    return await server.execute(graphs, config)


async def execute_async(
    graph: CompiledGraph, config: HeliumRequestConfig | None = None
) -> HeliumResponse:
    name = unique_id()
    response = await execute_all_async({name: graph}, config)
    response.outputs = response.outputs[name]
    return response


def execute_all(
    graphs: dict[str, CompiledGraph], config: HeliumRequestConfig | None = None
) -> HeliumResponse:
    return run_in_server_loop(execute_all_async(graphs, config))


def execute(
    graph: CompiledGraph, config: HeliumRequestConfig | None = None
) -> HeliumResponse:
    return run_in_server_loop(execute_async(graph, config))


async def profile_all_async(
    graphs: dict[str, CompiledGraph], config: HeliumRequestConfig | None = None
) -> dict[str, HeliumQueryProfile]:
    server = HeliumServer.get_started_instance()
    response = await server.profile(graphs, config)
    query_profile_map = response.query_profile_map
    assert query_profile_map is not None
    return query_profile_map


async def profile_async(
    graph: CompiledGraph, config: HeliumRequestConfig | None = None
) -> HeliumQueryProfile:
    name = unique_id()
    result = await profile_all_async({name: graph}, config)
    return result[name]


def profile_all(
    graphs: dict[str, CompiledGraph], config: HeliumRequestConfig | None = None
) -> dict[str, HeliumQueryProfile]:
    return run_in_server_loop(profile_all_async(graphs, config))


def profile(
    graph: CompiledGraph, config: HeliumRequestConfig | None = None
) -> HeliumQueryProfile:
    return run_in_server_loop(profile_async(graph, config))


Invocable = Op | Graph | CompiledGraph


async def invoke_async(
    inputs: Invocable | list[Invocable] | dict[str, Invocable],
    config: HeliumRequestConfig | None = None,
) -> Any:
    cached_op_names: dict[Op, str] = {}

    def compile(invocable: Invocable) -> CompiledGraph:
        if isinstance(invocable, CompiledGraph):
            compiled = invocable
        elif isinstance(invocable, Graph):
            compiled = invocable.compile()
        else:
            if not isinstance(invocable, OutputOp):
                name = unique_id()
                cached_op_names[invocable] = name
                invocable = OutputOp(name, invocable)
            compiled = Graph.from_ops([invocable]).compile()
        return compiled

    def unwrap(name: str, inp: Invocable, response: HeliumResponse) -> Any:
        out = response.outputs[name]
        if isinstance(inp, CompiledGraph) or isinstance(inp, Graph):
            return out
        if isinstance(inp, OutputOp):
            return out[inp.name]
        return out[cached_op_names[inp]]

    graphs: dict[str, CompiledGraph] = {}
    if isinstance(inputs, list):
        names = [unique_id() for _ in range(len(inputs))]
        graphs = {name: compile(inp) for name, inp in zip(names, inputs)}
        response = await execute_all_async(graphs, config)
        return [unwrap(name, inp, response) for name, inp in zip(names, inputs)]

    if isinstance(inputs, dict):
        graphs = {name: compile(inp) for name, inp in inputs.items()}
        response = await execute_all_async(graphs, config)
        return {name: unwrap(name, inp, response) for name, inp in inputs.items()}

    name = unique_id()
    graphs = {name: compile(inputs)}
    result = await execute_all_async(graphs, config)
    return unwrap(name, inputs, result)


def invoke(
    inputs: Invocable | list[Invocable] | dict[str, Invocable],
    config: HeliumRequestConfig | None = None,
) -> Any:
    return run_in_server_loop(invoke_async(inputs, config))
