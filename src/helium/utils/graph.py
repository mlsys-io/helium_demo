from collections.abc import Callable, Iterable
from typing import Any, TypeVar

T = TypeVar("T")


def topological_sort(
    graph: dict[T, set[T]], secondary_key: Callable[[T], Any] | None = None
) -> list[T]:
    in_degree = {node: 0 for node in graph}
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            in_degree[neighbor] = in_degree.get(neighbor, 0) + 1

    ready_nodes = [node for node, deg in in_degree.items() if deg == 0]
    result = []
    while ready_nodes:
        if secondary_key is not None:
            ready_nodes.sort(key=secondary_key)
        current_layer = ready_nodes
        ready_nodes = []

        result.extend(current_layer)

        # Decrement in-degree of neighbors and collect next layer
        for node in current_layer:
            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    ready_nodes.append(neighbor)

    if len(result) != len(in_degree):
        raise ValueError("Graph has at least one cycle.")

    return result


def reversed_topological_sort(
    graph: dict[T, Iterable[T]], secondary_key: Callable[[T], Any] | None = None
) -> list[T]:
    # Reverse the graph
    reversed_graph: dict[T, set[T]] = {node: set() for node in graph}
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            reversed_graph[neighbor].add(node)

    return topological_sort(reversed_graph, secondary_key)


def detect_sccs(graph: dict[T, set[T]]) -> list[list[T]]:
    """Detects SCCs in a directed graph using Tarjan's algorithm"""
    index: int = 0
    stack: list[T] = []
    indices: dict[T, int] = {}
    lowlink: dict[T, int] = {}
    on_stack = set()
    sccs = []

    def dfs(node: T) -> None:
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        neighbors = graph.get(node)
        if neighbors is not None:
            for neighbor in neighbors:
                if neighbor not in indices:
                    dfs(neighbor)
                    lowlink[node] = min(lowlink[node], lowlink[neighbor])
                elif neighbor in on_stack:
                    lowlink[node] = min(lowlink[node], indices[neighbor])

        if lowlink[node] == indices[node]:
            scc = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                scc.append(w)
                if w == node:
                    break
            sccs.append(scc)

    for node in graph:
        if node not in indices:
            dfs(node)

    return sccs


def detect_wccs(graph: dict[T, set[T]]) -> list[list[T]]:
    """Detect weakly connected components (WCCs) in a directed graph"""
    # Convert directed graph to undirected
    undirected_graph: dict[T, set[T]] = {}
    for node, neighbors in graph.items():
        if node not in undirected_graph:
            undirected_graph[node] = set()
        for neighbor in neighbors:
            if neighbor not in undirected_graph:
                undirected_graph[neighbor] = set()
            undirected_graph[node].add(neighbor)
            undirected_graph[neighbor].add(node)

    # Perform DFS to find WCCs
    visited = set()
    wccs = []
    stack = []
    for node in undirected_graph:
        if node not in visited:
            wcc = []
            stack.append(node)
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    wcc.append(current)
                    for neighbor in undirected_graph[current]:
                        if neighbor not in visited:
                            stack.append(neighbor)
            wccs.append(wcc)
    return wccs


def calculate_node_depths(graph: dict[T, set[T]]) -> dict[T, int]:
    """Calculates the depth of each node in a directed acyclic graph (DAG)"""
    depth = {node: 0 for node in graph}
    for node in topological_sort(graph):
        for neighbor in graph.get(node, []):
            depth[neighbor] = max(depth[neighbor], depth[node] + 1)
    return depth
