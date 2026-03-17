import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from bench_programs.utils.langgraph import get_langgraph_openai_client
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from helium.common import GenerationConfig
from helium.utils.graph import topological_sort

_FAR_FUTURE: int = 1_000_000


async def llm_ainvoke(
    llm: ChatOpenAI, input: LanguageModelInput, agent_id: str, **kwargs
) -> BaseMessage:
    response = await llm.ainvoke(
        input, extra_headers={"X-SGLANG-AGENT-ID": agent_id}, **kwargs
    )
    return response


async def llm_abatch(
    llm: ChatOpenAI, inputs: list[LanguageModelInput], agent_id: str, **kwargs
) -> Sequence[BaseMessage]:
    responses = await llm.abatch(
        inputs, extra_headers={"X-SGLANG-AGENT-ID": agent_id}, **kwargs
    )
    return responses


async def precompute_static_prompts(
    static_prompts: dict[str, list[BaseMessage]],
    generation_configs: list[GenerationConfig],
) -> None:
    coros: list[Any] = []
    repeat = 3
    for config in generation_configs:
        llm = get_langgraph_openai_client(config)
        for _ in range(repeat):
            for agent_id, static_prompt in static_prompts.items():
                coros.append(
                    llm_ainvoke(
                        llm,
                        [*static_prompt, HumanMessage("warmup")],
                        agent_id,
                        max_tokens=1,
                    )
                )
    await asyncio.gather(*coros)


@dataclass(frozen=True, slots=True)
class KVFlowUpdatePayload:
    agent_data: dict[str, Any]
    timestep_data: dict[int, list[str]]
    timestep_cnt: int


class KVFlowStepGraph:
    """
    A lightweight "agent step graph" for computing steps-to-execution.

    We treat agent nodes as strings (agent_id) and edges as control-flow.
    Steps-to-execution is computed under an ALL (fan-in barrier) semantics:
      - active agents have step 0
      - done agents are treated as satisfied dependencies (step 0) but are
        assigned far_future in the final output so they are evictable
      - for other agents, step(v) = max(step(u) for u in preds(v)) + 1
    """

    def __init__(
        self,
        edges: dict[str, set[str]],
        all_agents: set[str],
        far_future: int = _FAR_FUTURE,
    ) -> None:
        self._all_agents = frozenset(all_agents)
        self._far_future = far_future

        nodes = self._all_agents
        self._graph: dict[str, set[str]] = {
            u: {v for v in edges.get(u, set()) if v in nodes} for u in nodes
        }
        self._preds: dict[str, set[str]] = {a: set() for a in nodes}
        for u, vs in self._graph.items():
            for v in vs:
                self._preds[v].add(u)

        self._topo = topological_sort(self._graph, secondary_key=str)

    def steps_to_execution(
        self,
        active_agents: set[str],
        done_agents: set[str] | None = None,
    ) -> dict[str, int]:
        far_future = self._far_future

        nodes = self._all_agents
        active_set = active_agents & nodes
        done_set = (done_agents or set()) & nodes
        done_set -= active_set

        dep_steps: dict[str, int] = {a: far_future for a in self._all_agents}
        for a in done_set | active_set:
            dep_steps[a] = 0

        for a in self._topo:
            if a in active_set or a in done_set:
                continue
            if not self._preds[a]:
                continue

            pred_steps = [dep_steps[p] for p in self._preds[a]]
            if any(s >= far_future for s in pred_steps):
                continue

            step = max(pred_steps) + 1
            dep_steps[a] = step

        steps = dict(dep_steps)
        for a in done_set:
            steps[a] = far_future
        return steps

    def build_timestep_data(self, distances: dict[str, int]) -> dict[int, list[str]]:
        far_future = self._far_future
        buckets: dict[int, list[str]] = {}
        for agent_id, dist in distances.items():
            if dist >= far_future:
                continue
            buckets.setdefault(dist, []).append(agent_id)

        for k in buckets:
            buckets[k].sort()
        return buckets

    def build_payload(
        self, active_agents: set[str], done_agents: set[str] | None = None
    ) -> KVFlowUpdatePayload:
        distances = self.steps_to_execution(
            active_agents=active_agents, done_agents=done_agents
        )
        timestep_data = self.build_timestep_data(distances=distances)
        timestep_cnt = (max(timestep_data.keys()) + 1) if timestep_data else 0
        return KVFlowUpdatePayload(distances, timestep_data, timestep_cnt)


class KVFlowStepGraphUpdater:
    """
    Convenience wrapper that de-duplicates and serializes `/v1/update` calls.

    Design goal
    -----------
    KVFlow programs are executed via `workflow.abatch`, so each workflow node is
    invoked once per input item. Updating the agent step graph for every item
    (per-node, per-item) is too expensive and can distort benchmark latency.

    Instead, we only update when:
      1) a node starts executing for the first time (i.e., the first input
         reaches the node), and
      2) a node finishes executing for the entire batch (i.e., all inputs have
         completed that node).
    """

    def __init__(
        self,
        step_graph: KVFlowStepGraph,
        send_update: Callable[
            [dict[str, Any], dict[int, list[str]], int], Awaitable[None]
        ],
    ) -> None:
        self._step_graph = step_graph
        self._send_update = send_update

        self._lock = asyncio.Lock()
        self._last_payload: KVFlowUpdatePayload | None = None
        self._total_items: int | None = None

        self._active_agents: set[str] = set()
        self._done_agents: set[str] = set()
        self._started_agents: set[str] = set()
        self._completed_counts: dict[str, int] = {}

    async def _send_if_changed(self) -> None:
        payload = self._step_graph.build_payload(
            active_agents=self._active_agents, done_agents=self._done_agents
        )
        if payload == self._last_payload:
            return
        self._last_payload = payload
        await self._send_update(
            payload.agent_data,
            payload.timestep_data,
            payload.timestep_cnt,
        )

    async def reset(self, *, total_items: int) -> None:
        async with self._lock:
            if total_items <= 0:
                raise ValueError("total_items must be > 0")
            self._total_items = total_items
            self._active_agents = set()
            self._done_agents = set()
            self._started_agents = set()
            self._completed_counts = {}
            self._last_payload = None

    async def node_started(self, agent_id: str) -> None:
        if agent_id not in self._step_graph._all_agents:
            return
        async with self._lock:
            if agent_id in self._started_agents:
                return
            self._started_agents.add(agent_id)
            self._active_agents.add(agent_id)
            await self._send_if_changed()

    async def node_finished(self, agent_id: str) -> None:
        if agent_id not in self._step_graph._all_agents:
            return
        async with self._lock:
            if self._total_items is None:
                raise RuntimeError(
                    "KVFlowStepGraphUpdater.reset(total_items=...) must be called "
                    "before executing the workflow."
                )

            self._completed_counts[agent_id] = (
                self._completed_counts.get(agent_id, 0) + 1
            )
            if self._completed_counts[agent_id] < self._total_items:
                return

            if agent_id in self._done_agents:
                return
            self._active_agents.discard(agent_id)
            self._done_agents.add(agent_id)
            await self._send_if_changed()

    @asynccontextmanager
    async def node(self, agent_id: str) -> AsyncIterator[None]:
        await self.node_started(agent_id)
        try:
            yield
        finally:
            await self.node_finished(agent_id)
