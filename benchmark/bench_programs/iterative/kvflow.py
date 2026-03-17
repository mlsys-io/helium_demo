from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Any

from bench_programs.iterative.base import IterativeProgram
from bench_programs.utils.kvflow import (
    KVFlowStepGraph,
    KVFlowStepGraphUpdater,
    llm_ainvoke,
    precompute_static_prompts,
)
from bench_programs.utils.langgraph import get_langgraph_openai_client
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

from helium.common import GenerationConfig
from helium.runtime.protocol import HeliumSystemProfile


class State(TypedDict):
    index: int
    chunks: tuple[str, ...]
    summary: list[str]


@dataclass
class AgentIds:
    chunk_agent_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class KVFlowIterativeProgram(IterativeProgram):
    async def _run(
        self,
        system_prompt: str,
        first_prompt_fmt: str,
        subsequent_prompt_fmt: str,
        document_chunks: tuple[list[str], ...],
        generation_config: GenerationConfig | None,
        start_benchmark: Callable[[], Awaitable[None]] | None = None,
        stop_benchmark: Callable[[], Awaitable[HeliumSystemProfile]] | None = None,
        update_agent_step_graph: (
            Callable[[dict[str, Any], dict[int, list[str]], int], Awaitable[None]]
            | None
        ) = None,
        get_worker_generation_configs: (
            Callable[[GenerationConfig], list[GenerationConfig]] | None
        ) = None,
        **kwargs,
    ) -> tuple[list[str], HeliumSystemProfile]:
        assert start_benchmark is not None
        assert stop_benchmark is not None
        assert update_agent_step_graph is not None

        if generation_config is None:
            generation_config = GenerationConfig.from_env()

        # Prepare inputs
        flattened = self.flatten_inputs(document_chunks)
        num_chunks = len(document_chunks)

        step_graph_updater, agent_ids = await self._prepare_agent_step_graph(
            num_chunks=num_chunks,
            update_agent_step_graph=update_agent_step_graph,
        )

        # Precompute static prompts
        self.start_timer("precompute")
        assert get_worker_generation_configs is not None
        await self._precompute_static_prompts(
            agent_ids=agent_ids,
            system_prompt=system_prompt,
            generation_configs=get_worker_generation_configs(generation_config),
        )
        self.stop_timer()

        workflow = (
            await self._build_workflow(
                agent_ids=agent_ids,
                updater=step_graph_updater,
                system_prompt=system_prompt,
                first_prompt_fmt=first_prompt_fmt,
                subsequent_prompt_fmt=subsequent_prompt_fmt,
                num_chunks=num_chunks,
                generation_config=generation_config,
            )
        ).compile()

        inputs: list[State] = [
            {"index": index, "chunks": chunks, "summary": []}
            for index, chunks in flattened
        ]

        await start_benchmark()

        self.start_timer("generate")
        await step_graph_updater.reset(total_items=len(inputs))
        outputs: list[dict[str, Any]] = await workflow.abatch(inputs)  # type: ignore[arg-type]
        self.stop_timer()

        system_profile = await stop_benchmark()

        output_builder = self.OutputBuilder()
        for output in outputs:
            output_builder.add(output["index"], output["summary"][-1])

        return output_builder.build(), system_profile

    async def _build_workflow(
        self,
        agent_ids: AgentIds,
        updater: KVFlowStepGraphUpdater,
        system_prompt: str,
        first_prompt_fmt: str,
        subsequent_prompt_fmt: str,
        num_chunks: int,
        generation_config: GenerationConfig,
    ) -> StateGraph:
        llm = get_langgraph_openai_client(generation_config)

        def summary_node(chunk_idx: int, agent_id: str):
            async def summarize(state: State) -> dict[str, Any]:
                async with updater.node(agent_id):
                    chunk = state["chunks"][chunk_idx]
                    if chunk_idx == 0:
                        prompt = first_prompt_fmt.format(chunk)
                    else:
                        prev_summary = state["summary"][-1]
                        prompt = subsequent_prompt_fmt.format(prev_summary, chunk)

                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=prompt),
                    ]
                    response = await llm_ainvoke(llm, messages, agent_id)
                    content = response.content
                    summary = state["summary"] + [
                        content if isinstance(content, str) else str(content)
                    ]
                    return {"summary": summary}

            return summarize

        workflow_builder = StateGraph(State)
        prev_node = START
        for chunk_idx in range(num_chunks):
            node_name = agent_ids.chunk_agent_ids[chunk_idx]
            workflow_builder.add_node(node_name, summary_node(chunk_idx, node_name))
            workflow_builder.add_edge(prev_node, node_name)
            prev_node = node_name

        return workflow_builder

    def _build_agent_ids(self, num_chunks: int) -> AgentIds:
        return AgentIds(chunk_agent_ids=[f"chunk_{i}" for i in range(num_chunks)])

    async def _prepare_agent_step_graph(
        self,
        num_chunks: int,
        update_agent_step_graph: Callable[
            [dict[str, Any], dict[int, list[str]], int], Awaitable[None]
        ],
    ) -> tuple[KVFlowStepGraphUpdater, AgentIds]:
        agent_ids = self._build_agent_ids(num_chunks=num_chunks)
        step_graph = self._build_agent_step_graph(**agent_ids.to_dict())
        updater: KVFlowStepGraphUpdater = KVFlowStepGraphUpdater(
            step_graph=step_graph,
            send_update=update_agent_step_graph,
        )
        return updater, agent_ids

    def _build_agent_step_graph(self, chunk_agent_ids: list[str]) -> KVFlowStepGraph:
        edges: dict[str, set[str]] = {a: set() for a in chunk_agent_ids}
        for i in range(len(chunk_agent_ids) - 1):
            edges[chunk_agent_ids[i]].add(chunk_agent_ids[i + 1])
        return KVFlowStepGraph(edges=edges, all_agents=set(chunk_agent_ids))

    def _build_static_prompts(
        self,
        agent_ids: AgentIds,
        system_prompt: str,
    ) -> dict[str, list[BaseMessage]]:
        static: list[BaseMessage] = [SystemMessage(content=system_prompt)]
        return {agent_id: static for agent_id in agent_ids.chunk_agent_ids}

    async def _precompute_static_prompts(
        self,
        agent_ids: AgentIds,
        system_prompt: str,
        generation_configs: list[GenerationConfig],
    ) -> None:
        static_prompts = self._build_static_prompts(
            agent_ids=agent_ids,
            system_prompt=system_prompt,
        )
        await precompute_static_prompts(static_prompts, generation_configs)
