from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Annotated, Any

from bench_programs.parallel.base import ParallelProgram
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


def update_extracted_insights(
    left: list[dict[int, str]],
    right: tuple[int, int, str] | list[dict[int, str]],
) -> list[dict[int, str]]:
    if isinstance(right, list):
        return right
    expert_idx, review_chunk_idx, insight = right
    left[expert_idx][review_chunk_idx] = insight
    return left


def update_insight_summaries(
    left: dict[int, str],
    right: tuple[int, str] | dict[int, str],
) -> dict[int, str]:
    if isinstance(right, dict):
        return right
    expert_idx, summary = right
    left[expert_idx] = summary
    return left


class AgentContextsState(TypedDict):
    index: int
    item_description: str
    review_chunks: tuple[str, ...]
    extracted_insights: Annotated[list[dict[int, str]], update_extracted_insights]
    insight_summaries: Annotated[dict[int, str], update_insight_summaries]
    report_context: list[BaseMessage]


@dataclass
class AgentIds:
    extract_ids_by_expert: list[list[str]]
    summary_agent_ids: list[str]
    writer_agent_id: str

    def to_tuple(self) -> tuple[list[list[str]], list[str], str]:
        return self.extract_ids_by_expert, self.summary_agent_ids, self.writer_agent_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class KVFlowParallelProgram(ParallelProgram):
    async def _run(
        self,
        expert_system_prompts: list[str],
        writer_system_prompt: str,
        extraction_instruction_fmt: str,
        summary_instruction_fmt: str,
        report_instruction_fmt: str,
        item_descriptions: list[str],
        review_chunks: tuple[list[str], ...],
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

        flattened = self.flatten_inputs(item_descriptions, review_chunks)
        num_experts = len(expert_system_prompts)
        num_review_chunks_per_item = len(review_chunks)

        step_graph_updater, agent_ids = await self._prepare_agent_step_graph(
            num_experts=num_experts,
            num_review_chunks_per_item=num_review_chunks_per_item,
            update_agent_step_graph=update_agent_step_graph,
        )

        # Precompute static prompts
        self.start_timer("precompute")
        assert get_worker_generation_configs is not None
        await self._precompute_static_prompts(
            agent_ids=agent_ids,
            expert_system_prompts=expert_system_prompts,
            writer_system_prompt=writer_system_prompt,
            generation_configs=get_worker_generation_configs(generation_config),
        )
        self.stop_timer()

        workflow = (
            await self._build_workflow(
                agent_ids=agent_ids,
                updater=step_graph_updater,
                expert_system_prompts=expert_system_prompts,
                extraction_instruction_fmt=extraction_instruction_fmt,
                summary_instruction_fmt=summary_instruction_fmt,
                report_instruction_fmt=report_instruction_fmt,
                generation_config=generation_config,
            )
        ).compile()

        inputs: list[AgentContextsState] = [
            {
                "index": index,
                "item_description": item_description,
                "review_chunks": chunks,
                "extracted_insights": [{} for _ in range(num_experts)],
                "insight_summaries": {},
                "report_context": [SystemMessage(writer_system_prompt)],
            }
            for index, item_description, chunks in flattened
        ]

        # Start benchmarking
        await step_graph_updater.reset(total_items=len(inputs))
        await start_benchmark()

        self.start_timer("generate")
        outputs: list[dict[str, Any]] = await workflow.abatch(inputs)  # type: ignore[arg-type]
        self.stop_timer()

        system_profile = await stop_benchmark()

        output_builder = self.OutputBuilder()
        for output in outputs:
            output_builder.add(output["index"], output["report_context"][-1].content)

        return output_builder.build(), system_profile

    async def _build_workflow(
        self,
        agent_ids: AgentIds,
        updater: KVFlowStepGraphUpdater,
        expert_system_prompts: list[str],
        extraction_instruction_fmt: str,
        summary_instruction_fmt: str,
        report_instruction_fmt: str,
        generation_config: GenerationConfig,
    ) -> StateGraph:
        extract_ids_by_expert, summary_agent_ids, writer_agent_id = agent_ids.to_tuple()
        llm = get_langgraph_openai_client(generation_config)
        num_review_chunks_per_item = (
            len(extract_ids_by_expert[0]) if extract_ids_by_expert else 0
        )

        def expert_insights(agent_id: str, expert_idx: int, review_chunk_idx: int):
            async def extract_insights(state: AgentContextsState) -> dict[str, Any]:
                async with updater.node(agent_id):
                    item_description = state["item_description"]
                    review_chunk = state["review_chunks"][review_chunk_idx]
                    msgs = [
                        SystemMessage(expert_system_prompts[expert_idx]),
                        HumanMessage(
                            extraction_instruction_fmt.format(
                                item_description, review_chunk
                            )
                        ),
                    ]
                    resp = await llm_ainvoke(llm, msgs, agent_id)
                    content = resp.content
                    return {
                        "extracted_insights": (
                            expert_idx,
                            review_chunk_idx,
                            content if isinstance(content, str) else str(content),
                        )
                    }

            return extract_insights

        def expert_summary(agent_id: str, expert_idx: int):
            async def summarize_insights(state: AgentContextsState) -> dict[str, Any]:
                async with updater.node(agent_id):
                    item_description = state["item_description"]
                    expert_insights_for_item = [
                        insight
                        for _, insight in sorted(
                            state["extracted_insights"][expert_idx].items()
                        )
                    ]
                    msgs = [
                        SystemMessage(expert_system_prompts[expert_idx]),
                        HumanMessage(
                            summary_instruction_fmt.format(
                                item_description, *expert_insights_for_item
                            )
                        ),
                    ]
                    resp = await llm_ainvoke(llm, msgs, agent_id)
                    content = resp.content
                    return {
                        "insight_summaries": (
                            expert_idx,
                            content if isinstance(content, str) else str(content),
                        )
                    }

            return summarize_insights

        async def write_report(state: AgentContextsState) -> dict[str, Any]:
            async with updater.node(writer_agent_id):
                item_description = state["item_description"]
                insight_summaries = [
                    summary for _, summary in sorted(state["insight_summaries"].items())
                ]
                report_context = state["report_context"] + [
                    HumanMessage(
                        report_instruction_fmt.format(
                            item_description, *insight_summaries
                        )
                    )
                ]
                resp = await llm_ainvoke(llm, report_context, writer_agent_id)
                report_context.append(resp)
                return {"report_context": report_context}

        workflow_builder = StateGraph(AgentContextsState)

        summarization_names: list[str] = []
        for expert_idx in range(len(expert_system_prompts)):
            extraction_names: list[str] = []
            for review_chunk_idx in range(num_review_chunks_per_item):
                agent_name = extract_ids_by_expert[expert_idx][review_chunk_idx]
                workflow_builder.add_node(
                    agent_name,
                    expert_insights(agent_name, expert_idx, review_chunk_idx),
                )
                workflow_builder.add_edge(START, agent_name)
                extraction_names.append(agent_name)

            summarization_name = summary_agent_ids[expert_idx]
            workflow_builder.add_node(
                summarization_name,
                expert_summary(summarization_name, expert_idx),
            )
            for extraction_name in extraction_names:
                workflow_builder.add_edge(extraction_name, summarization_name)
            summarization_names.append(summarization_name)

        workflow_builder.add_node(writer_agent_id, write_report)
        for summarization_name in summarization_names:
            workflow_builder.add_edge(summarization_name, writer_agent_id)

        return workflow_builder

    async def _precompute_static_prompts(
        self,
        agent_ids: AgentIds,
        expert_system_prompts: list[str],
        writer_system_prompt: str,
        generation_configs: list[GenerationConfig],
    ) -> None:
        static_prompts = self._build_static_prompts(
            agent_ids=agent_ids,
            expert_system_prompts=expert_system_prompts,
            writer_system_prompt=writer_system_prompt,
        )
        await precompute_static_prompts(static_prompts, generation_configs)

    def _build_agent_ids(
        self, num_experts: int, num_review_chunks_per_item: int
    ) -> AgentIds:
        extract_ids_by_expert = [
            [
                f"expert_{expert_idx}_extract_{review_chunk_idx}"
                for review_chunk_idx in range(num_review_chunks_per_item)
            ]
            for expert_idx in range(num_experts)
        ]
        summary_agent_ids = [
            f"expert_{expert_idx}_summarize" for expert_idx in range(num_experts)
        ]
        writer_agent_id = "writer"
        return AgentIds(extract_ids_by_expert, summary_agent_ids, writer_agent_id)

    async def _prepare_agent_step_graph(
        self,
        num_experts: int,
        num_review_chunks_per_item: int,
        update_agent_step_graph: Callable[
            [dict[str, Any], dict[int, list[str]], int], Awaitable[None]
        ],
    ) -> tuple[KVFlowStepGraphUpdater, AgentIds]:
        agent_ids = self._build_agent_ids(
            num_experts=num_experts,
            num_review_chunks_per_item=num_review_chunks_per_item,
        )
        step_graph = self._build_agent_step_graph(**agent_ids.to_dict())
        updater: KVFlowStepGraphUpdater = KVFlowStepGraphUpdater(
            step_graph=step_graph,
            send_update=update_agent_step_graph,
        )
        return updater, agent_ids

    def _build_agent_step_graph(
        self,
        extract_ids_by_expert: list[list[str]],
        summary_agent_ids: list[str],
        writer_agent_id: str,
    ) -> KVFlowStepGraph:
        extract_agent_ids = [
            agent_id for per_expert in extract_ids_by_expert for agent_id in per_expert
        ]

        edges: dict[str, set[str]] = {
            agent_id: set() for agent_id in [*extract_agent_ids, *summary_agent_ids]
        }
        for expert_idx, extract_ids in enumerate(extract_ids_by_expert):
            summary_id = summary_agent_ids[expert_idx]
            for extract_id in extract_ids:
                edges.setdefault(extract_id, set()).add(summary_id)
            edges.setdefault(summary_id, set()).add(writer_agent_id)

        return KVFlowStepGraph(
            edges=edges,
            all_agents={*extract_agent_ids, *summary_agent_ids, writer_agent_id},
        )

    def _build_static_prompts(
        self,
        agent_ids: AgentIds,
        expert_system_prompts: list[str],
        writer_system_prompt: str,
    ) -> dict[str, list[BaseMessage]]:
        extract_ids_by_expert, summary_agent_ids, writer_agent_id = agent_ids.to_tuple()
        static_prompts: dict[str, list[BaseMessage]] = {}
        for expert_idx, extract_ids in enumerate(extract_ids_by_expert):
            expert_prompt: list[BaseMessage] = [
                SystemMessage(expert_system_prompts[expert_idx])
            ]
            for agent_id in extract_ids:
                static_prompts[agent_id] = expert_prompt
            static_prompts[summary_agent_ids[expert_idx]] = expert_prompt
        static_prompts[writer_agent_id] = [SystemMessage(writer_system_prompt)]
        return static_prompts
