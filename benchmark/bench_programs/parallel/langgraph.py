from typing import Annotated, Any, TypedDict

from bench_programs.parallel.base import ParallelProgram
from bench_programs.utils.common import try_start_benchmark, try_stop_benchmark
from bench_programs.utils.langgraph import get_langgraph_openai_client
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph

from helium.common import GenerationConfig
from helium.runtime.protocol import HeliumSystemProfile


def update_extracted_insights(
    left: list[dict[int, str]],
    right: tuple[int, int, str] | list[dict[int, str]],
) -> list[dict[int, str]]:
    if isinstance(right, list):
        return right
    agent_idx, review_chunk_idx, insight = right
    left[agent_idx][review_chunk_idx] = insight
    return left


def update_insight_summaries(
    left: dict[int, str],
    right: tuple[int, str] | dict[int, str],
) -> dict[int, str]:
    if isinstance(right, dict):
        return right
    agent_idx, summary = right
    left[agent_idx] = summary
    return left


class AgentContextsState(TypedDict):
    index: int
    item_description: str
    review_chunks: tuple[str, ...]
    extracted_insights: Annotated[list[dict[int, str]], update_extracted_insights]
    insight_summaries: Annotated[dict[int, str], update_insight_summaries]
    report_context: list[BaseMessage]


class LangGraphParallelProgram(ParallelProgram):
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
        **kwargs,
    ) -> tuple[list[str], HeliumSystemProfile]:
        if generation_config is None:
            generation_config = GenerationConfig.from_env()

        base_url = generation_config.base_url

        workflow = self._build_workflow(
            expert_system_prompts,
            len(review_chunks),
            extraction_instruction_fmt,
            summary_instruction_fmt,
            report_instruction_fmt,
            generation_config,
        ).compile()

        # Prepare inputs
        flattened = self.flatten_inputs(item_descriptions, review_chunks)
        inputs = [
            {
                "index": index,
                "item_description": item_description,
                "review_chunks": chunks,
                "extracted_insights": [{} for _ in expert_system_prompts],
                "insight_summaries": {},
                "report_context": [SystemMessage(writer_system_prompt)],
            }
            for index, item_description, chunks in flattened
        ]

        # Start benchmarking
        try_start_benchmark(base_url)

        self.start_timer("generate")
        outputs: list[dict[str, Any]] = await workflow.abatch(inputs)
        self.stop_timer()

        # Stop benchmarking
        system_profile = try_stop_benchmark(base_url)

        output_builder = self.OutputBuilder()
        for output in outputs:
            output_builder.add(output["index"], output["report_context"][-1].content)

        return output_builder.build(), system_profile

    def _build_workflow(
        self,
        expert_system_prompts: list[str],
        num_review_chunks_per_item: int,
        extraction_instruction_fmt: str,
        summary_instruction_fmt: str,
        report_instruction_fmt: str,
        generation_config: GenerationConfig,
    ) -> StateGraph:
        llm = get_langgraph_openai_client(generation_config)

        def expert_insights(expert_idx: int, review_chunk_idx: int):
            async def extract_insights(state: AgentContextsState) -> dict[str, Any]:
                item_description = state["item_description"]
                review_chunk = state["review_chunks"][review_chunk_idx]
                extraction_context = [
                    SystemMessage(expert_system_prompts[expert_idx]),
                    HumanMessage(
                        extraction_instruction_fmt.format(
                            item_description, review_chunk
                        )
                    ),
                ]
                response = await llm.ainvoke(extraction_context)
                return {
                    "extracted_insights": (
                        expert_idx,
                        review_chunk_idx,
                        response.content,
                    )
                }

            return extract_insights

        def expert_summary(expert_idx: int):
            async def summarize_insights(state: AgentContextsState) -> dict[str, Any]:
                item_description = state["item_description"]
                expert_insights = [
                    insights
                    for _, insights in sorted(
                        state["extracted_insights"][expert_idx].items()
                    )
                ]
                summary_context = [
                    SystemMessage(expert_system_prompts[expert_idx]),
                    HumanMessage(
                        summary_instruction_fmt.format(
                            item_description, *expert_insights
                        )
                    ),
                ]
                response = await llm.ainvoke(summary_context)
                return {"insight_summaries": (expert_idx, response.content)}

            return summarize_insights

        async def write_report(state: AgentContextsState) -> dict[str, Any]:
            item_description = state["item_description"]
            insight_summaries = [
                summary for _, summary in sorted(state["insight_summaries"].items())
            ]
            report_context = state["report_context"] + [
                HumanMessage(
                    report_instruction_fmt.format(item_description, *insight_summaries)
                )
            ]
            response = await llm.ainvoke(report_context)
            report_context.append(response)
            return {"report_context": report_context}

        workflow_builder = StateGraph(AgentContextsState)

        summarization_names: list[str] = []
        for expert_idx in range(len(expert_system_prompts)):
            extraction_names: list[str] = []
            # Extract insights for each review chunk
            for review_chunk_idx in range(num_review_chunks_per_item):
                agent_name = f"expert_{expert_idx}_extract_{review_chunk_idx}"
                workflow_builder.add_node(
                    agent_name,
                    expert_insights(expert_idx, review_chunk_idx),
                )
                workflow_builder.add_edge(START, agent_name)
                extraction_names.append(agent_name)
            # Summarize insights for the expert
            summarization_name = f"expert_{expert_idx}_summarize"
            workflow_builder.add_node(
                summarization_name,
                expert_summary(expert_idx),
            )
            for extraction_name in extraction_names:
                workflow_builder.add_edge(extraction_name, summarization_name)
            summarization_names.append(summarization_name)
        # Write the report
        workflow_builder.add_node("writer", write_report)
        for summarization_name in summarization_names:
            workflow_builder.add_edge(summarization_name, "writer")

        return workflow_builder
