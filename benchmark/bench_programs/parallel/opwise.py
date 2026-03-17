from typing import Any

from bench_programs.parallel.base import ParallelProgram
from bench_programs.utils.common import try_start_benchmark, try_stop_benchmark
from bench_programs.utils.openai import openai_generate_async, prepare_openai
from bench_programs.utils.opwise import WorkflowDAG
from openai import AsyncOpenAI

from helium.common import GenerationConfig, Message
from helium.runtime.protocol import HeliumSystemProfile


class OpWiseParallelProgram(ParallelProgram):
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
        client, generation_config = prepare_openai(generation_config)
        generation_kwargs = generation_config.openai_kwargs()
        base_url = generation_config.base_url

        # Prepare inputs
        flattened = self.flatten_inputs(item_descriptions, review_chunks)
        inputs = [
            {
                "index": index,
                "item_description": item_description,
                "review_chunks": chunks,
                "extracted_insights": [{} for _ in expert_system_prompts],
                "insight_summaries": {},
            }
            for index, item_description, chunks in flattened
        ]

        workflow = self._build_workflow(
            client,
            expert_system_prompts,
            len(review_chunks),
            writer_system_prompt,
            extraction_instruction_fmt,
            summary_instruction_fmt,
            report_instruction_fmt,
            generation_kwargs,
        )

        # Start benchmarking
        try_start_benchmark(base_url)

        self.start_timer("generate")
        states = await workflow.execute(inputs)
        self.stop_timer()

        # Stop benchmarking
        system_profile = try_stop_benchmark(base_url)

        output_builder = self.OutputBuilder()
        for state in states:
            output_builder.add(state["index"], state["report"])

        return output_builder.build(), system_profile

    def _build_workflow(
        self,
        client: AsyncOpenAI,
        expert_system_prompts: list[str],
        num_review_chunks_per_item: int,
        writer_system_prompt: str,
        extraction_instruction_fmt: str,
        summary_instruction_fmt: str,
        report_instruction_fmt: str,
        generation_kwargs: dict[str, Any],
    ) -> WorkflowDAG:
        def expert_insights(expert_idx: int, review_chunk_idx: int):
            async def extract_insights(state: dict) -> dict:
                item_description = state["item_description"]
                review_chunk = state["review_chunks"][review_chunk_idx]
                messages = [
                    Message("system", expert_system_prompts[expert_idx]),
                    Message(
                        "user",
                        extraction_instruction_fmt.format(
                            item_description, review_chunk
                        ),
                    ),
                ]
                extracted_insight = await openai_generate_async(
                    client, messages, generation_kwargs
                )
                state["extracted_insights"][expert_idx][
                    review_chunk_idx
                ] = extracted_insight
                return state

            return extract_insights

        def expert_summary(expert_idx: int):
            async def summarize_insights(state: dict) -> dict:
                item_description = state["item_description"]
                expert_insights = [
                    insights
                    for _, insights in sorted(
                        state["extracted_insights"][expert_idx].items()
                    )
                ]
                messages = [
                    Message("system", expert_system_prompts[expert_idx]),
                    Message(
                        "user",
                        summary_instruction_fmt.format(
                            item_description, *expert_insights
                        ),
                    ),
                ]
                summary = await openai_generate_async(
                    client, messages, generation_kwargs
                )
                state["insight_summaries"][expert_idx] = summary
                return state

            return summarize_insights

        async def write_report(state: dict) -> dict:
            item_description = state["item_description"]
            insight_summaries = [
                summary for _, summary in sorted(state["insight_summaries"].items())
            ]
            messages = [
                Message("system", writer_system_prompt),
                Message(
                    "user",
                    report_instruction_fmt.format(item_description, *insight_summaries),
                ),
            ]
            report = await openai_generate_async(client, messages, generation_kwargs)
            state["report"] = report
            return state

        workflow = WorkflowDAG()

        summarization_names: list[str] = []
        for expert_idx in range(len(expert_system_prompts)):
            extraction_names: list[str] = []
            # Extract insights for each review chunk
            for review_chunk_idx in range(num_review_chunks_per_item):
                agent_name = f"expert_{expert_idx}_extract_{review_chunk_idx}"
                workflow.add_node(
                    agent_name, expert_insights(expert_idx, review_chunk_idx)
                )
                extraction_names.append(agent_name)
            # Summarize insights for the expert
            summarization_name = f"expert_{expert_idx}_summarize"
            workflow.add_node(summarization_name, expert_summary(expert_idx))
            for extraction_name in extraction_names:
                workflow.add_edge(extraction_name, summarization_name)
            summarization_names.append(summarization_name)
        # Write the report
        workflow.add_node("writer", write_report)
        for summarization_name in summarization_names:
            workflow.add_edge(summarization_name, "writer")

        return workflow
