import asyncio
from typing import Any

from bench_programs.parallel.base import ParallelProgram
from bench_programs.utils.common import try_start_benchmark, try_stop_benchmark
from bench_programs.utils.openai import (
    BATCH_SIZE,
    openai_generate_async,
    openai_iter_batch,
    prepare_openai,
)
from openai import AsyncOpenAI

from helium.common import GenerationConfig, Message
from helium.runtime.protocol import HeliumSystemProfile


class QueryWiseParallelProgram(ParallelProgram):
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

        # Start benchmarking
        try_start_benchmark(base_url)

        self.start_timer("generate")
        output_builder = self.OutputBuilder()
        for batch in openai_iter_batch(flattened, BATCH_SIZE):
            batch_outputs = await asyncio.gather(
                *[
                    self._run_item(
                        index,
                        client,
                        expert_system_prompts,
                        writer_system_prompt,
                        extraction_instruction_fmt,
                        summary_instruction_fmt,
                        report_instruction_fmt,
                        item_description,
                        chunks,
                        generation_kwargs,
                    )
                    for index, item_description, chunks in batch
                ]
            )
            output_builder.update(batch_outputs)
        self.stop_timer()

        # Stop benchmarking
        system_profile = try_stop_benchmark(base_url)

        return output_builder.build(), system_profile

    async def _run_item(
        self,
        index: int,
        client: AsyncOpenAI,
        expert_system_prompts: list[str],
        writer_system_prompt: str,
        extraction_instruction_fmt: str,
        summary_instruction_fmt: str,
        report_instruction_fmt: str,
        item_description: str,
        review_chunks: tuple[str, ...],
        generation_kwargs: dict[str, Any],
    ) -> tuple[int, str]:
        insight_summaries: list[str] = []
        for expert_prompt in expert_system_prompts:
            system_message = Message("system", expert_prompt)
            # Extract insights from each review chunk
            extracted_insights: list[str] = []
            for review_chunk in review_chunks:
                messages = [
                    system_message,
                    Message(
                        "user",
                        extraction_instruction_fmt.format(
                            item_description, review_chunk
                        ),
                    ),
                ]
                insight = await openai_generate_async(
                    client, messages, generation_kwargs
                )
                extracted_insights.append(insight)
            # Summarize extracted insights
            messages = [
                system_message,
                Message(
                    "user",
                    summary_instruction_fmt.format(
                        item_description, *extracted_insights
                    ),
                ),
            ]
            insight_summary = await openai_generate_async(
                client, messages, generation_kwargs
            )
            insight_summaries.append(insight_summary)
        # Write the final report
        messages = [
            Message("system", writer_system_prompt),
            Message(
                "user",
                report_instruction_fmt.format(item_description, *insight_summaries),
            ),
        ]
        insight_report = await openai_generate_async(
            client, messages, generation_kwargs
        )

        return index, insight_report
