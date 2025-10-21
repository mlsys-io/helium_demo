from collections.abc import Sequence
from typing import Literal

from bench_programs.parallel.base import OutputType, ParallelProgram
from bench_programs.utils.common import random_shuffle

from helium import ops
from helium.common import GenerationConfig
from helium.frontend.agents import Agent
from helium.frontend.programs import Program as HeliumProgram
from helium.runtime import HeliumServerConfig
from helium.runtime.protocol import (
    HeliumRequestConfig,
    HeliumResponse,
    HeliumSystemProfile,
)


class ParallelAgent(Agent):
    def __init__(
        self,
        expert_system_prompts: list[str],
        writer_system_prompt: str,
        extraction_instruction_fmt: str,
        summary_instruction_fmt: str,
        report_instruction_fmt: str,
        item_op: ops.Op,
        review_chunk_ops: Sequence[ops.Op],
        server_config: HeliumServerConfig | None,
        generation_config: GenerationConfig | None,
    ) -> None:
        super().__init__(
            server_config=server_config,
            expert_system_prompts=expert_system_prompts,
            writer_system_prompt=writer_system_prompt,
            extraction_instruction_fmt=extraction_instruction_fmt,
            summary_instruction_fmt=summary_instruction_fmt,
            report_instruction_fmt=report_instruction_fmt,
            item_op=item_op,
            review_chunk_ops=review_chunk_ops,
            generation_config=generation_config,
        )

    def build_ops(
        self,
        expert_system_prompts: list[str],
        writer_system_prompt: str,
        extraction_instruction_fmt: str,
        summary_instruction_fmt: str,
        report_instruction_fmt: str,
        item_op: ops.Op,
        review_chunk_ops: Sequence[ops.Op],
        generation_config: GenerationConfig | None,
    ) -> list[ops.OutputOp]:
        insight_summaries: list[ops.Op] = []
        for system_prompt in expert_system_prompts:
            extracted_insights: list[ops.Op] = []
            for review_chunk in review_chunk_ops:
                extraction_messages = [
                    ops.OpMessage(role="system", content=system_prompt),
                    ops.OpMessage(
                        role="user",
                        content=ops.format_op(
                            extraction_instruction_fmt, item_op, review_chunk
                        ),
                    ),
                ]
                extracted_insights.append(
                    ops.llm_chat(
                        extraction_messages, generation_config, return_history=False
                    )
                )

            summary_messages = [
                ops.OpMessage(role="system", content=system_prompt),
                ops.OpMessage(
                    role="user",
                    content=ops.format_op(
                        summary_instruction_fmt, item_op, *extracted_insights
                    ),
                ),
            ]
            summary_op = ops.llm_chat(
                summary_messages, generation_config, return_history=False
            )
            insight_summaries.append(summary_op)

        report_messages = [
            ops.OpMessage(role="system", content=writer_system_prompt),
            ops.OpMessage(
                role="user",
                content=ops.format_op(
                    report_instruction_fmt, item_op, *insight_summaries
                ),
            ),
        ]
        insight_report = ops.llm_chat(
            report_messages, generation_config, return_history=False
        )

        output_ops = [ops.as_output("insight_report", insight_report)]
        return output_ops


class HeliumParallelProgram(ParallelProgram, HeliumProgram):
    def __init__(
        self,
        request_config: HeliumRequestConfig | None = None,
        server_config: HeliumServerConfig | None = None,
        parallel_agent: ParallelAgent | None = None,
    ) -> None:
        ParallelProgram.__init__(self)
        HeliumProgram.__init__(self, server_config=server_config)
        self.parallel_agent = parallel_agent
        self.request_config = request_config

    def create_agent(
        self,
        expert_system_prompts: list[str],
        writer_system_prompt: str,
        extraction_instruction_fmt: str,
        summary_instruction_fmt: str,
        report_instruction_fmt: str,
        item_descriptions: list[str],
        review_chunks: tuple[list[str], ...],  # chunk -> item reviews
        generation_config: GenerationConfig | None,
        **_,
    ) -> ParallelAgent:
        item_op = ops.InputOp("item")
        review_chunk_ops = [
            ops.InputOp(f"review_chunk_{i}") for i in range(len(review_chunks))
        ]

        if self.parallel_agent is None:
            parallel_agent = ParallelAgent(
                expert_system_prompts=expert_system_prompts,
                writer_system_prompt=writer_system_prompt,
                extraction_instruction_fmt=extraction_instruction_fmt,
                summary_instruction_fmt=summary_instruction_fmt,
                report_instruction_fmt=report_instruction_fmt,
                item_op=item_op,
                review_chunk_ops=review_chunk_ops,
                server_config=self.server_config,
                generation_config=generation_config,
            )
        else:
            parallel_agent = self.parallel_agent
            # Replace LLM ops' generation config
            for op in parallel_agent.graph.iter_ops(ops.LLMOp):
                op.config = generation_config or GenerationConfig.from_env()

        inputs = {
            item_op.name: item_descriptions,
            **{op.name: chunk for op, chunk in zip(review_chunk_ops, review_chunks)},
        }

        parallel_agent.compile(**inputs)
        return parallel_agent

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
    ) -> tuple[OutputType, HeliumSystemProfile]:
        # Prepare inputs
        indices = random_shuffle(list(range(len(item_descriptions))))
        item_descriptions = random_shuffle(item_descriptions, inplace=False)
        review_chunks = tuple(
            random_shuffle(chunks, inplace=False) for chunks in review_chunks
        )

        parallel_agent = self.create_agent(
            expert_system_prompts=expert_system_prompts,
            writer_system_prompt=writer_system_prompt,
            extraction_instruction_fmt=extraction_instruction_fmt,
            summary_instruction_fmt=summary_instruction_fmt,
            report_instruction_fmt=report_instruction_fmt,
            item_descriptions=item_descriptions,
            review_chunks=review_chunks,
            generation_config=generation_config,
        )

        self.start_timer("generate")
        response = await parallel_agent.run_async(self.request_config)
        self.stop_timer()

        outputs = response.outputs["insight_report"]
        insight_reports = self.OutputBuilder().update(zip(indices, outputs)).build()

        return insight_reports, response.system_profile

    async def _precompute(
        self,
        expert_system_prompts: list[str],
        writer_system_prompt: str,
        extraction_instruction_fmt: str,
        summary_instruction_fmt: str,
        report_instruction_fmt: str,
        item_descriptions: list[str],
        review_chunks: tuple[list[str], ...],
        generation_config: GenerationConfig | None,
        precompute_mode: Literal["none", "only", "both"],
        **kwargs,
    ) -> HeliumResponse:
        # Prepare inputs
        item_descriptions = random_shuffle(item_descriptions, inplace=False)
        review_chunks = tuple(
            random_shuffle(chunks, inplace=False) for chunks in review_chunks
        )

        parallel_agent = self.create_agent(
            expert_system_prompts=expert_system_prompts,
            writer_system_prompt=writer_system_prompt,
            extraction_instruction_fmt=extraction_instruction_fmt,
            summary_instruction_fmt=summary_instruction_fmt,
            report_instruction_fmt=report_instruction_fmt,
            item_descriptions=item_descriptions,
            review_chunks=review_chunks,
            generation_config=generation_config,
        )

        request_config = (
            HeliumRequestConfig()
            if self.request_config is None
            else self.request_config.model_copy()
        )
        request_config.precompute_mode = precompute_mode
        return await parallel_agent.run_async(request_config)
