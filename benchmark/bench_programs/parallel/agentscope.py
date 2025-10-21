from bench_programs.parallel.base import ParallelProgram
from bench_programs.utils.agentscope import (
    AgentScopeAgent,
    FormatMsg,
    Msg,
    PlaceholderMsg,
    RpcObject,
    agentscope_call_agent,
    agentscope_reinit_from_config,
)
from bench_programs.utils.common import try_start_benchmark, try_stop_benchmark

from helium.common import GenerationConfig
from helium.runtime.protocol import HeliumSystemProfile


class ASParallelProgram(ParallelProgram):
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
        self.start_timer("prepare")
        llm_config = agentscope_reinit_from_config(generation_config)
        base_url = llm_config.base_url

        expert_agents = [
            AgentScopeAgent.dist(f"expert_{i}", prompt)
            for i, prompt in enumerate(expert_system_prompts)
        ]
        writer_agent = AgentScopeAgent.dist("writer", writer_system_prompt)
        self.stop_timer()

        # Prepare inputs
        flattened = self.flatten_inputs(item_descriptions, review_chunks)

        # Start benchmarking
        try_start_benchmark(base_url)

        self.start_timer("generate")
        outputs = [
            self._run_item(
                index,
                expert_agents,
                writer_agent,
                extraction_instruction_fmt,
                summary_instruction_fmt,
                report_instruction_fmt,
                item_description,
                chunks,
                llm_config,
            )
            for index, item_description, chunks in flattened
        ]
        output_builder = self.OutputBuilder()
        for index, output in outputs:
            output_builder.add(index, output.content)
        self.stop_timer()

        # Stop benchmarking
        system_profile = try_stop_benchmark(base_url)

        AgentScopeAgent.stop_all(expert_agents + [writer_agent])

        return output_builder.build(), system_profile

    def _run_item(
        self,
        index: int,
        expert_agents: list[RpcObject],
        writer_agent: RpcObject,
        extraction_instruction_fmt: str,
        summary_instruction_fmt: str,
        report_instruction_fmt: str,
        item_description: str,
        review_chunks: tuple[str, ...],
        generation_config: GenerationConfig,
    ) -> tuple[int, PlaceholderMsg]:
        insight_summaries: list[PlaceholderMsg] = []
        for expert_agent in expert_agents:
            extracted_insights: list[PlaceholderMsg] = []
            for review_chunk in review_chunks:
                extraction_messages = [
                    Msg(
                        "user",
                        extraction_instruction_fmt.format(
                            item_description, review_chunk
                        ),
                        "user",
                    )
                ]
                extracted_insights.append(
                    agentscope_call_agent(
                        expert_agent, extraction_messages, generation_config
                    )
                )

            summary_messages = [
                FormatMsg(
                    "user",
                    summary_instruction_fmt,
                    item_description,
                    *extracted_insights,
                )
            ]
            insight_summaries.append(
                agentscope_call_agent(expert_agent, summary_messages, generation_config)
            )

        report_message = [
            FormatMsg(
                "user", report_instruction_fmt, item_description, *insight_summaries
            )
        ]
        insight_report = agentscope_call_agent(
            writer_agent, report_message, generation_config
        )

        return index, insight_report
