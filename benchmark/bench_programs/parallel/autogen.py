import asyncio

from autogen import AssistantAgent
from bench_programs.parallel.base import ParallelProgram
from bench_programs.utils.autogen import autogen_generate_async, autogen_get_llm_config
from bench_programs.utils.common import try_start_benchmark, try_stop_benchmark

from helium.common import GenerationConfig
from helium.runtime.protocol import HeliumSystemProfile


class AutoGenParallelProgram(ParallelProgram):
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
        base_url, llm_config = autogen_get_llm_config(generation_config)
        expert_agents = [
            AssistantAgent(
                "expert_" + str(i), system_message=prompt, llm_config=llm_config
            )
            for i, prompt in enumerate(expert_system_prompts)
        ]
        writer_agent = AssistantAgent(
            "writer",
            system_message=writer_system_prompt,
            llm_config=llm_config,
        )

        # Prepare inputs
        flattened = self.flatten_inputs(item_descriptions, review_chunks)

        # Start benchmarking
        try_start_benchmark(base_url)

        self.start_timer("generate")
        outputs = await asyncio.gather(
            *[
                self._run_item(
                    index,
                    expert_agents,
                    writer_agent,
                    extraction_instruction_fmt,
                    summary_instruction_fmt,
                    report_instruction_fmt,
                    item_description,
                    chunks,
                )
                for index, item_description, chunks in flattened
            ]
        )
        self.stop_timer()

        # Stop benchmarking
        system_profile = try_stop_benchmark(base_url)

        insight_reports = self.OutputBuilder().update(outputs).build()

        return insight_reports, system_profile

    async def _run_item(
        self,
        index: int,
        expert_agents: list[AssistantAgent],
        writer_agent: AssistantAgent,
        extraction_instruction_fmt: str,
        summary_instruction_fmt: str,
        report_instruction_fmt: str,
        item_description: str,
        review_chunks: tuple[str, ...],
    ) -> tuple[int, str]:
        async def summarize_insights(
            expert_agent: AssistantAgent, task: list[asyncio.Task[str]]
        ) -> str:
            extracted_insights = await asyncio.gather(*task)
            summary_message = [
                {
                    "role": "user",
                    "content": summary_instruction_fmt.format(
                        item_description, *extracted_insights
                    ),
                }
            ]
            return await autogen_generate_async(expert_agent, summary_message)

        summary_tasks: list[asyncio.Task[str]] = []
        for expert_agent in expert_agents:
            expert_tasks: list[asyncio.Task[str]] = []
            for review_chunk in review_chunks:
                extraction_message = [
                    {
                        "role": "user",
                        "content": extraction_instruction_fmt.format(
                            item_description, review_chunk
                        ),
                    }
                ]
                expert_tasks.append(
                    asyncio.create_task(
                        autogen_generate_async(expert_agent, extraction_message)
                    )
                )
            summary_tasks.append(
                asyncio.create_task(summarize_insights(expert_agent, expert_tasks))
            )

        insight_summaries = await asyncio.gather(*summary_tasks)
        report_message = [
            {
                "role": "user",
                "content": report_instruction_fmt.format(
                    item_description, *insight_summaries
                ),
            }
        ]
        insight_report = await autogen_generate_async(writer_agent, report_message)

        return index, insight_report
