import asyncio

from autogen import AssistantAgent
from bench_programs.iterative.base import IterativeProgram
from bench_programs.utils.autogen import autogen_generate_async, autogen_get_llm_config
from bench_programs.utils.common import try_start_benchmark, try_stop_benchmark

from helium.common import GenerationConfig
from helium.runtime.protocol import HeliumSystemProfile


class AutoGenIterativeProgram(IterativeProgram):
    async def _run(
        self,
        system_prompt: str,
        first_prompt_fmt: str,
        subsequent_prompt_fmt: str,
        document_chunks: tuple[list[str], ...],
        generation_config: GenerationConfig | None,
        **kwargs,
    ) -> tuple[list[str], HeliumSystemProfile]:
        base_url, llm_config = autogen_get_llm_config(generation_config)
        agent = AssistantAgent(
            "agent", system_message=system_prompt, llm_config=llm_config
        )

        # Prepare inputs
        flattened = self.flatten_inputs(document_chunks)

        # Start benchmarking
        try_start_benchmark(base_url)

        self.start_timer("generate")
        outputs = await asyncio.gather(
            *[
                self._run_chunks(
                    index, agent, first_prompt_fmt, subsequent_prompt_fmt, chunks
                )
                for index, chunks in flattened
            ]
        )
        self.stop_timer()

        # Stop benchmarking
        system_profile = try_stop_benchmark(base_url)

        summaries = self.OutputBuilder().update(outputs).build()

        return summaries, system_profile

    async def _run_chunks(
        self,
        index: int,
        agent: AssistantAgent,
        first_prompt_fmt: str,
        subsequent_prompt_fmt: str,
        chunks: tuple[str, ...],
    ) -> tuple[int, str]:
        first_chunk, *remaining_chunks = chunks
        message = [{"role": "user", "content": first_prompt_fmt.format(first_chunk)}]
        summary = await autogen_generate_async(agent, message)
        for chunk in remaining_chunks:
            message = [
                {
                    "role": "user",
                    "content": subsequent_prompt_fmt.format(summary, chunk),
                }
            ]
            summary = await autogen_generate_async(agent, message)
        return index, summary
