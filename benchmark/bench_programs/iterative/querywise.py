import asyncio
from typing import Any

from bench_programs.iterative.base import IterativeProgram
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


class QueryWiseIterativeProgram(IterativeProgram):
    async def _run(
        self,
        system_prompt: str,
        first_prompt_fmt: str,
        subsequent_prompt_fmt: str,
        document_chunks: tuple[list[str], ...],
        generation_config: GenerationConfig | None,
        **kwargs,
    ) -> tuple[list[str], HeliumSystemProfile]:
        client, generation_config = prepare_openai(generation_config)
        generation_kwargs = generation_config.openai_kwargs()
        base_url = generation_config.base_url

        # Prepare inputs
        flattened = self.flatten_inputs(document_chunks)

        # Start benchmarking
        try_start_benchmark(base_url)

        self.start_timer("generate")
        output_builder = self.OutputBuilder()
        for batch in openai_iter_batch(flattened, BATCH_SIZE):
            batch_outputs = await asyncio.gather(
                *[
                    self._run_chunks(
                        index,
                        client,
                        system_prompt,
                        first_prompt_fmt,
                        subsequent_prompt_fmt,
                        chunks,
                        generation_kwargs,
                    )
                    for index, chunks in batch
                ]
            )
            output_builder.update(batch_outputs)
        self.stop_timer()

        # Stop benchmarking
        system_profile = try_stop_benchmark(base_url)

        return output_builder.build(), system_profile

    async def _run_chunks(
        self,
        index: int,
        client: AsyncOpenAI,
        system_prompt: str,
        first_prompt_fmt: str,
        subsequent_prompt_fmt: str,
        chunks: tuple[str, ...],
        generation_kwargs: dict[str, Any],
    ) -> tuple[int, str]:
        first_chunk, *remaining_chunks = chunks
        system_message = Message("system", system_prompt)
        messages = [
            system_message,
            Message("user", first_prompt_fmt.format(first_chunk)),
        ]
        summary = await openai_generate_async(client, messages, generation_kwargs)
        for chunk in remaining_chunks:
            messages = [
                system_message,
                Message("user", subsequent_prompt_fmt.format(summary, chunk)),
            ]
            summary = await openai_generate_async(client, messages, generation_kwargs)
        return index, summary
