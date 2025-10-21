import asyncio
from typing import Any

from bench_programs.reflection.base import ReflectionProgram
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


class QueryWiseReflectionProgram(ReflectionProgram):
    async def _run(
        self,
        contexts: list[str],
        context_questions: tuple[list[str], ...],
        system_prompts: tuple[str, str, str],
        financial_analyst_fmt: str,
        extraction_critic_fmt: str,
        calculation_critic_fmt: str,
        final_answer_fmt: str,
        generation_config: GenerationConfig | None,
        **kwargs,
    ) -> tuple[list[tuple[str, ...]], HeliumSystemProfile]:
        client, generation_config = prepare_openai(generation_config)
        generation_kwargs = generation_config.openai_kwargs()
        base_url = generation_config.base_url

        # Prepare inputs
        flattened = self.flatten_inputs(contexts, context_questions)

        # Start benchmarking
        try_start_benchmark(base_url)

        self.start_timer("generate")
        output_builder = self.OutputBuilder()
        for batch in openai_iter_batch(flattened, BATCH_SIZE):
            batch_outputs = await asyncio.gather(
                *[
                    self._run_question(
                        context_idx,
                        question_idx,
                        client,
                        system_prompts,
                        context,
                        question,
                        financial_analyst_fmt,
                        extraction_critic_fmt,
                        calculation_critic_fmt,
                        final_answer_fmt,
                        generation_kwargs,
                    )
                    for context_idx, question_idx, context, question in batch
                ]
            )
            output_builder.update(batch_outputs)
        self.stop_timer()

        # Stop benchmarking
        system_profile = try_stop_benchmark(base_url)

        return output_builder.build(), system_profile

    async def _run_question(
        self,
        context_idx: int,
        question_idx: int,
        client: AsyncOpenAI,
        system_prompts: tuple[str, str, str],
        context: str,
        question: str,
        financial_analyst_fmt: str,
        extraction_critic_fmt: str,
        calculation_critic_fmt: str,
        final_answer_fmt: str,
        generation_kwargs: dict[str, Any],
    ) -> tuple[int, int, str]:
        fin_system_prompt, ext_system_prompt, calc_system_prompt = system_prompts
        # Financial Analyst
        financial_analyst_history: list[Message] = [
            Message("system", fin_system_prompt),
            Message(
                "user", financial_analyst_fmt.format(context=context, question=question)
            ),
        ]
        answer = await openai_generate_async(
            client, financial_analyst_history, generation_kwargs
        )
        financial_analyst_history.append(Message("assistant", answer))

        # Extraction Critic
        messages = [
            Message("system", ext_system_prompt),
            Message(
                "user",
                extraction_critic_fmt.format(
                    context=context, question=question, response=answer
                ),
            ),
        ]
        extraction_critic = await openai_generate_async(
            client, messages, generation_kwargs
        )

        # Calculation Critic
        messages = [
            Message("system", calc_system_prompt),
            Message(
                "user",
                calculation_critic_fmt.format(
                    context=context,
                    question=question,
                    response=answer,
                    critic=extraction_critic,
                ),
            ),
        ]
        calculation_critic = await openai_generate_async(
            client, messages, generation_kwargs
        )

        # Final Answer
        financial_analyst_history.append(
            Message(
                "user",
                final_answer_fmt.format(
                    extraction_critic=extraction_critic,
                    calculation_critic=calculation_critic,
                ),
            )
        )
        final_answer = await openai_generate_async(
            client, financial_analyst_history, generation_kwargs
        )

        return context_idx, question_idx, final_answer
