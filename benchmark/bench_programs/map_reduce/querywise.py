import asyncio
from collections.abc import Sequence
from typing import Any

from bench_programs.map_reduce.base import MapReduceProgram
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


class QueryWiseMapReduceProgram(MapReduceProgram):
    async def _run(
        self,
        expert_system_prompt: str,
        summarizer_system_prompt: str,
        summary_prompt: str,
        role_prompts: list[str] | None,
        context_prompts: list[str] | None,
        context_question_prompts: tuple[list[str], ...],
        num_agents: int,
        generation_config: GenerationConfig | None,
        **kwargs,
    ) -> tuple[list[tuple[str, ...]], HeliumSystemProfile]:
        client, generation_config = prepare_openai(generation_config)
        generation_kwargs = generation_config.openai_kwargs()
        base_url = generation_config.base_url

        # Prepare inputs
        flattened = self.flatten_inputs(context_prompts, context_question_prompts)

        # Start benchmarking
        try_start_benchmark(base_url)

        output_builder = self.OutputBuilder()
        self.start_timer("generate")
        for batch in openai_iter_batch(flattened, BATCH_SIZE):
            batch_outputs = await asyncio.gather(
                *[
                    self._run_question(
                        context_idx,
                        question_idx,
                        client,
                        num_agents,
                        expert_system_prompt,
                        summarizer_system_prompt,
                        role_prompts,
                        context_prompt,
                        question_prompt,
                        summary_prompt,
                        generation_kwargs,
                    )
                    for context_idx, question_idx, context_prompt, question_prompt in batch
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
        num_agents: int,
        expert_system_prompt: str,
        summarizer_system_prompt: str,
        role_prompts: list[str] | None,
        context_prompt: str | None,
        question_prompt: str,
        summary_prompt: str,
        generation_kwargs: dict[str, Any],
    ) -> tuple[int, int, str]:
        # Expert answers
        maybe_role_prompts: Sequence[str | None]
        if role_prompts is None:
            maybe_role_prompts = [None] * num_agents
        else:
            maybe_role_prompts = role_prompts
        expert_user_prompts = [
            self.build_user_prompt(role_prompt, context_prompt, question_prompt)
            for role_prompt in maybe_role_prompts
        ]
        expert_answers: list[str] = []
        for user_prompt in expert_user_prompts:
            context = [
                Message("system", expert_system_prompt),
                Message("user", user_prompt),
            ]
            answer = await openai_generate_async(client, context, generation_kwargs)
            expert_answers.append(answer)

        # Answer summarization
        summarizer_user_prompt = summary_prompt
        if context_prompt is not None:
            summarizer_user_prompt += f"\n\nContext: {context_prompt}"
        summarizer_user_prompt += f"\n\nQuestion: {question_prompt}\n"
        for agent_i, expert_answer in enumerate(expert_answers):
            summarizer_user_prompt += f"\nExpert {agent_i + 1} Answer:\n{expert_answer}"
        context = [
            Message("system", summarizer_system_prompt),
            Message("user", summarizer_user_prompt),
        ]
        answer_summary = await openai_generate_async(client, context, generation_kwargs)

        return context_idx, question_idx, answer_summary
