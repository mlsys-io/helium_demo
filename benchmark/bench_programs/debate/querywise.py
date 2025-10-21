import asyncio
from collections.abc import Sequence
from typing import Any

from bench_programs.debate.base import DebateProgram
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


class QueryWiseDebateProgram(DebateProgram):
    async def _run(
        self,
        system_prompt: str,
        role_prompts: list[str] | None,
        context_prompts: list[str] | None,
        context_question_prompts: tuple[list[str], ...],
        revise_prompts: tuple[str, str],
        num_agents: int,
        num_rounds: int,
        generation_config: GenerationConfig | None,
        dump_conversations: bool = False,
        **kwargs,
    ) -> tuple[list[tuple[list[str], ...]], HeliumSystemProfile]:
        client, generation_config = prepare_openai(generation_config)
        generation_kwargs = generation_config.openai_kwargs()
        base_url = generation_config.base_url

        # Prepare inputs
        flattened = self.flatten_inputs(context_prompts, context_question_prompts)

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
                        num_agents,
                        num_rounds,
                        system_prompt,
                        role_prompts,
                        context_prompt,
                        question_prompt,
                        revise_prompts,
                        generation_kwargs,
                    )
                    for context_idx, question_idx, context_prompt, question_prompt in batch
                ]
            )
            for context_idx, question_idx, agent_contexts in batch_outputs:
                for agent_idx, agent_context in enumerate(agent_contexts):
                    output_builder.add(
                        context_idx, question_idx, agent_idx, agent_context[-1].content
                    )
        self.stop_timer()

        # Stop benchmarking
        system_profile = try_stop_benchmark(base_url)

        return output_builder.build(), system_profile

    def _get_revise_message(
        self,
        other_agent_contexts: list[list[Message]],
        revise_prompts: tuple[str, str],
    ) -> str:
        if len(other_agent_contexts) == 0:
            return revise_prompts[0]
        return "\n\n ".join(
            [
                "These are the solutions to the problem from other agents: ",
                *[
                    f"One agent solution: ```{other_context[-1].content}```"
                    for other_context in other_agent_contexts
                ],
                revise_prompts[1],
            ]
        )

    async def _run_question(
        self,
        context_idx: int,
        question_idx: int,
        client: AsyncOpenAI,
        num_agents: int,
        num_rounds: int,
        system_prompt: str,
        role_prompts: list[str] | None,
        context_prompt: str | None,
        question_prompt: str,
        revise_prompts: tuple[str, str],
        generation_kwargs: dict[str, Any],
    ) -> tuple[int, int, list[list[Message]]]:
        maybe_role_prompts: Sequence[str | None]
        if role_prompts is None:
            maybe_role_prompts = [None] * num_agents
        else:
            maybe_role_prompts = role_prompts
        user_prompts = [
            self.build_user_prompt(role_prompt, context_prompt, question_prompt)
            for role_prompt in maybe_role_prompts
        ]
        agent_contexts = [
            [Message("system", system_prompt), Message("user", user_prompt)]
            for user_prompt in user_prompts
        ]
        for round in range(num_rounds):
            for i, agent_context in enumerate(agent_contexts):
                if round > 0:
                    other_agent_contexts = agent_contexts[:i] + agent_contexts[i + 1 :]
                    new_message_content = self._get_revise_message(
                        other_agent_contexts, revise_prompts
                    )
                    agent_context.append(Message("user", new_message_content))

                answer = await openai_generate_async(
                    client, agent_context, generation_kwargs
                )
                agent_context.append(Message("assistant", answer))

        return context_idx, question_idx, agent_contexts
