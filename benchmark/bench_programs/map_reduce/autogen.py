import asyncio
from collections.abc import Sequence

from autogen import AssistantAgent
from bench_programs.map_reduce.base import MapReduceProgram
from bench_programs.utils.autogen import autogen_generate_async, autogen_get_llm_config
from bench_programs.utils.common import try_start_benchmark, try_stop_benchmark

from helium.common import GenerationConfig
from helium.runtime.protocol import HeliumSystemProfile


class AutoGenMapReduceProgram(MapReduceProgram):
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
        base_url, llm_config = autogen_get_llm_config(generation_config)
        expert_agents = [
            AssistantAgent(
                f"expert_{i}",
                system_message=expert_system_prompt,
                llm_config=llm_config,
            )
            for i in range(num_agents)
        ]
        summarizer_agent = AssistantAgent(
            "summarizer",
            system_message=summarizer_system_prompt,
            llm_config=llm_config,
        )

        # Prepare inputs
        flattened = self.flatten_inputs(context_prompts, context_question_prompts)

        # Start benchmarking
        try_start_benchmark(base_url)

        self.start_timer("generate")
        outputs = await asyncio.gather(
            *[
                self._run_question(
                    context_idx,
                    question_idx,
                    expert_agents,
                    summarizer_agent,
                    role_prompts,
                    context_prompt,
                    question_prompt,
                    summary_prompt,
                )
                for context_idx, question_idx, context_prompt, question_prompt in flattened
            ]
        )
        self.stop_timer()

        # Stop benchmarking
        system_profile = try_stop_benchmark(base_url)

        answers = self.OutputBuilder().update(outputs).build()

        return answers, system_profile

    async def _run_question(
        self,
        context_idx: int,
        question_idx: int,
        expert_agents: list[AssistantAgent],
        summarizer_agent: AssistantAgent,
        role_prompts: list[str] | None,
        context_prompt: str | None,
        question_prompt: str,
        summary_prompt: str,
    ) -> tuple[int, int, str]:
        # Expert answers
        maybe_role_prompts: Sequence[str | None]
        if role_prompts is None:
            maybe_role_prompts = [None] * len(expert_agents)
        else:
            maybe_role_prompts = role_prompts
        expert_user_prompts = [
            self.build_user_prompt(role_prompt, context_prompt, question_prompt)
            for role_prompt in maybe_role_prompts
        ]
        expert_contexts: list[list[dict[str, str]]] = [
            [{"role": "user", "content": user_prompt}]
            for user_prompt in expert_user_prompts
        ]
        expert_answers = await asyncio.gather(
            *[
                autogen_generate_async(expert, expert_context)
                for expert, expert_context in zip(expert_agents, expert_contexts)
            ]
        )

        # Answer summarization
        summarizer_user_prompt = summary_prompt
        if context_prompt is not None:
            summarizer_user_prompt += f"\n\nContext: {context_prompt}"
        summarizer_user_prompt += f"\n\nQuestion: {question_prompt}\n"
        for agent_i, expert_answer in enumerate(expert_answers):
            summarizer_user_prompt += f"\nExpert {agent_i + 1} Answer:\n{expert_answer}"
        summarizer_context = [{"role": "user", "content": summarizer_user_prompt}]
        answer_summary = await autogen_generate_async(
            summarizer_agent, summarizer_context
        )

        return context_idx, question_idx, answer_summary
