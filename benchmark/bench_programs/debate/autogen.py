import asyncio
from collections.abc import Sequence

from autogen import AssistantAgent
from bench_programs.debate.base import DebateProgram
from bench_programs.utils.autogen import autogen_generate_async, autogen_get_llm_config
from bench_programs.utils.common import try_start_benchmark, try_stop_benchmark

from helium.common import GenerationConfig
from helium.runtime.protocol import HeliumSystemProfile


class AutoGenDebateProgram(DebateProgram):
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
        base_url, llm_config = autogen_get_llm_config(generation_config)
        agents = [
            AssistantAgent(
                f"agent_{i}", system_message=system_prompt, llm_config=llm_config
            )
            for i in range(num_agents)
        ]

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
                    agents,
                    num_rounds,
                    role_prompts,
                    context_prompt,
                    question_prompt,
                    revise_prompts,
                    dump_conversations,
                )
                for context_idx, question_idx, context_prompt, question_prompt in flattened
            ]
        )
        self.stop_timer()

        # Stop benchmarking
        system_profile = try_stop_benchmark(base_url)

        output_builder = self.OutputBuilder()
        for context_idx, question_idx, agent_contexts in outputs:
            for agent_idx, agent_context in enumerate(agent_contexts):
                response = agent_context[-1]["content"]
                output_builder.add(context_idx, question_idx, agent_idx, response)

        if dump_conversations:
            self._dump_conversations("logs/debate_dump_autogen.txt")

        return output_builder.build(), system_profile

    def _get_revise_message(
        self,
        other_agent_contexts: list[list[dict[str, str]]],
        revise_prompts: tuple[str, str],
    ) -> str:
        if len(other_agent_contexts) == 0:
            return revise_prompts[0]
        return "\n\n ".join(
            [
                "These are the solutions to the problem from other agents: ",
                *[
                    f"One agent solution: ```{other_context[-1]['content']}```"
                    for other_context in other_agent_contexts
                ],
                revise_prompts[1],
            ]
        )

    async def _run_question(
        self,
        context_idx: int,
        question_idx: int,
        agents: list[AssistantAgent],
        num_rounds: int,
        role_prompts: list[str] | None,
        context_prompt: str | None,
        question_prompt: str,
        revise_prompts: tuple[str, str],
        dump_conversations: bool,
    ) -> tuple[int, int, list[list[dict[str, str]]]]:
        maybe_role_prompts: Sequence[str | None]
        if role_prompts is None:
            maybe_role_prompts = [None] * len(agents)
        else:
            maybe_role_prompts = role_prompts
        user_prompts = [
            self.build_user_prompt(role_prompt, context_prompt, question_prompt)
            for role_prompt in maybe_role_prompts
        ]
        agent_contexts: list[list[dict[str, str]]] = [
            [{"role": "user", "content": user_prompt}] for user_prompt in user_prompts
        ]
        for round in range(num_rounds):
            if round > 0:
                new_agent_contexts: list[list[dict[str, str]]] = []
                for i, agent_context in enumerate(agent_contexts):
                    other_agent_contexts = agent_contexts[:i] + agent_contexts[i + 1 :]
                    new_message_content = self._get_revise_message(
                        other_agent_contexts, revise_prompts
                    )
                    new_agent_contexts.append(
                        [
                            *agent_context,
                            {"role": "user", "content": new_message_content},
                        ]
                    )
                agent_contexts = new_agent_contexts
            agent_replies = await asyncio.gather(
                *[
                    autogen_generate_async(agent, agent_context)
                    for agent, agent_context in zip(agents, agent_contexts)
                ]
            )
            for agent_context, agent_reply in zip(agent_contexts, agent_replies):
                if isinstance(agent_reply, str):
                    agent_context.append({"role": "assistant", "content": agent_reply})
                elif isinstance(agent_reply, dict):
                    agent_context.append(
                        {"role": "assistant", "content": agent_reply["content"]}
                    )
                else:
                    raise ValueError("Invalid agent reply type")

        if dump_conversations:
            for agent_idx, agent_context in enumerate(agent_contexts):
                self._add_conversation(
                    context_idx, question_idx, agent_idx, agent_context
                )

        return context_idx, question_idx, agent_contexts
