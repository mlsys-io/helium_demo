from collections.abc import Sequence
from typing import Literal

from bench_programs.debate.base import DebateProgram
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


class ASDebateProgram(DebateProgram):
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
        self.start_timer("prepare")
        llm_config = agentscope_reinit_from_config(generation_config)
        base_url = llm_config.base_url

        agents = [
            AgentScopeAgent.dist(f"agent_{i}", system_prompt) for i in range(num_agents)
        ]
        self.stop_timer()

        # Prepare inputs
        flattened = self.flatten_inputs(context_prompts, context_question_prompts)

        # Start benchmarking
        try_start_benchmark(base_url)

        self.start_timer("generate")
        outputs = [
            self._run_question(
                context_idx,
                question_idx,
                agents,
                num_rounds,
                role_prompts,
                context_prompt,
                question_prompt,
                revise_prompts,
                llm_config,
            )
            for context_idx, question_idx, context_prompt, question_prompt in flattened
        ]
        output_builder = self.OutputBuilder()
        for context_idx, question_idx, agent_contexts in outputs:
            for agent_idx, agent_context in enumerate(agent_contexts):
                response = agent_context[-1]
                assert isinstance(response, PlaceholderMsg)
                output_builder.add(
                    context_idx, question_idx, agent_idx, response.content
                )
        self.stop_timer()

        # Stop benchmarking
        system_profile = try_stop_benchmark(base_url)

        AgentScopeAgent.stop_all(agents)

        return output_builder.build(), system_profile

    def _get_revise_message(
        self,
        role: Literal["user", "assistant"],
        other_agent_contexts: list[list[Msg | FormatMsg | PlaceholderMsg]],
        revise_prompts: tuple[str, str],
    ) -> Msg | FormatMsg:
        if len(other_agent_contexts) == 0:
            return Msg(role, revise_prompts[0], role)
        format_str = "\n\n ".join(
            [
                "These are the solutions to the problem from other agents: ",
                *[
                    "One agent solution: ```{}```"
                    for _ in range(len(other_agent_contexts))
                ],
                revise_prompts[1],
            ]
        )

        other_agent_messages: list[str | PlaceholderMsg] = []
        for context in other_agent_contexts:
            message = context[-1]
            content: str | PlaceholderMsg
            if isinstance(message, PlaceholderMsg):
                content = message
            else:
                content_ = message.content
                assert isinstance(content_, str)
                content = content_
            other_agent_messages.append(content)

        return FormatMsg(role, format_str, *other_agent_messages)

    def _run_question(
        self,
        context_idx: int,
        question_idx: int,
        agents: list[RpcObject],
        num_rounds: int,
        role_prompts: list[str] | None,
        context_prompt: str | None,
        question_prompt: str,
        revise_prompts: tuple[str, str],
        generation_config: GenerationConfig,
    ) -> tuple[int, int, list[list[Msg | FormatMsg | PlaceholderMsg]]]:
        maybe_role_prompts: Sequence[str | None]
        if role_prompts is None:
            maybe_role_prompts = [None] * len(agents)
        else:
            maybe_role_prompts = role_prompts
        user_prompts = [
            self.build_user_prompt(role_prompt, context_prompt, question_prompt)
            for role_prompt in maybe_role_prompts
        ]
        agent_contexts: list[list[Msg | FormatMsg | PlaceholderMsg]] = [
            [Msg("user", user_prompt, "user")] for user_prompt in user_prompts
        ]
        for round in range(num_rounds):
            if round > 0:
                new_agent_contexts: list[list[Msg | FormatMsg | PlaceholderMsg]] = []
                for i, agent_context in enumerate(agent_contexts):
                    other_agent_contexts = agent_contexts[:i] + agent_contexts[i + 1 :]
                    new_message = self._get_revise_message(
                        "user", other_agent_contexts, revise_prompts
                    )
                    new_agent_contexts.append(agent_context + [new_message])
                agent_contexts = new_agent_contexts
            agent_replies = [
                agentscope_call_agent(agent, agent_context, generation_config)
                for agent, agent_context in zip(agents, agent_contexts)
            ]
            for agent_context, agent_reply in zip(agent_contexts, agent_replies):
                agent_context.append(agent_reply)

        return context_idx, question_idx, agent_contexts
