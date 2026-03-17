from collections.abc import Sequence

from bench_programs.map_reduce.base import MapReduceProgram
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


class ASMapReduceProgram(MapReduceProgram):
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
        self.start_timer("prepare")
        llm_config = agentscope_reinit_from_config(generation_config)
        base_url = llm_config.base_url

        expert_agents = [
            AgentScopeAgent.dist(f"expert_{i}", expert_system_prompt)
            for i in range(num_agents)
        ]
        summarizer_agent = AgentScopeAgent.dist("summarizer", summarizer_system_prompt)
        self.stop_timer()

        # Prepare inputs
        flattened = self.flatten_inputs(context_prompts, context_question_prompts)

        # Start benchmarking
        try_start_benchmark(base_url)

        self.start_timer("generate")
        results = [
            self._run_question(
                context_idx,
                question_idx,
                expert_agents,
                summarizer_agent,
                role_prompts,
                context_prompt,
                question_prompt,
                summary_prompt,
                llm_config,
            )
            for context_idx, question_idx, context_prompt, question_prompt in flattened
        ]
        output_builder = self.OutputBuilder()
        for context_idx, question_idx, answer in results:
            output_builder.add(context_idx, question_idx, answer.content)
        self.stop_timer()

        # Stop benchmarking
        system_profile = try_stop_benchmark(base_url)

        AgentScopeAgent.stop_all(expert_agents + [summarizer_agent])

        return output_builder.build(), system_profile

    def _run_question(
        self,
        context_idx: int,
        question_idx: int,
        expert_agents: list[RpcObject],
        summarizer_agent: RpcObject,
        role_prompts: list[str] | None,
        context_prompt: str | None,
        question_prompt: str,
        summary_prompt: str,
        generation_config: GenerationConfig,
    ) -> tuple[int, int, PlaceholderMsg]:
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
        expert_contexts: list[list[Msg | PlaceholderMsg]] = [
            [Msg("user", user_prompt, "user")] for user_prompt in expert_user_prompts
        ]
        expert_answers = [
            agentscope_call_agent(expert, expert_context, generation_config)
            for expert, expert_context in zip(expert_agents, expert_contexts)
        ]

        # Answer summarization
        summarizer_user_prompt = summary_prompt
        format_args = []
        if context_prompt is not None:
            summarizer_user_prompt += "\n\nContext: {}"
            format_args.append(context_prompt)
        summarizer_user_prompt += "\n\nQuestion: {}\n"
        format_args.append(question_prompt)
        for agent_i in range(len(expert_answers)):
            summarizer_user_prompt += f"\nExpert {agent_i + 1} Answer:\n{{}}"
        summarizer_context = [
            FormatMsg("user", summarizer_user_prompt, *format_args, *expert_answers)
        ]
        answer_summary = agentscope_call_agent(
            summarizer_agent, summarizer_context, generation_config
        )

        return context_idx, question_idx, answer_summary
