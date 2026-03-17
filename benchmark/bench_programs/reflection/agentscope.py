from bench_programs.reflection.base import ReflectionProgram
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


class ASReflectionProgram(ReflectionProgram):
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
        self.start_timer("prepare")
        llm_config = agentscope_reinit_from_config(generation_config)
        base_url = llm_config.base_url

        financial_analyst_agent = AgentScopeAgent.dist(
            "financial_analyst", system_prompts[0]
        )
        extraction_critic_agent = AgentScopeAgent.dist(
            "extraction_critic", system_prompts[1]
        )
        calculation_critic_agent = AgentScopeAgent.dist(
            "calculation_critic", system_prompts[2]
        )
        agents = (
            financial_analyst_agent,
            extraction_critic_agent,
            calculation_critic_agent,
        )
        self.stop_timer()

        # Prepare inputs
        flattened = self.flatten_inputs(contexts, context_questions)

        # Start benchmarking
        try_start_benchmark(base_url)

        self.start_timer("generate")
        outputs = [
            self._run_question(
                context_idx,
                question_idx,
                agents,
                context,
                question,
                financial_analyst_fmt,
                extraction_critic_fmt,
                calculation_critic_fmt,
                final_answer_fmt,
                llm_config,
            )
            for context_idx, question_idx, context, question in flattened
        ]
        output_builder = self.OutputBuilder()
        for context_idx, question_idx, answer in outputs:
            output_builder.add(context_idx, question_idx, answer.content)
        self.stop_timer()

        # Stop benchmarking
        system_profile = try_stop_benchmark(base_url)

        AgentScopeAgent.stop_all(agents)

        return output_builder.build(), system_profile

    def _run_question(
        self,
        context_idx: int,
        question_idx: int,
        agents: tuple[RpcObject, RpcObject, RpcObject],
        context: str,
        question: str,
        financial_analyst_fmt: str,
        extraction_critic_fmt: str,
        calculation_critic_fmt: str,
        final_answer_fmt: str,
        generation_config: GenerationConfig,
    ) -> tuple[int, int, PlaceholderMsg]:
        financial_analyst_agent, extraction_critic_agent, calculation_critic_agent = (
            agents
        )

        # Financial Analyst
        financial_analyst_history: list[Msg | FormatMsg | PlaceholderMsg] = [
            Msg(
                "user",
                financial_analyst_fmt.format(context=context, question=question),
                "user",
            )
        ]
        answer = agentscope_call_agent(
            financial_analyst_agent, financial_analyst_history, generation_config
        )
        financial_analyst_history.append(answer)

        # Extraction Critic
        messages = [
            FormatMsg(
                "user",
                extraction_critic_fmt,
                context=context,
                question=question,
                response=answer,
            )
        ]
        extraction_critic = agentscope_call_agent(
            extraction_critic_agent, messages, generation_config
        )

        # Calculation Critic
        messages = [
            FormatMsg(
                "user",
                calculation_critic_fmt,
                context=context,
                question=question,
                response=answer,
                critic=extraction_critic,
            )
        ]
        calculation_critic = agentscope_call_agent(
            calculation_critic_agent, messages, generation_config
        )

        # Final Answer
        financial_analyst_history.append(
            FormatMsg(
                "user",
                final_answer_fmt,
                extraction_critic=extraction_critic,
                calculation_critic=calculation_critic,
            )
        )
        final_answer = agentscope_call_agent(
            financial_analyst_agent, financial_analyst_history, generation_config
        )

        return context_idx, question_idx, final_answer
