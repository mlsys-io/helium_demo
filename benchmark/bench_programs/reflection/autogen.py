import asyncio

from autogen import AssistantAgent
from bench_programs.reflection.base import ReflectionProgram
from bench_programs.utils.autogen import autogen_generate_async, autogen_get_llm_config
from bench_programs.utils.common import try_start_benchmark, try_stop_benchmark

from helium.common import GenerationConfig
from helium.runtime.protocol import HeliumSystemProfile


class AutoGenReflectionProgram(ReflectionProgram):
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
        base_url, llm_config = autogen_get_llm_config(generation_config)
        financial_analyst_agent = AssistantAgent(
            "financial_analyst", system_message=system_prompts[0], llm_config=llm_config
        )
        extraction_critic_agent = AssistantAgent(
            "extraction_critic", system_message=system_prompts[1], llm_config=llm_config
        )
        calculation_critic_agent = AssistantAgent(
            "calculation_critic",
            system_message=system_prompts[2],
            llm_config=llm_config,
        )
        agents = (
            financial_analyst_agent,
            extraction_critic_agent,
            calculation_critic_agent,
        )

        # Prepare inputs
        flattened = self.flatten_inputs(contexts, context_questions)

        # Start benchmarking
        try_start_benchmark(base_url)

        self.start_timer("generate")
        outputs = await asyncio.gather(
            *[
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
                )
                for context_idx, question_idx, context, question in flattened
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
        agents: tuple[AssistantAgent, AssistantAgent, AssistantAgent],
        context: str,
        question: str,
        financial_analyst_fmt: str,
        extraction_critic_fmt: str,
        calculation_critic_fmt: str,
        final_answer_fmt: str,
    ) -> tuple[int, int, str]:
        financial_analyst_agent, extraction_critic_agent, calculation_critic_agent = (
            agents
        )

        # Financial Analyst
        financial_analyst_history: list[dict[str, str]] = [
            {
                "role": "user",
                "content": financial_analyst_fmt.format(
                    context=context, question=question
                ),
            }
        ]
        answer = await autogen_generate_async(
            financial_analyst_agent, financial_analyst_history
        )
        financial_analyst_history.append({"role": "assistant", "content": answer})

        # Extraction Critic
        messages = [
            {
                "role": "user",
                "content": extraction_critic_fmt.format(
                    context=context, question=question, response=answer
                ),
            }
        ]
        extraction_critic = await autogen_generate_async(
            extraction_critic_agent, messages
        )

        # Calculation Critic
        messages = [
            {
                "role": "user",
                "content": calculation_critic_fmt.format(
                    context=context,
                    question=question,
                    response=answer,
                    critic=extraction_critic,
                ),
            }
        ]
        calculation_critic = await autogen_generate_async(
            calculation_critic_agent, messages
        )

        # Final Answer
        financial_analyst_history.append(
            {
                "role": "user",
                "content": final_answer_fmt.format(
                    extraction_critic=extraction_critic,
                    calculation_critic=calculation_critic,
                ),
            }
        )
        final_answer = await autogen_generate_async(
            financial_analyst_agent, financial_analyst_history
        )

        return context_idx, question_idx, final_answer
