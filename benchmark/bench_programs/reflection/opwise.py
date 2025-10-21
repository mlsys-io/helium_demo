from typing import Any

from bench_programs.reflection.base import ReflectionProgram
from bench_programs.utils.common import try_start_benchmark, try_stop_benchmark
from bench_programs.utils.openai import openai_generate_async, prepare_openai
from bench_programs.utils.opwise import WorkflowDAG
from openai import AsyncOpenAI

from helium.common import GenerationConfig, Message
from helium.runtime.protocol import HeliumSystemProfile


class OpWiseReflectionProgram(ReflectionProgram):
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
        inputs = [
            {
                "context_idx": context_idx,
                "question_idx": question_idx,
                "context": context,
                "question": question,
                "financial_analyst_context": [Message("system", system_prompts[0])],
            }
            for context_idx, question_idx, context, question in flattened
        ]

        workflow = self._build_workflow(
            client,
            system_prompts,
            financial_analyst_fmt,
            extraction_critic_fmt,
            calculation_critic_fmt,
            final_answer_fmt,
            generation_kwargs,
        )

        # Start benchmarking
        try_start_benchmark(base_url)

        self.start_timer("generate")
        states = await workflow.execute(inputs)
        self.stop_timer()

        # Stop benchmarking
        system_profile = try_stop_benchmark(base_url)

        output_builder = self.OutputBuilder()
        for state in states:
            context_idx = state["context_idx"]
            question_idx = state["question_idx"]
            answer = state["financial_analyst_context"][-1].content
            output_builder.add(context_idx, question_idx, answer)

        return output_builder.build(), system_profile

    def _build_workflow(
        self,
        client: AsyncOpenAI,
        system_prompts: tuple[str, str, str],
        financial_analyst_fmt: str,
        extraction_critic_fmt: str,
        calculation_critic_fmt: str,
        final_answer_fmt: str,
        generation_kwargs: dict[str, Any],
    ) -> WorkflowDAG:
        _, ext_system_prompt, calc_system_prompt = system_prompts

        async def financial_analyst(state: dict) -> dict:
            context = state["context"]
            question = state["question"]
            financial_analyst_context: list[Message] = state[
                "financial_analyst_context"
            ]
            financial_analyst_context.append(
                Message(
                    "user",
                    financial_analyst_fmt.format(context=context, question=question),
                )
            )
            answer = await openai_generate_async(
                client, financial_analyst_context, generation_kwargs
            )
            financial_analyst_context.append(Message("assistant", answer))
            return state

        async def extraction_critic(state: dict) -> dict:
            context = state["context"]
            question = state["question"]
            answer = state["financial_analyst_context"][-1].content
            messages = [
                Message("system", ext_system_prompt),
                Message(
                    "user",
                    extraction_critic_fmt.format(
                        context=context, question=question, response=answer
                    ),
                ),
            ]
            critic = await openai_generate_async(client, messages, generation_kwargs)
            state["extraction_critic"] = critic
            return state

        async def calculation_critic(state: dict) -> dict:
            context = state["context"]
            question = state["question"]
            answer = state["financial_analyst_context"][-1].content
            critic = state["extraction_critic"]
            messages = [
                Message("system", calc_system_prompt),
                Message(
                    "user",
                    calculation_critic_fmt.format(
                        context=context,
                        question=question,
                        response=answer,
                        critic=critic,
                    ),
                ),
            ]
            calc_critic = await openai_generate_async(
                client, messages, generation_kwargs
            )
            state["calculation_critic"] = calc_critic
            return state

        async def final_answer(state: dict) -> dict:
            messages: list[Message] = state["financial_analyst_context"]
            extraction_critic = state["extraction_critic"]
            calculation_critic = state["calculation_critic"]
            messages.append(
                Message(
                    "user",
                    final_answer_fmt.format(
                        extraction_critic=extraction_critic,
                        calculation_critic=calculation_critic,
                    ),
                )
            )
            answer = await openai_generate_async(client, messages, generation_kwargs)
            messages.append(Message("assistant", answer))
            return state

        workflow = WorkflowDAG()
        workflow.add_node("financial_analyst", financial_analyst)
        workflow.add_node("extraction_critic", extraction_critic)
        workflow.add_node("calculation_critic", calculation_critic)
        workflow.add_node("final_answer", final_answer)

        workflow.add_edge("financial_analyst", "extraction_critic")
        workflow.add_edge("extraction_critic", "calculation_critic")
        workflow.add_edge("calculation_critic", "final_answer")

        return workflow
