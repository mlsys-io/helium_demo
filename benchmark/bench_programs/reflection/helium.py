from collections.abc import Sequence
from typing import Literal

from bench_programs.reflection.base import OutputType, ReflectionProgram
from bench_programs.utils.common import random_shuffle

from helium import ops
from helium.common import GenerationConfig
from helium.frontend.agents import Agent
from helium.frontend.programs import Program as HeliumProgram
from helium.runtime import HeliumServerConfig
from helium.runtime.protocol import (
    HeliumRequestConfig,
    HeliumResponse,
    HeliumSystemProfile,
)


class ReflectionAgent(Agent):
    def __init__(
        self,
        system_prompts: tuple[str, str, str],
        context_op: ops.Op,
        question_ops: Sequence[ops.Op],
        financial_analyst_fmt: str,
        extraction_critic_fmt: str,
        calculation_critic_fmt: str,
        final_answer_fmt: str,
        server_config: HeliumServerConfig | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        super().__init__(
            server_config=server_config,
            system_prompts=system_prompts,
            context_op=context_op,
            question_ops=question_ops,
            financial_analyst_fmt=financial_analyst_fmt,
            extraction_critic_fmt=extraction_critic_fmt,
            calculation_critic_fmt=calculation_critic_fmt,
            final_answer_fmt=final_answer_fmt,
            generation_config=generation_config,
        )

    def build_ops(
        self,
        system_prompts: tuple[str, str, str],
        context_op: ops.Op,
        question_ops: Sequence[ops.Op],
        financial_analyst_fmt: str,
        extraction_critic_fmt: str,
        calculation_critic_fmt: str,
        final_answer_fmt: str,
        generation_config: GenerationConfig | None = None,
    ) -> list[ops.OutputOp]:
        (
            financial_analyst_system_prompt,
            extraction_critic_system_prompt,
            calculation_critic_system_prompt,
        ) = system_prompts
        output_ops: list[ops.OutputOp] = []
        for question_i, question_op in enumerate(question_ops):
            user_prompt = ops.format_op(
                financial_analyst_fmt, context=context_op, question=question_op
            )
            agent_messages = [
                ops.OpMessage(role="system", content=financial_analyst_system_prompt),
                ops.OpMessage(role="user", content=user_prompt),
            ]
            financial_agent_history: ops.Op = ops.llm_chat(
                agent_messages, generation_config, return_history=True
            )

            answer = ops.get_last_message(financial_agent_history)
            user_prompt = ops.format_op(
                extraction_critic_fmt,
                context=context_op,
                question=question_op,
                response=answer,
            )
            agent_messages = [
                ops.OpMessage(role="system", content=extraction_critic_system_prompt),
                ops.OpMessage(role="user", content=user_prompt),
            ]
            extraction_critic = ops.llm_chat(
                agent_messages, generation_config, return_history=False
            )

            user_prompt = ops.format_op(
                calculation_critic_fmt,
                context=context_op,
                question=question_op,
                response=answer,
                critic=extraction_critic,
            )
            agent_messages = [
                ops.OpMessage(role="system", content=calculation_critic_system_prompt),
                ops.OpMessage(role="user", content=user_prompt),
            ]
            calculation_critic = ops.llm_chat(
                agent_messages, generation_config, return_history=False
            )

            user_prompt = ops.format_op(
                final_answer_fmt,
                extraction_critic=extraction_critic,
                calculation_critic=calculation_critic,
            )
            financial_agent_history = ops.append_message(
                financial_agent_history, user_prompt
            )
            final_answer = ops.llm_chat(
                financial_agent_history, generation_config, return_history=False
            )

            output_ops.append(ops.as_output(f"question_{question_i}", final_answer))

        return output_ops


class HeliumReflectionProgram(ReflectionProgram, HeliumProgram):
    def __init__(
        self,
        request_config: HeliumRequestConfig | None = None,
        server_config: HeliumServerConfig | None = None,
        reflection_agent: ReflectionAgent | None = None,
    ) -> None:
        ReflectionProgram.__init__(self)
        HeliumProgram.__init__(self, server_config=server_config)
        self.reflection_agent = reflection_agent
        self.request_config = request_config

    def create_agent(
        self,
        contexts: list[str],
        context_questions: tuple[list[str], ...],
        system_prompts: tuple[str, str, str],
        financial_analyst_fmt: str,
        extraction_critic_fmt: str,
        calculation_critic_fmt: str,
        final_answer_fmt: str,
        generation_config: GenerationConfig | None,
        **_,
    ) -> ReflectionAgent:
        context_op = ops.InputOp("context")
        question_ops = [
            ops.InputOp(f"question_{i}") for i in range(len(context_questions))
        ]

        if self.reflection_agent is None:
            reflection_agent = ReflectionAgent(
                system_prompts=system_prompts,
                context_op=context_op,
                question_ops=question_ops,
                financial_analyst_fmt=financial_analyst_fmt,
                extraction_critic_fmt=extraction_critic_fmt,
                calculation_critic_fmt=calculation_critic_fmt,
                final_answer_fmt=final_answer_fmt,
                server_config=self.server_config,
                generation_config=generation_config,
            )
        else:
            reflection_agent = self.reflection_agent
            # Replace LLM ops' generation config
            for op in reflection_agent.graph.iter_ops(ops.LLMOp):
                op.config = generation_config or GenerationConfig.from_env()

        inputs = {
            context_op.name: contexts,
            **{
                op.name: questions
                for op, questions in zip(question_ops, context_questions)
            },
        }
        reflection_agent.compile(**inputs)
        return reflection_agent

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
    ) -> tuple[OutputType, HeliumSystemProfile]:
        # Prepare inputs
        context_indices = random_shuffle(list(range(len(contexts))))
        contexts = random_shuffle(contexts, inplace=False)
        context_questions = tuple(
            random_shuffle(questions, inplace=False) for questions in context_questions
        )

        reflection_agent = self.create_agent(
            contexts=contexts,
            context_questions=context_questions,
            system_prompts=system_prompts,
            financial_analyst_fmt=financial_analyst_fmt,
            extraction_critic_fmt=extraction_critic_fmt,
            calculation_critic_fmt=calculation_critic_fmt,
            final_answer_fmt=final_answer_fmt,
            generation_config=generation_config,
        )

        self.start_timer("generate")
        response = await reflection_agent.run_async(self.request_config)
        self.stop_timer()

        output_builder = self.OutputBuilder()
        for question_idx in range(len(context_questions)):
            question_answers = response.outputs[f"question_{question_idx}"]
            for context_idx, answer in zip(context_indices, question_answers):
                output_builder.add(context_idx, question_idx, answer)

        return output_builder.build(), response.system_profile

    async def _precompute(
        self,
        contexts: list[str],
        context_questions: tuple[list[str], ...],
        system_prompts: tuple[str, str, str],
        financial_analyst_fmt: str,
        extraction_critic_fmt: str,
        calculation_critic_fmt: str,
        final_answer_fmt: str,
        generation_config: GenerationConfig | None,
        precompute_mode: Literal["none", "only", "both"],
        **kwargs,
    ) -> HeliumResponse:
        # Prepare inputs
        contexts = random_shuffle(contexts, inplace=False)
        context_questions = tuple(
            random_shuffle(questions, inplace=False) for questions in context_questions
        )

        reflection_agent = self.create_agent(
            contexts=contexts,
            context_questions=context_questions,
            system_prompts=system_prompts,
            financial_analyst_fmt=financial_analyst_fmt,
            extraction_critic_fmt=extraction_critic_fmt,
            calculation_critic_fmt=calculation_critic_fmt,
            final_answer_fmt=final_answer_fmt,
            generation_config=generation_config,
        )

        request_config = (
            HeliumRequestConfig()
            if self.request_config is None
            else self.request_config.model_copy()
        )
        request_config.precompute_mode = precompute_mode
        return await reflection_agent.run_async(request_config)
