from collections.abc import Sequence
from typing import Literal

from bench_programs.map_reduce.base import MapReduceProgram, OutputType
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


class MapReduceAgent(Agent):
    def __init__(
        self,
        num_agents: int,
        expert_system_prompt: str,
        summarizer_system_prompt: str,
        summary_prompt: str,
        role_prompts: Sequence[str] | None,
        context_op: ops.Op | None,
        question_ops: Sequence[ops.Op],
        server_config: HeliumServerConfig | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        if num_agents <= 0:
            raise ValueError(f"num_agents must be positive, got {num_agents}")
        super().__init__(
            server_config=server_config,
            num_agents=num_agents,
            expert_system_prompt=expert_system_prompt,
            summarizer_system_prompt=summarizer_system_prompt,
            summary_prompt=summary_prompt,
            role_prompts=role_prompts,
            context_op=context_op,
            question_ops=question_ops,
            generation_config=generation_config,
        )

    def build_ops(
        self,
        num_agents: int,
        expert_system_prompt: str,
        summarizer_system_prompt: str,
        summary_prompt: str,
        role_prompts: Sequence[str] | None,
        context_op: ops.Op | None,
        question_ops: Sequence[ops.Op],
        generation_config: GenerationConfig | None = None,
    ) -> list[ops.OutputOp]:
        # Build experts' prompt format str
        expert_prompt_fmts: list[str] | None
        if role_prompts is None and context_op is None:
            expert_prompt_fmts = None
        else:
            context_fmt = None if context_op is None else "{context}"
            maybe_role_prompts: Sequence[str | None]
            if role_prompts is None:
                maybe_role_prompts = [None] * num_agents
            else:
                maybe_role_prompts = role_prompts
            expert_prompt_fmts = [
                MapReduceProgram.build_user_prompt(
                    role_prompt, context_fmt, "{question}"
                )
                for role_prompt in maybe_role_prompts
            ]

        # Build summarizer's prompt format str
        summarizer_prompt_fmt = summary_prompt
        if context_op is not None:
            summarizer_prompt_fmt += "\n\nContext: {context}"
        summarizer_prompt_fmt += "\n\nQuestion: {question}\n"
        for agent_i in range(num_agents):
            summarizer_prompt_fmt += (
                f"\nExpert {agent_i + 1} Answer:\n{{agent_{agent_i}}}"
            )

        output_ops: list[ops.OutputOp] = []
        for question_i, question_op in enumerate(question_ops):
            # Create user prompt ops
            if expert_prompt_fmts is None:
                context_question_ops = [question_op for _ in range(num_agents)]
            else:
                format_kwargs = {"question": question_op}
                if context_op is not None:
                    format_kwargs["context"] = context_op
                context_question_ops = [
                    ops.format_op(user_prompt_fmt, **format_kwargs)
                    for user_prompt_fmt in expert_prompt_fmts
                ]

            # Expert answers
            expert_message_list = [
                ops.message_data(
                    [
                        ops.OpMessage(role="system", content=expert_system_prompt),
                        ops.OpMessage(role="user", content=context_question_op),
                    ]
                )
                for context_question_op in context_question_ops
            ]
            expert_answers = [
                ops.llm_chat(message, generation_config, return_history=False)
                for message in expert_message_list
            ]

            # Answer summarization
            summarizer_format_kwargs: dict[str, ops.Op] = {
                f"agent_{i}": expert_answer
                for i, expert_answer in enumerate(expert_answers)
            }
            summarizer_format_kwargs["question"] = question_op
            if context_op is not None:
                summarizer_format_kwargs["context"] = context_op
            summarizer_user_prompt = ops.format_op(
                summarizer_prompt_fmt, **summarizer_format_kwargs
            )
            summarizer_message = [
                ops.OpMessage(role="system", content=summarizer_system_prompt),
                ops.OpMessage(role="user", content=summarizer_user_prompt),
            ]
            answer_summary = ops.llm_chat(
                ops.message_data(summarizer_message),
                generation_config,
                return_history=False,
            )
            output_ops.append(ops.OutputOp(f"question_{question_i}", answer_summary))
        return output_ops


class HeliumMapReduceProgram(MapReduceProgram, HeliumProgram):
    def __init__(
        self,
        request_config: HeliumRequestConfig | None = None,
        server_config: HeliumServerConfig | None = None,
        map_reduce_agent: MapReduceAgent | None = None,
    ) -> None:
        MapReduceProgram.__init__(self)
        HeliumProgram.__init__(self, server_config=server_config)
        self.map_reduce_agent = map_reduce_agent
        self.request_config = request_config

    def create_agent(
        self,
        expert_system_prompt: str,
        summarizer_system_prompt: str,
        summary_prompt: str,
        role_prompts: list[str] | None,
        context_prompts: list[str] | None,
        context_question_prompts: tuple[list[str], ...],
        num_agents: int,
        generation_config: GenerationConfig | None,
        **_,
    ) -> MapReduceAgent:
        context_op = None if context_prompts is None else ops.InputOp("context")
        question_ops = [
            ops.InputOp(f"question_{i}") for i in range(len(context_question_prompts))
        ]

        if self.map_reduce_agent is None:
            map_reduce_agent = MapReduceAgent(
                num_agents=num_agents,
                expert_system_prompt=expert_system_prompt,
                summarizer_system_prompt=summarizer_system_prompt,
                summary_prompt=summary_prompt,
                role_prompts=role_prompts,
                context_op=context_op,
                question_ops=question_ops,
                server_config=self.server_config,
                generation_config=generation_config,
            )
        else:
            map_reduce_agent = self.map_reduce_agent
            # Replace LLM ops' generation config
            for op in map_reduce_agent.graph.iter_ops(ops.LLMOp):
                op.config = generation_config or GenerationConfig.from_env()

        inputs = {
            op.name: context_questions
            for op, context_questions in zip(question_ops, context_question_prompts)
        }
        if context_op is not None:
            assert context_prompts is not None
            inputs[context_op.name] = context_prompts

        map_reduce_agent.compile(**inputs)
        return map_reduce_agent

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
    ) -> tuple[OutputType, HeliumSystemProfile]:
        # Prepare inputs
        if context_prompts is not None:
            context_prompts = random_shuffle(context_prompts, inplace=False)
        context_indices = random_shuffle(list(range(len(context_question_prompts[0]))))
        context_question_prompts = tuple(
            random_shuffle(questions, inplace=False)
            for questions in context_question_prompts
        )

        map_reduce_agent = self.create_agent(
            expert_system_prompt=expert_system_prompt,
            summarizer_system_prompt=summarizer_system_prompt,
            summary_prompt=summary_prompt,
            role_prompts=role_prompts,
            context_prompts=context_prompts,
            context_question_prompts=context_question_prompts,
            num_agents=num_agents,
            generation_config=generation_config,
        )

        self.start_timer("generate")
        response = await map_reduce_agent.run_async(self.request_config)
        self.stop_timer()

        output_builder = self.OutputBuilder()
        for question_idx in range(len(context_question_prompts)):
            answers = response.outputs[f"question_{question_idx}"]
            for context_idx, answer in zip(context_indices, answers):
                output_builder.add(context_idx, question_idx, answer)

        return output_builder.build(), response.system_profile

    async def _precompute(
        self,
        expert_system_prompt: str,
        summarizer_system_prompt: str,
        summary_prompt: str,
        role_prompts: list[str] | None,
        context_prompts: list[str] | None,
        context_question_prompts: tuple[list[str], ...],
        num_agents: int,
        generation_config: GenerationConfig | None,
        precompute_mode: Literal["none", "only", "both"],
        **kwargs,
    ) -> HeliumResponse:
        # Prepare inputs
        if context_prompts is not None:
            context_prompts = random_shuffle(context_prompts, inplace=False)
        context_question_prompts = tuple(
            random_shuffle(questions, inplace=False)
            for questions in context_question_prompts
        )

        map_reduce_agent = self.create_agent(
            expert_system_prompt=expert_system_prompt,
            summarizer_system_prompt=summarizer_system_prompt,
            summary_prompt=summary_prompt,
            role_prompts=role_prompts,
            context_prompts=context_prompts,
            context_question_prompts=context_question_prompts,
            num_agents=num_agents,
            generation_config=generation_config,
        )

        request_config = (
            HeliumRequestConfig()
            if self.request_config is None
            else self.request_config.model_copy()
        )
        request_config.precompute_mode = precompute_mode
        return await map_reduce_agent.run_async(request_config)
