"""
Adapted from https://github.com/composable-models/llm_multiagent_debate/
"""

from collections.abc import Sequence
from typing import Literal

from bench_programs.debate.base import DebateProgram, OutputType
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


class DebateAgent(Agent):
    def __init__(
        self,
        num_agents: int,
        num_rounds: int,
        system_prompt: str,
        role_prompts: Sequence[str] | None,
        revise_prompts: tuple[str, str],
        context_op: ops.Op | None,
        question_ops: Sequence[ops.Op],
        server_config: HeliumServerConfig | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        if num_agents <= 0:
            raise ValueError(f"num_agents must be positive, got {num_agents}")
        if num_rounds <= 0:
            raise ValueError(f"num_rounds must be positive, got {num_rounds}")
        super().__init__(
            server_config=server_config,
            num_agents=num_agents,
            num_rounds=num_rounds,
            system_prompt=system_prompt,
            role_prompts=role_prompts,
            revise_prompts=revise_prompts,
            context_op=context_op,
            question_ops=question_ops,
            generation_config=generation_config,
        )

    def build_ops(
        self,
        num_agents: int,
        num_rounds: int,
        system_prompt: str,
        role_prompts: Sequence[str] | None,
        revise_prompts: tuple[str, str],
        context_op: ops.Op | None,
        question_ops: Sequence[ops.Op],
        generation_config: GenerationConfig | None = None,
    ) -> list[ops.OutputOp]:
        # Build user prompt format str
        user_prompt_fmts: list[str] | None
        if role_prompts is None and context_op is None:
            user_prompt_fmts = None
        else:
            context_fmt = None if context_op is None else "{context}"
            maybe_role_prompts: Sequence[str | None]
            if role_prompts is None:
                maybe_role_prompts = [None] * num_agents
            else:
                maybe_role_prompts = role_prompts
            user_prompt_fmts = [
                DebateProgram.build_user_prompt(role_prompt, context_fmt, "{question}")
                for role_prompt in maybe_role_prompts
            ]

        output_ops: list[ops.OutputOp] = []
        for question_i, question_op in enumerate(question_ops):
            # Create user prompt ops
            if user_prompt_fmts is None:
                context_question_ops = [question_op for _ in range(num_agents)]
            else:
                format_kwargs = {"question": question_op}
                if context_op is not None:
                    format_kwargs["context"] = context_op
                context_question_ops = [
                    ops.format_op(user_prompt_fmt, **format_kwargs)
                    for user_prompt_fmt in user_prompt_fmts
                ]

            # First round
            initial_message_list = [
                ops.message_data(
                    [
                        ops.OpMessage(role="system", content=system_prompt),
                        ops.OpMessage(role="user", content=context_question_op),
                    ]
                )
                for context_question_op in context_question_ops
            ]
            history_list = [
                ops.llm_chat(message, generation_config, return_history=True)
                for message in initial_message_list
            ]

            if num_rounds == 1:
                output_ops.extend(
                    [
                        ops.as_output(f"question_{question_i}_agent_{agent_i}", history)
                        for agent_i, history in enumerate(history_list)
                    ]
                )
                continue

            # Debate rounds
            revise_prompt: ops.Op
            if num_agents == 1:
                revise_prompt = ops.data(revise_prompts[0])
                new_convo_list = [
                    ops.append_message(history, revise_prompt)
                    for history in history_list
                ]
            else:
                last_message_list = [
                    ops.get_last_message(history) for history in history_list
                ]
                new_convo_list = []
                for i, history in enumerate(history_list):
                    other_agent_answers = (
                        last_message_list[:i] + last_message_list[i + 1 :]
                    )
                    revise_prompt = ops.format_op(
                        "\n\n ".join(
                            [
                                "These are the solutions to the problem from other agents: ",
                                *[
                                    f"One agent solution: ```{{agent_{j}}}```"
                                    for j in range(num_agents - 1)
                                ],
                                revise_prompts[1],
                            ]
                        ),
                        **{
                            f"agent_{j}": ans
                            for j, ans in enumerate(other_agent_answers)
                        },
                    )
                    new_convo_list.append(ops.append_message(history, revise_prompt))
            revised_history_list = [
                ops.llm_chat(convo, generation_config, return_history=True)
                for convo in new_convo_list
            ]
            debate_loop = ops.loop(history_list, revised_history_list, num_rounds - 1)

            output_ops.extend(
                [
                    ops.as_output(
                        f"question_{question_i}_agent_{agent_i}", agent_history
                    )
                    for agent_i, agent_history in enumerate(debate_loop)
                ]
            )

        return output_ops


class HeliumDebateProgram(DebateProgram, HeliumProgram):
    def __init__(
        self,
        request_config: HeliumRequestConfig | None = None,
        server_config: HeliumServerConfig | None = None,
        debate_agent: DebateAgent | None = None,
    ) -> None:
        DebateProgram.__init__(self)
        HeliumProgram.__init__(self, server_config=server_config)
        self.debate_agent = debate_agent
        self.request_config = request_config

    def create_agent(
        self,
        system_prompt: str,
        role_prompts: list[str] | None,
        context_prompts: list[str] | None,
        context_question_prompts: tuple[list[str], ...],
        revise_prompts: tuple[str, str],
        num_agents: int,
        num_rounds: int,
        generation_config: GenerationConfig | None,
        **_,
    ) -> DebateAgent:
        context_op = None if context_prompts is None else ops.InputOp("context")
        question_ops = [
            ops.InputOp(f"question_{i}") for i in range(len(context_question_prompts))
        ]

        if self.debate_agent is None:
            debate_agent = DebateAgent(
                num_agents=num_agents,
                num_rounds=num_rounds,
                system_prompt=system_prompt,
                role_prompts=role_prompts,
                revise_prompts=revise_prompts,
                context_op=context_op,
                question_ops=question_ops,
                server_config=self.server_config,
                generation_config=generation_config,
            )
        else:
            debate_agent = self.debate_agent
            # Replace LLM ops' generation config
            for op in debate_agent.graph.iter_ops(ops.LLMOp):
                op.config = generation_config or GenerationConfig.from_env()

        inputs = {
            op.name: context_questions
            for op, context_questions in zip(question_ops, context_question_prompts)
        }
        if context_op is not None:
            assert context_prompts is not None
            inputs[context_op.name] = context_prompts

        debate_agent.compile(**inputs)
        return debate_agent

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
    ) -> tuple[OutputType, HeliumSystemProfile]:
        # Prepare inputs
        if context_prompts is not None:
            context_prompts = random_shuffle(context_prompts, inplace=False)
        context_indices = random_shuffle(list(range(len(context_question_prompts[0]))))
        context_question_prompts = tuple(
            random_shuffle(questions, inplace=False)
            for questions in context_question_prompts
        )

        debate_agent = self.create_agent(
            system_prompt=system_prompt,
            role_prompts=role_prompts,
            context_prompts=context_prompts,
            context_question_prompts=context_question_prompts,
            revise_prompts=revise_prompts,
            num_agents=num_agents,
            num_rounds=num_rounds,
            generation_config=generation_config,
        )

        self.start_timer("generate")
        response = await debate_agent.run_async(self.request_config)
        self.stop_timer()

        output_builder = self.OutputBuilder()
        for question_idx in range(len(context_question_prompts)):
            for agent_idx in range(num_agents):
                output_name = f"question_{question_idx}_agent_{agent_idx}"
                agent_responses = response.outputs[output_name]
                for context_idx, context_responses in zip(
                    context_indices, agent_responses
                ):
                    output_builder.add(
                        context_idx,
                        question_idx,
                        agent_idx,
                        context_responses[-1]["content"],
                    )

        return output_builder.build(), response.system_profile

    async def _precompute(
        self,
        system_prompt: str,
        role_prompts: list[str] | None,
        context_prompts: list[str] | None,
        context_question_prompts: tuple[list[str], ...],
        revise_prompts: tuple[str, str],
        num_agents: int,
        num_rounds: int,
        generation_config: GenerationConfig | None,
        precompute_mode: Literal["none", "only", "both"],
        dump_conversations: bool = False,
        **kwargs,
    ) -> HeliumResponse:
        # Prepare inputs
        if context_prompts is not None:
            context_prompts = random_shuffle(context_prompts, inplace=False)
        context_question_prompts = tuple(
            random_shuffle(questions, inplace=False)
            for questions in context_question_prompts
        )

        debate_agent = self.create_agent(
            system_prompt=system_prompt,
            role_prompts=role_prompts,
            context_prompts=context_prompts,
            context_question_prompts=context_question_prompts,
            revise_prompts=revise_prompts,
            num_agents=num_agents,
            num_rounds=num_rounds,
            generation_config=generation_config,
        )

        request_config = (
            HeliumRequestConfig()
            if self.request_config is None
            else self.request_config.model_copy()
        )
        request_config.precompute_mode = precompute_mode
        return await debate_agent.run_async(request_config)
