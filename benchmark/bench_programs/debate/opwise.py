from collections.abc import Sequence
from typing import Any

from bench_programs.debate.base import DebateProgram
from bench_programs.utils.common import try_start_benchmark, try_stop_benchmark
from bench_programs.utils.openai import openai_generate_async, prepare_openai
from bench_programs.utils.opwise import WorkflowDAG
from openai import AsyncOpenAI

from helium.common import GenerationConfig, Message
from helium.runtime.protocol import HeliumSystemProfile


class OpWiseDebateProgram(DebateProgram):
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
        client, generation_config = prepare_openai(generation_config)
        generation_kwargs = generation_config.openai_kwargs()
        base_url = generation_config.base_url

        maybe_role_prompts: Sequence[str | None]
        if role_prompts is None:
            maybe_role_prompts = [None] * num_agents
        else:
            maybe_role_prompts = role_prompts
        flattened = self.flatten_inputs(context_prompts, context_question_prompts)
        inputs: list[dict] = [
            {
                "context_idx": context_idx,
                "question_idx": question_idx,
                "agent_contexts": [
                    [
                        Message("system", system_prompt),
                        Message(
                            "user",
                            self.build_user_prompt(
                                role_prompt, context_prompt, question_prompt
                            ),
                        ),
                    ]
                    for role_prompt in maybe_role_prompts
                ],
            }
            for context_idx, question_idx, context_prompt, question_prompt in flattened
        ]

        workflow = self._build_workflow(
            client, revise_prompts, num_agents, num_rounds, generation_kwargs
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
            agent_context: list[Message]
            for agent_idx, agent_context in enumerate(state["agent_contexts"]):
                output_builder.add(
                    context_idx, question_idx, agent_idx, agent_context[-1].content
                )

        return output_builder.build(), system_profile

    def _get_revise_message(
        self,
        other_agent_contexts: list[list[Message]],
        revise_prompts: tuple[str, str],
        round: int,
    ) -> str:
        if len(other_agent_contexts) == 0:
            return revise_prompts[0]
        message_idx = 2 * round
        return "\n\n ".join(
            [
                "These are the solutions to the problem from other agents: ",
                *[
                    f"One agent solution: ```{other_context[message_idx].content}```"
                    for other_context in other_agent_contexts
                ],
                revise_prompts[1],
            ]
        )

    def _build_workflow(
        self,
        client: AsyncOpenAI,
        revise_prompts: tuple[str, str],
        num_agents: int,
        num_rounds: int,
        generation_kwargs: dict[str, Any],
    ) -> WorkflowDAG:
        def agent_answer(round: int, agent_idx: int):
            async def answer_question(state: dict) -> dict:
                agent_contexts = state["agent_contexts"]
                agent_context: list[Message] = agent_contexts[agent_idx]

                if round > 0:
                    other_agent_contexts = (
                        agent_contexts[:agent_idx] + agent_contexts[agent_idx + 1 :]
                    )
                    new_message_content = self._get_revise_message(
                        other_agent_contexts, revise_prompts, round
                    )
                    agent_context.append(Message("user", new_message_content))
                answer = await openai_generate_async(
                    client, agent_context, generation_kwargs
                )
                agent_context.append(Message("assistant", answer))
                return state

            return answer_question

        workflow = WorkflowDAG()
        next_round_nodes: list[str] = []
        for round in range(num_rounds):
            prev_round_nodes = next_round_nodes
            next_round_nodes = []
            for agent_idx in range(num_agents):
                node_name = f"agent-{agent_idx}-{round}"
                next_round_nodes.append(node_name)
                workflow.add_node(node_name, agent_answer(round, agent_idx))
                if len(prev_round_nodes) > 0:
                    for prev_node in prev_round_nodes:
                        workflow.add_edge(prev_node, node_name)

        return workflow
