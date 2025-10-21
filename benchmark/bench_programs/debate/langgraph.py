from collections.abc import Sequence
from typing import Annotated, Any, cast

from bench_programs.debate.base import DebateProgram
from bench_programs.utils.common import try_start_benchmark, try_stop_benchmark
from bench_programs.utils.langgraph import get_langgraph_openai_client
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

from helium.common import GenerationConfig
from helium.runtime.protocol import HeliumSystemProfile


def update_agent_context(
    left: list[list[BaseMessage]],
    right: tuple[int, list[BaseMessage]] | list[list[BaseMessage]],
) -> list[list[BaseMessage]]:
    if isinstance(right, list):
        return right
    agent_idx, agent_context = right
    return left[:agent_idx] + [agent_context] + left[agent_idx + 1 :]


class AgentContextsState(TypedDict):
    context_idx: int
    question_idx: int
    agent_contexts: Annotated[list[list[BaseMessage]], update_agent_context]


class LangGraphDebateProgram(DebateProgram):
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
        if generation_config is None:
            generation_config = GenerationConfig.from_env()

        base_url = generation_config.base_url

        workflow = self._build_workflow(
            revise_prompts, num_agents, num_rounds, generation_config
        ).compile()

        maybe_role_prompts: Sequence[str | None]
        if role_prompts is None:
            maybe_role_prompts = [None] * num_agents
        else:
            maybe_role_prompts = role_prompts

        # Prepare inputs
        flattened = self.flatten_inputs(context_prompts, context_question_prompts)
        inputs = [
            {
                "context_idx": context_idx,
                "question_idx": question_idx,
                "agent_contexts": [
                    [
                        SystemMessage(system_prompt),
                        HumanMessage(
                            self.build_user_prompt(
                                role_prompt, context_prompt, question_prompt
                            )
                        ),
                    ]
                    for role_prompt in maybe_role_prompts
                ],
            }
            for context_idx, question_idx, context_prompt, question_prompt in flattened
        ]

        # Start benchmarking
        try_start_benchmark(base_url)

        self.start_timer("generate")
        outputs: list[dict[str, Any]] = await workflow.abatch(inputs)
        self.stop_timer()

        # Stop benchmarking
        system_profile = try_stop_benchmark(base_url)

        output_builder = self.OutputBuilder()
        for output in outputs:
            context_idx = output["context_idx"]
            question_idx = output["question_idx"]
            agent_context: list[BaseMessage]
            for agent_idx, agent_context in enumerate(output["agent_contexts"]):
                output_builder.add(
                    context_idx,
                    question_idx,
                    agent_idx,
                    cast(str, agent_context[-1].content),
                )

        return output_builder.build(), system_profile

    def _get_revise_message(
        self,
        other_agent_contexts: list[list[BaseMessage]],
        revise_prompts: tuple[str, str],
    ) -> str:
        if len(other_agent_contexts) == 0:
            return revise_prompts[0]
        return "\n\n ".join(
            [
                "These are the solutions to the problem from other agents: ",
                *[
                    f"One agent solution: ```{other_context[-1].content}```"
                    for other_context in other_agent_contexts
                ],
                revise_prompts[1],
            ]
        )

    def _build_workflow(
        self,
        revise_prompts: tuple[str, str],
        num_agents: int,
        num_rounds: int,
        generation_config: GenerationConfig,
    ) -> StateGraph:
        llm = get_langgraph_openai_client(generation_config)

        def agent_answer(round: int, agent_idx: int):
            async def answer_question(
                state: AgentContextsState,
            ) -> dict[str, Any]:
                """Answer the question using the LLM."""
                agent_contexts = state["agent_contexts"]
                agent_context = agent_contexts[agent_idx].copy()

                if round > 0:
                    other_agent_contexts = (
                        agent_contexts[:agent_idx] + agent_contexts[agent_idx + 1 :]
                    )
                    new_message_content = self._get_revise_message(
                        other_agent_contexts, revise_prompts
                    )
                    agent_context.append(HumanMessage(new_message_content))
                response = await llm.ainvoke(agent_context)
                agent_context.append(response)
                return {"agent_contexts": (agent_idx, agent_context)}

            return answer_question

        workflow_builder = StateGraph(AgentContextsState)

        next_round_nodes: list[str] = []
        for round in range(num_rounds):
            prev_round_nodes = next_round_nodes
            next_round_nodes = []
            for agent_idx in range(num_agents):
                node_name = f"agent-{agent_idx}-{round}"
                next_round_nodes.append(node_name)
                workflow_builder.add_node(node_name, agent_answer(round, agent_idx))
                if len(prev_round_nodes) == 0:
                    workflow_builder.add_edge(START, node_name)
                else:
                    for prev_node in prev_round_nodes:
                        workflow_builder.add_edge(prev_node, node_name)

        return workflow_builder
