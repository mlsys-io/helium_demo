from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Annotated, Any, cast

from bench_programs.debate.base import DebateProgram
from bench_programs.utils.kvflow import (
    KVFlowStepGraph,
    KVFlowStepGraphUpdater,
    llm_ainvoke,
    precompute_static_prompts,
)
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
    context: str | None
    question: str
    agent_contexts: Annotated[list[list[BaseMessage]], update_agent_context]


@dataclass
class AgentIds:
    # round -> agent_idx -> agent_id
    agent_ids_by_round: list[list[str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class KVFlowDebateProgram(DebateProgram):
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
        start_benchmark: Callable[[], Awaitable[None]] | None = None,
        stop_benchmark: Callable[[], Awaitable[HeliumSystemProfile]] | None = None,
        update_agent_step_graph: (
            Callable[[dict[str, Any], dict[int, list[str]], int], Awaitable[None]]
            | None
        ) = None,
        get_worker_generation_configs: (
            Callable[[GenerationConfig], list[GenerationConfig]] | None
        ) = None,
        **kwargs,
    ) -> tuple[list[tuple[list[str], ...]], HeliumSystemProfile]:
        assert start_benchmark is not None
        assert stop_benchmark is not None
        assert update_agent_step_graph is not None

        if generation_config is None:
            generation_config = GenerationConfig.from_env()

        step_graph_updater, agent_ids = await self._prepare_agent_step_graph(
            num_agents=num_agents,
            num_rounds=num_rounds,
            update_agent_step_graph=update_agent_step_graph,
        )

        # Precompute static prompts
        self.start_timer("precompute")
        assert get_worker_generation_configs is not None
        await self._precompute_static_prompts(
            agent_ids=agent_ids,
            system_prompt=system_prompt,
            generation_configs=get_worker_generation_configs(generation_config),
        )
        self.stop_timer()

        workflow = (
            await self._build_workflow(
                agent_ids=agent_ids,
                updater=step_graph_updater,
                revise_prompts=revise_prompts,
                num_agents=num_agents,
                num_rounds=num_rounds,
                generation_config=generation_config,
            )
        ).compile()

        flattened = self.flatten_inputs(context_prompts, context_question_prompts)

        if role_prompts is None:
            maybe_role_prompts: list[str | None] = [None] * num_agents
        else:
            maybe_role_prompts = list(role_prompts)

        inputs: list[AgentContextsState] = [
            {
                "context_idx": context_idx,
                "question_idx": question_idx,
                "context": context_prompt,
                "question": question_prompt,
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

        await start_benchmark()

        self.start_timer("generate")
        await step_graph_updater.reset(total_items=len(inputs))
        outputs: list[dict[str, Any]] = await workflow.abatch(inputs)  # type: ignore[arg-type]
        self.stop_timer()

        system_profile = await stop_benchmark()

        output_builder = self.OutputBuilder()
        for output in outputs:
            context_idx = output["context_idx"]
            question_idx = output["question_idx"]
            agent_ctxs = output["agent_contexts"]
            for agent_idx, agent_ctx in enumerate(agent_ctxs):
                output_builder.add(
                    context_idx,
                    question_idx,
                    agent_idx,
                    cast(str, agent_ctx[-1].content),
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

    async def _build_workflow(
        self,
        agent_ids: AgentIds,
        updater: KVFlowStepGraphUpdater,
        revise_prompts: tuple[str, str],
        num_agents: int,
        num_rounds: int,
        generation_config: GenerationConfig,
    ) -> StateGraph:
        llm = get_langgraph_openai_client(generation_config)

        def agent_answer(round_idx: int, agent_idx: int, node_agent_id: str):
            async def answer_question(state: AgentContextsState) -> dict[str, Any]:
                async with updater.node(node_agent_id):
                    agent_contexts = state["agent_contexts"]
                    agent_context = agent_contexts[agent_idx].copy()

                    if round_idx > 0:
                        other_agent_contexts = (
                            agent_contexts[:agent_idx] + agent_contexts[agent_idx + 1 :]
                        )
                        new_message_content = self._get_revise_message(
                            other_agent_contexts, revise_prompts
                        )
                        agent_context.append(HumanMessage(new_message_content))

                    response = await llm_ainvoke(llm, agent_context, node_agent_id)
                    agent_context.append(response)
                    return {"agent_contexts": (agent_idx, agent_context)}

            return answer_question

        workflow_builder = StateGraph(AgentContextsState)

        next_round_nodes: list[str] = []
        for round_idx in range(num_rounds):
            prev_round_nodes = next_round_nodes
            next_round_nodes = []
            for agent_idx in range(num_agents):
                node_name = agent_ids.agent_ids_by_round[round_idx][agent_idx]
                next_round_nodes.append(node_name)
                workflow_builder.add_node(
                    node_name,
                    agent_answer(round_idx, agent_idx, node_name),
                )
                if len(prev_round_nodes) == 0:
                    workflow_builder.add_edge(START, node_name)
                else:
                    for prev_node in prev_round_nodes:
                        workflow_builder.add_edge(prev_node, node_name)

        return workflow_builder

    def _build_agent_ids(self, num_agents: int, num_rounds: int) -> AgentIds:
        agent_ids_by_round: list[list[str]] = [
            [f"agent-{agent_idx}-{round_idx}" for agent_idx in range(num_agents)]
            for round_idx in range(num_rounds)
        ]
        return AgentIds(agent_ids_by_round=agent_ids_by_round)

    async def _prepare_agent_step_graph(
        self,
        num_agents: int,
        num_rounds: int,
        update_agent_step_graph: Callable[
            [dict[str, Any], dict[int, list[str]], int], Awaitable[None]
        ],
    ) -> tuple[KVFlowStepGraphUpdater, AgentIds]:
        agent_ids = self._build_agent_ids(num_agents=num_agents, num_rounds=num_rounds)
        step_graph = self._build_agent_step_graph(**agent_ids.to_dict())
        updater: KVFlowStepGraphUpdater = KVFlowStepGraphUpdater(
            step_graph=step_graph,
            send_update=update_agent_step_graph,
        )
        return updater, agent_ids

    def _build_agent_step_graph(
        self, agent_ids_by_round: list[list[str]]
    ) -> KVFlowStepGraph:
        all_agents = {a for per_round in agent_ids_by_round for a in per_round}
        edges: dict[str, set[str]] = {a: set() for a in all_agents}
        for ridx in range(len(agent_ids_by_round) - 1):
            for u in agent_ids_by_round[ridx]:
                for v in agent_ids_by_round[ridx + 1]:
                    edges[u].add(v)
        return KVFlowStepGraph(edges=edges, all_agents=all_agents)

    def _build_static_prompts(
        self,
        agent_ids: AgentIds,
        system_prompt: str,
    ) -> dict[str, list[BaseMessage]]:
        static: list[BaseMessage] = [SystemMessage(system_prompt)]
        return {
            agent_id: static
            for per_round in agent_ids.agent_ids_by_round
            for agent_id in per_round
        }

    async def _precompute_static_prompts(
        self,
        agent_ids: AgentIds,
        system_prompt: str,
        generation_configs: list[GenerationConfig],
    ) -> None:
        static_prompts = self._build_static_prompts(
            agent_ids=agent_ids, system_prompt=system_prompt
        )
        await precompute_static_prompts(static_prompts, generation_configs)
