from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Annotated, Any, cast

from bench_programs.map_reduce.base import MapReduceProgram
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
    expert_contexts: Annotated[list[list[BaseMessage]], update_agent_context]
    summarizer_context: list[BaseMessage]


@dataclass
class AgentIds:
    expert_agent_ids: list[str]
    summarizer_agent_id: str

    def to_tuple(self) -> tuple[list[str], str]:
        return self.expert_agent_ids, self.summarizer_agent_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class KVFlowMapReduceProgram(MapReduceProgram):
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
    ) -> tuple[list[tuple[str, ...]], HeliumSystemProfile]:
        assert start_benchmark is not None
        assert stop_benchmark is not None
        assert update_agent_step_graph is not None

        if generation_config is None:
            generation_config = GenerationConfig.from_env()

        step_graph_updater, agent_ids = await self._prepare_agent_step_graph(
            num_agents=num_agents,
            update_agent_step_graph=update_agent_step_graph,
        )

        # Precompute static prompts
        self.start_timer("precompute")
        assert get_worker_generation_configs is not None
        await self._precompute_static_prompts(
            agent_ids=agent_ids,
            expert_system_prompt=expert_system_prompt,
            summarizer_system_prompt=summarizer_system_prompt,
            generation_configs=get_worker_generation_configs(generation_config),
        )
        self.stop_timer()

        workflow = (
            await self._build_workflow(
                agent_ids=agent_ids,
                updater=step_graph_updater,
                expert_system_prompt=expert_system_prompt,
                summarizer_system_prompt=summarizer_system_prompt,
                summary_prompt=summary_prompt,
                role_prompts=role_prompts,
                num_agents=num_agents,
                generation_config=generation_config,
            )
        ).compile()

        flattened = self.flatten_inputs(context_prompts, context_question_prompts)
        inputs: list[AgentContextsState] = [
            {
                "context_idx": context_idx,
                "question_idx": question_idx,
                "context": context_prompt,
                "question": question_prompt,
                "expert_contexts": [
                    [SystemMessage(expert_system_prompt)] for _ in range(num_agents)
                ],
                "summarizer_context": [SystemMessage(summarizer_system_prompt)],
            }
            for context_idx, question_idx, context_prompt, question_prompt in flattened
        ]

        # Start benchmarking
        await step_graph_updater.reset(total_items=len(inputs))
        await start_benchmark()

        self.start_timer("generate")
        outputs: list[dict[str, Any]] = await workflow.abatch(inputs)  # type: ignore[arg-type]
        self.stop_timer()

        # Stop benchmarking
        system_profile = await stop_benchmark()

        output_builder = self.OutputBuilder()
        for output in outputs:
            context_idx = output["context_idx"]
            question_idx = output["question_idx"]
            out = output["summarizer_context"][-1].content
            output_builder.add(context_idx, question_idx, cast(str, out))

        return output_builder.build(), system_profile

    async def _build_workflow(
        self,
        agent_ids: AgentIds,
        updater: KVFlowStepGraphUpdater,
        expert_system_prompt: str,
        summarizer_system_prompt: str,
        summary_prompt: str,
        role_prompts: list[str] | None,
        num_agents: int,
        generation_config: GenerationConfig,
    ) -> StateGraph:
        expert_agent_ids, summarizer_agent_id = agent_ids.to_tuple()
        llm = get_langgraph_openai_client(generation_config)

        def expert_answer(agent_id: str, agent_idx: int):
            async def answer_question(state: AgentContextsState) -> dict[str, Any]:
                async with updater.node(agent_id):
                    role_prompt = role_prompts[agent_idx] if role_prompts else None
                    expert_user_prompt = self.build_user_prompt(
                        role_prompt, state["context"], state["question"]
                    )
                    expert_context = state["expert_contexts"][agent_idx] + [
                        HumanMessage(expert_user_prompt)
                    ]
                    response = await llm_ainvoke(llm, expert_context, agent_id)
                    expert_context.append(response)
                    return {"expert_contexts": (agent_idx, expert_context)}

            return answer_question

        async def summarize_answers(state: AgentContextsState) -> dict[str, Any]:
            async with updater.node(summarizer_agent_id):
                expert_contexts = state["expert_contexts"]
                expert_answers = [
                    cast(str, expert_context[-1].content)
                    for expert_context in expert_contexts
                ]

                summarizer_user_prompt = summary_prompt
                context = state["context"]
                if context is not None:
                    summarizer_user_prompt += f"\n\nContext: {context}"
                summarizer_user_prompt += f"\n\nQuestion: {state['question']}\n"
                for agent_i, expert_answer in enumerate(expert_answers):
                    summarizer_user_prompt += (
                        f"\nExpert {agent_i + 1} Answer:\n{expert_answer}"
                    )

                summarizer_context = state["summarizer_context"] + [
                    HumanMessage(summarizer_user_prompt)
                ]
                response = await llm_ainvoke(
                    llm, summarizer_context, summarizer_agent_id
                )
                summarizer_context.append(response)
                return {"summarizer_context": summarizer_context}

        workflow_builder = StateGraph(AgentContextsState)
        workflow_builder.add_node(summarizer_agent_id, summarize_answers)
        for agent_idx in range(num_agents):
            agent_name = expert_agent_ids[agent_idx]
            workflow_builder.add_node(agent_name, expert_answer(agent_name, agent_idx))
            workflow_builder.add_edge(START, agent_name)
            workflow_builder.add_edge(agent_name, summarizer_agent_id)
        return workflow_builder

    def _build_static_prompts(
        self,
        agent_ids: AgentIds,
        expert_system_prompt: str,
        summarizer_system_prompt: str,
    ) -> dict[str, list[BaseMessage]]:
        expert_agent_ids, summarizer_agent_id = agent_ids.to_tuple()
        static_prompts: dict[str, list[BaseMessage]] = {
            agent_id: [SystemMessage(expert_system_prompt)]
            for agent_id in expert_agent_ids
        }
        static_prompts[summarizer_agent_id] = [SystemMessage(summarizer_system_prompt)]
        return static_prompts

    async def _precompute_static_prompts(
        self,
        agent_ids: AgentIds,
        expert_system_prompt: str,
        summarizer_system_prompt: str,
        generation_configs: list[GenerationConfig],
    ) -> None:
        static_prompts = self._build_static_prompts(
            agent_ids=agent_ids,
            expert_system_prompt=expert_system_prompt,
            summarizer_system_prompt=summarizer_system_prompt,
        )
        await precompute_static_prompts(static_prompts, generation_configs)

    def _build_agent_ids(self, num_agents: int) -> AgentIds:
        expert_agent_ids = [f"expert_{i}" for i in range(num_agents)]
        summarizer_agent_id = "summarizer"
        return AgentIds(
            expert_agent_ids=expert_agent_ids,
            summarizer_agent_id=summarizer_agent_id,
        )

    async def _prepare_agent_step_graph(
        self,
        num_agents: int,
        update_agent_step_graph: Callable[
            [dict[str, Any], dict[int, list[str]], int], Awaitable[None]
        ],
    ) -> tuple[KVFlowStepGraphUpdater, AgentIds]:
        agent_ids = self._build_agent_ids(num_agents=num_agents)
        step_graph = self._build_agent_step_graph(**agent_ids.to_dict())
        updater: KVFlowStepGraphUpdater = KVFlowStepGraphUpdater(
            step_graph=step_graph,
            send_update=update_agent_step_graph,
        )
        return updater, agent_ids

    def _build_agent_step_graph(
        self,
        expert_agent_ids: list[str],
        summarizer_agent_id: str,
    ) -> KVFlowStepGraph:
        edges: dict[str, set[str]] = {a: set() for a in expert_agent_ids}
        for expert_id in expert_agent_ids:
            edges.setdefault(expert_id, set()).add(summarizer_agent_id)
        return KVFlowStepGraph(
            edges=edges,
            all_agents={*expert_agent_ids, summarizer_agent_id},
        )
