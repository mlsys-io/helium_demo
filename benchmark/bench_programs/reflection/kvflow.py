from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Any, cast

from bench_programs.reflection.base import ReflectionProgram
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


class AgentContextState(TypedDict):
    context_idx: int
    question_idx: int
    context: str
    question: str
    financial_analyst_context: list[BaseMessage]
    extraction_critic_context: list[BaseMessage]
    calculation_critic_context: list[BaseMessage]


@dataclass
class AgentIds:
    financial_analyst_agent_id: str
    extraction_critic_agent_id: str
    calculation_critic_agent_id: str
    final_answer_agent_id: str

    def to_tuple(self) -> tuple[str, str, str, str]:
        return (
            self.financial_analyst_agent_id,
            self.extraction_critic_agent_id,
            self.calculation_critic_agent_id,
            self.final_answer_agent_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class KVFlowReflectionProgram(ReflectionProgram):
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
            update_agent_step_graph=update_agent_step_graph
        )

        self.start_timer("precompute")
        assert get_worker_generation_configs is not None
        await self._precompute_static_prompts(
            agent_ids=agent_ids,
            system_prompts=system_prompts,
            generation_configs=get_worker_generation_configs(generation_config),
        )
        self.stop_timer()

        workflow = (
            await self._build_workflow(
                agent_ids=agent_ids,
                updater=step_graph_updater,
                financial_analyst_fmt=financial_analyst_fmt,
                extraction_critic_fmt=extraction_critic_fmt,
                calculation_critic_fmt=calculation_critic_fmt,
                final_answer_fmt=final_answer_fmt,
                generation_config=generation_config,
            )
        ).compile()

        flattened = self.flatten_inputs(contexts, context_questions)
        inputs: list[AgentContextState] = [
            {
                "context_idx": context_idx,
                "question_idx": question_idx,
                "context": context,
                "question": question,
                "financial_analyst_context": [SystemMessage(content=system_prompts[0])],
                "extraction_critic_context": [SystemMessage(content=system_prompts[1])],
                "calculation_critic_context": [
                    SystemMessage(content=system_prompts[2])
                ],
            }
            for context_idx, question_idx, context, question in flattened
        ]

        await step_graph_updater.reset(total_items=len(inputs))
        await start_benchmark()

        self.start_timer("generate")
        outputs: list[dict[str, Any]] = await workflow.abatch(inputs)  # type: ignore[arg-type]
        self.stop_timer()

        system_profile = await stop_benchmark()

        output_builder = self.OutputBuilder()
        for output in outputs:
            context_idx = output["context_idx"]
            question_idx = output["question_idx"]
            answer = output["financial_analyst_context"][-1].content
            output_builder.add(context_idx, question_idx, cast(str, answer))

        return output_builder.build(), system_profile

    async def _build_workflow(
        self,
        agent_ids: AgentIds,
        updater: KVFlowStepGraphUpdater,
        financial_analyst_fmt: str,
        extraction_critic_fmt: str,
        calculation_critic_fmt: str,
        final_answer_fmt: str,
        generation_config: GenerationConfig,
    ) -> StateGraph:
        (
            financial_analyst_agent_id,
            extraction_critic_agent_id,
            calculation_critic_agent_id,
            final_answer_agent_id,
        ) = agent_ids.to_tuple()

        llm = get_langgraph_openai_client(generation_config)

        async def financial_analyst(state: AgentContextState) -> dict[str, Any]:
            async with updater.node(financial_analyst_agent_id):
                context = state["context"]
                question = state["question"]
                fa_context = state["financial_analyst_context"]
                user_message = HumanMessage(
                    content=financial_analyst_fmt.format(
                        context=context, question=question
                    )
                )
                messages = fa_context + [user_message]
                response = await llm_ainvoke(llm, messages, financial_analyst_agent_id)
                messages.append(response)
                return {"financial_analyst_context": messages}

        async def extraction_critic(state: AgentContextState) -> dict[str, Any]:
            async with updater.node(extraction_critic_agent_id):
                context = state["context"]
                question = state["question"]
                answer = state["financial_analyst_context"][-1].content
                user_message = HumanMessage(
                    content=extraction_critic_fmt.format(
                        context=context, question=question, response=answer
                    )
                )
                messages = state["extraction_critic_context"] + [user_message]
                response = await llm_ainvoke(llm, messages, extraction_critic_agent_id)
                messages.append(response)
                return {"extraction_critic_context": messages}

        async def calculation_critic(state: AgentContextState) -> dict[str, Any]:
            async with updater.node(calculation_critic_agent_id):
                context = state["context"]
                question = state["question"]
                answer = state["financial_analyst_context"][-1].content
                critic = state["extraction_critic_context"][-1].content
                user_message = HumanMessage(
                    content=calculation_critic_fmt.format(
                        context=context,
                        question=question,
                        response=answer,
                        critic=critic,
                    )
                )
                messages = state["calculation_critic_context"] + [user_message]
                response = await llm_ainvoke(llm, messages, calculation_critic_agent_id)
                messages.append(response)
                return {"calculation_critic_context": messages}

        async def final_answer(state: AgentContextState) -> dict[str, Any]:
            async with updater.node(final_answer_agent_id):
                fa_context = state["financial_analyst_context"]
                extraction_critic_text = state["extraction_critic_context"][-1].content
                calculation_critic_text = state["calculation_critic_context"][
                    -1
                ].content
                user_message = HumanMessage(
                    content=final_answer_fmt.format(
                        extraction_critic=extraction_critic_text,
                        calculation_critic=calculation_critic_text,
                    )
                )
                messages = fa_context + [user_message]
                response = await llm_ainvoke(llm, messages, final_answer_agent_id)
                messages.append(response)
                return {"financial_analyst_context": messages}

        workflow_builder = StateGraph(AgentContextState)
        workflow_builder.add_node(financial_analyst_agent_id, financial_analyst)
        workflow_builder.add_node(extraction_critic_agent_id, extraction_critic)
        workflow_builder.add_node(calculation_critic_agent_id, calculation_critic)
        workflow_builder.add_node(final_answer_agent_id, final_answer)

        workflow_builder.add_edge(START, financial_analyst_agent_id)
        workflow_builder.add_edge(
            financial_analyst_agent_id, extraction_critic_agent_id
        )
        workflow_builder.add_edge(
            extraction_critic_agent_id, calculation_critic_agent_id
        )
        workflow_builder.add_edge(calculation_critic_agent_id, final_answer_agent_id)
        return workflow_builder

    def _build_agent_ids(self) -> AgentIds:
        return AgentIds(
            financial_analyst_agent_id="financial_analyst",
            extraction_critic_agent_id="extraction_critic",
            calculation_critic_agent_id="calculation_critic",
            final_answer_agent_id="final_answer",
        )

    async def _prepare_agent_step_graph(
        self,
        update_agent_step_graph: Callable[
            [dict[str, Any], dict[int, list[str]], int], Awaitable[None]
        ],
    ) -> tuple[KVFlowStepGraphUpdater, AgentIds]:
        agent_ids = self._build_agent_ids()
        step_graph = self._build_agent_step_graph(**agent_ids.to_dict())
        updater = KVFlowStepGraphUpdater(
            step_graph=step_graph,
            send_update=update_agent_step_graph,
        )
        return updater, agent_ids

    def _build_agent_step_graph(
        self,
        financial_analyst_agent_id: str,
        extraction_critic_agent_id: str,
        calculation_critic_agent_id: str,
        final_answer_agent_id: str,
    ) -> KVFlowStepGraph:
        edges: dict[str, set[str]] = {
            financial_analyst_agent_id: {extraction_critic_agent_id},
            extraction_critic_agent_id: {calculation_critic_agent_id},
            calculation_critic_agent_id: {final_answer_agent_id},
            final_answer_agent_id: set(),
        }
        return KVFlowStepGraph(
            edges=edges,
            all_agents={
                financial_analyst_agent_id,
                extraction_critic_agent_id,
                calculation_critic_agent_id,
                final_answer_agent_id,
            },
        )

    def _build_static_prompts(
        self,
        agent_ids: AgentIds,
        system_prompts: tuple[str, str, str],
    ) -> dict[str, list[BaseMessage]]:
        (
            financial_analyst_agent_id,
            extraction_critic_agent_id,
            calculation_critic_agent_id,
            final_answer_agent_id,
        ) = agent_ids.to_tuple()
        static: dict[str, list[BaseMessage]] = {
            financial_analyst_agent_id: [SystemMessage(content=system_prompts[0])],
            extraction_critic_agent_id: [SystemMessage(content=system_prompts[1])],
            calculation_critic_agent_id: [SystemMessage(content=system_prompts[2])],
            # final answer reuses the financial analyst system prompt since it writes into
            # `financial_analyst_context`.
            final_answer_agent_id: [SystemMessage(content=system_prompts[0])],
        }
        return static

    async def _precompute_static_prompts(
        self,
        agent_ids: AgentIds,
        system_prompts: tuple[str, str, str],
        generation_configs: list[GenerationConfig],
    ) -> None:
        static_prompts = self._build_static_prompts(
            agent_ids=agent_ids,
            system_prompts=system_prompts,
        )
        await precompute_static_prompts(static_prompts, generation_configs)
