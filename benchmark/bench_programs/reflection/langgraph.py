from typing import Any, TypedDict

from bench_programs.reflection.base import ReflectionProgram
from bench_programs.utils.common import try_start_benchmark, try_stop_benchmark
from bench_programs.utils.langgraph import get_langgraph_openai_client
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph

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


class LangGraphReflectionProgram(ReflectionProgram):
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
        if generation_config is None:
            generation_config = GenerationConfig.from_env()

        base_url = generation_config.base_url

        workflow = self._build_workflow(
            financial_analyst_fmt,
            extraction_critic_fmt,
            calculation_critic_fmt,
            final_answer_fmt,
            generation_config,
        ).compile()

        # Prepare inputs
        flattened = self.flatten_inputs(contexts, context_questions)
        inputs = [
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
            answer = output["financial_analyst_context"][-1].content
            output_builder.add(context_idx, question_idx, answer)

        return output_builder.build(), system_profile

    def _build_workflow(
        self,
        financial_analyst_fmt: str,
        extraction_critic_fmt: str,
        calculation_critic_fmt: str,
        final_answer_fmt: str,
        generation_config: GenerationConfig,
    ) -> StateGraph:
        llm = get_langgraph_openai_client(generation_config)

        async def financial_analyst(
            state: AgentContextState,
        ) -> dict[str, list[BaseMessage]]:
            context = state["context"]
            question = state["question"]
            financial_analyst_context = state["financial_analyst_context"]
            user_message = HumanMessage(
                content=financial_analyst_fmt.format(context=context, question=question)
            )
            messages = financial_analyst_context + [user_message]
            response = await llm.ainvoke(messages)
            messages.append(response)
            return {"financial_analyst_context": messages}

        async def extraction_critic(
            state: AgentContextState,
        ) -> dict[str, list[BaseMessage]]:
            context = state["context"]
            question = state["question"]
            answer = state["financial_analyst_context"][-1].content
            user_message = HumanMessage(
                content=extraction_critic_fmt.format(
                    context=context, question=question, response=answer
                )
            )
            messages = state["extraction_critic_context"] + [user_message]
            response = await llm.ainvoke(messages)
            messages.append(response)
            return {"extraction_critic_context": messages}

        async def calculation_critic(
            state: AgentContextState,
        ) -> dict[str, list[BaseMessage]]:
            context = state["context"]
            question = state["question"]
            answer = state["financial_analyst_context"][-1].content
            critic = state["extraction_critic_context"][-1].content
            user_message = HumanMessage(
                content=calculation_critic_fmt.format(
                    context=context, question=question, response=answer, critic=critic
                )
            )
            messages = state["calculation_critic_context"] + [user_message]
            response = await llm.ainvoke(messages)
            messages.append(response)
            return {"calculation_critic_context": messages}

        async def final_answer(
            state: AgentContextState,
        ) -> dict[str, list[BaseMessage]]:
            financial_analyst_context = state["financial_analyst_context"]
            extraction_critic = state["extraction_critic_context"][-1].content
            calculation_critic = state["calculation_critic_context"][-1].content
            user_message = HumanMessage(
                content=final_answer_fmt.format(
                    extraction_critic=extraction_critic,
                    calculation_critic=calculation_critic,
                )
            )
            messages = financial_analyst_context + [user_message]
            response = await llm.ainvoke(messages)
            messages.append(response)
            return {"financial_analyst_context": messages}

        workflow_builder = StateGraph(AgentContextState)
        workflow_builder.add_node("financial_analyst", financial_analyst)
        workflow_builder.add_node("extraction_critic", extraction_critic)
        workflow_builder.add_node("calculation_critic", calculation_critic)
        workflow_builder.add_node("final_answer", final_answer)

        workflow_builder.add_edge(START, "financial_analyst")
        workflow_builder.add_edge("financial_analyst", "extraction_critic")
        workflow_builder.add_edge("extraction_critic", "calculation_critic")
        workflow_builder.add_edge("calculation_critic", "final_answer")

        return workflow_builder
