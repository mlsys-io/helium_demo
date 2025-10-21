from typing import Annotated, Any

from bench_programs.map_reduce.base import MapReduceProgram
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
    context: str | None
    question: str
    expert_contexts: Annotated[list[list[BaseMessage]], update_agent_context]
    summarizer_context: list[BaseMessage]


class LangGraphMapReduceProgram(MapReduceProgram):
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
    ) -> tuple[list[tuple[str, ...]], HeliumSystemProfile]:
        if generation_config is None:
            generation_config = GenerationConfig.from_env()

        base_url = generation_config.base_url

        workflow = self._build_workflow(
            summary_prompt, role_prompts, num_agents, generation_config
        ).compile()

        # Prepare inputs
        flattened = self.flatten_inputs(context_prompts, context_question_prompts)
        inputs = [
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
        try_start_benchmark(base_url)

        self.start_timer("generate")
        outputs: list[dict[str, Any]] = await workflow.abatch(inputs)
        self.stop_timer()

        # Stop benchmarking
        system_profile = try_stop_benchmark(base_url)

        output_builder = self.OutputBuilder()
        for output in outputs:
            output_builder.add(
                output["context_idx"],
                output["question_idx"],
                output["summarizer_context"][-1].content,
            )

        return output_builder.build(), system_profile

    def _build_workflow(
        self,
        summary_prompt: str,
        role_prompts: list[str] | None,
        num_agents: int,
        generation_config: GenerationConfig,
    ) -> StateGraph:
        llm = get_langgraph_openai_client(generation_config)

        def expert_answer(agent_idx: int):
            async def answer_question(state: AgentContextsState) -> dict[str, Any]:
                role_prompt = role_prompts[agent_idx] if role_prompts else None
                expert_user_prompt = self.build_user_prompt(
                    role_prompt, state["context"], state["question"]
                )
                expert_context = state["expert_contexts"][agent_idx] + [
                    HumanMessage(expert_user_prompt)
                ]
                response = await llm.ainvoke(expert_context)
                expert_context.append(response)
                return {"expert_contexts": (agent_idx, expert_context)}

            return answer_question

        async def summarize_answers(state: AgentContextsState) -> dict[str, Any]:
            expert_contexts = state["expert_contexts"]
            expert_answers = [
                expert_context[-1].content for expert_context in expert_contexts
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
            response = await llm.ainvoke(summarizer_context)
            summarizer_context.append(response)
            return {"summarizer_context": summarizer_context}

        workflow_builder = StateGraph(AgentContextsState)

        workflow_builder.add_node("summarizer", summarize_answers)
        for agent_idx in range(num_agents):
            agent_name = f"expert_{agent_idx}"
            workflow_builder.add_node(agent_name, expert_answer(agent_idx))
            workflow_builder.add_edge(START, agent_name)
            workflow_builder.add_edge(agent_name, "summarizer")

        return workflow_builder
