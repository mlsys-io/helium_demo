from typing import Any, cast

from bench_programs.map_reduce.base import MapReduceProgram
from bench_programs.utils.common import try_start_benchmark, try_stop_benchmark
from bench_programs.utils.openai import openai_generate_async, prepare_openai
from bench_programs.utils.opwise import WorkflowDAG
from openai import AsyncOpenAI

from helium.common import GenerationConfig, Message
from helium.runtime.protocol import HeliumSystemProfile


class OpWiseMapReduceProgram(MapReduceProgram):
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
        client, generation_config = prepare_openai(generation_config)
        generation_kwargs = generation_config.openai_kwargs()
        base_url = generation_config.base_url

        # Prepare inputs
        flattened = self.flatten_inputs(context_prompts, context_question_prompts)
        inputs: list[dict] = [
            {
                "context_idx": context_idx,
                "question_idx": question_idx,
                "context": context_prompt,
                "question": question_prompt,
            }
            for context_idx, question_idx, context_prompt, question_prompt in flattened
        ]

        workflow = self._build_workflow(
            client,
            expert_system_prompt,
            summarizer_system_prompt,
            summary_prompt,
            role_prompts,
            num_agents,
            generation_kwargs,
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
            output_builder.add(
                state["context_idx"], state["question_idx"], state["summary"]
            )

        return output_builder.build(), system_profile

    def _build_workflow(
        self,
        client: AsyncOpenAI,
        expert_system_prompt: str,
        summarizer_system_prompt: str,
        summary_prompt: str,
        role_prompts: list[str] | None,
        num_agents: int,
        generation_kwargs: dict[str, Any],
    ) -> WorkflowDAG:
        def expert_answer(agent_idx: int):
            async def answer_question(state: dict) -> dict:
                role_prompt = role_prompts[agent_idx] if role_prompts else None
                expert_user_prompt = self.build_user_prompt(
                    role_prompt, state["context"], state["question"]
                )
                messages = [
                    Message("system", expert_system_prompt),
                    Message("user", expert_user_prompt),
                ]
                answer = await openai_generate_async(
                    client, messages, generation_kwargs
                )
                if "expert_answers" not in state:
                    state["expert_answers"] = {}
                state["expert_answers"][agent_idx] = answer
                return state

            return answer_question

        async def summarize_answers(state: dict) -> dict:
            unordered_answers = cast(dict[int, str], state["expert_answers"])
            expert_answers = [unordered_answers[i] for i in range(num_agents)]
            summarizer_user_prompt = summary_prompt
            context = state["context"]
            if context is not None:
                summarizer_user_prompt += f"\n\nContext: {context}"
            summarizer_user_prompt += f"\n\nQuestion: {state['question']}\n"
            for agent_i, expert_answer in enumerate(expert_answers):
                summarizer_user_prompt += (
                    f"\nExpert {agent_i + 1} Answer:\n{expert_answer}"
                )
            messages = [
                Message("system", summarizer_system_prompt),
                Message("user", summarizer_user_prompt),
            ]
            answer = await openai_generate_async(client, messages, generation_kwargs)
            state["summary"] = answer
            return state

        workflow = WorkflowDAG()
        workflow.add_node("summarizer", summarize_answers)
        for agent_idx in range(num_agents):
            agent_name = f"expert_{agent_idx}"
            workflow.add_node(agent_name, expert_answer(agent_idx))
            workflow.add_edge(agent_name, "summarizer")
        return workflow
