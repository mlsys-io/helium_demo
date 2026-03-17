import asyncio
from collections.abc import Sequence

from bench_programs.map_reduce.base import MapReduceProgram
from bench_programs.utils.parrot import (
    ParrotMixin,
    SemanticFunction,
    SemanticVariable,
    parrot_sampling_config,
    parrot_semantic_function,
    parrot_semantic_variable,
    parrot_start_benchmark,
    parrot_stop_benchmark,
)
from parrot import P

from helium.common import GenerationConfig
from helium.runtime.protocol import HeliumSystemProfile


class ParrotMapReduceProgram(ParrotMixin, MapReduceProgram):
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
        with self.get_vm() as vm:
            if generation_config is None:
                generation_config = GenerationConfig.from_env()

            # Start benchmarking
            parrot_start_benchmark(vm)

            # Prepare inputs
            flattened = self.flatten_inputs(context_prompts, context_question_prompts)

            # Start generation
            self.start_timer("generate")

            self.start_timer("prepare")
            sampling_config = parrot_sampling_config(generation_config)
            semantic_functions = self._create_semantic_functions(
                vm,
                expert_system_prompt,
                summarizer_system_prompt,
                summary_prompt,
                role_prompts,
                context_prompts is not None,
                num_agents,
                generation_config.model,
                sampling_config,
            )

            context_prompt_cache: dict[int, SemanticVariable] = {}
            input_semantic_vars: list[
                tuple[int, int, SemanticVariable | None, SemanticVariable]
            ] = []
            for (
                context_idx,
                question_idx,
                context_prompt,
                question_prompt,
            ) in flattened:
                if context_prompt is None:
                    context_var = None
                else:
                    context_var = context_prompt_cache.get(context_idx)
                    if context_var is None:
                        context_var = P.variable(content=context_prompt)
                        context_prompt_cache[context_idx] = context_var
                question_var = P.variable(content=question_prompt)
                input_semantic_vars.append(
                    (context_idx, question_idx, context_var, question_var)
                )
            semantic_variables = [
                (
                    context_idx,
                    question_idx,
                    self._create_semantic_variable(
                        semantic_functions, num_agents, context_var, question_var
                    ),
                )
                for context_idx, question_idx, context_var, question_var in input_semantic_vars
            ]
            self.stop_timer()

            outputs = await asyncio.gather(
                *[variable.aget() for _, _, variable in semantic_variables]
            )

            # Stop generation
            self.stop_timer()

            # Stop benchmarking
            system_profile = parrot_stop_benchmark(vm)

        output_builder = self.OutputBuilder()
        for (context_idx, question_idx, _), answer in zip(semantic_variables, outputs):
            output_builder.add(context_idx, question_idx, answer)

        return output_builder.build(), system_profile

    def _create_semantic_functions(
        self,
        vm: P.VirtualMachine,
        expert_system_prompt: str,
        summarizer_system_prompt: str,
        summary_prompt: str,
        role_prompts: list[str] | None,
        has_contexts: bool,
        num_agents: int,
        model: str,
        sampling_config: P.SamplingConfig,
    ) -> dict[str, SemanticFunction]:
        semantic_functions: dict[str, SemanticFunction] = {}

        # Expert answers
        maybe_role_prompts: Sequence[str | None]
        if role_prompts is None:
            maybe_role_prompts = [None] * num_agents
        else:
            maybe_role_prompts = role_prompts
        expert_user_prompts = [
            self.build_user_prompt(
                role_prompt,
                "{{context}}" if has_contexts else None,
                "{{question}}",
            )
            for role_prompt in maybe_role_prompts
        ]
        expert_system_message = {"role": "system", "content": expert_system_prompt}
        expert_names = [f"expert_{i}" for i in range(num_agents)]

        kwargs = {"question": P.Input, "answer": P.Output(sampling_config)}
        if has_contexts:
            kwargs["context"] = P.Input

        expert_answers = [
            parrot_semantic_function(
                vm,
                expert_name,
                model,
                [
                    expert_system_message,
                    {"role": "user", "content": expert_user_prompt},
                    {"role": "assistant", "content": "{{answer}}"},
                ],
                **kwargs,
            )
            for expert_name, expert_user_prompt in zip(
                expert_names, expert_user_prompts
            )
        ]
        semantic_functions.update(dict(zip(expert_names, expert_answers)))

        # Answer summarization
        summarizer_user_prompt = summary_prompt
        if has_contexts:
            summarizer_user_prompt += "\n\nContext: {{context}}"
        summarizer_user_prompt += "\n\nQuestion: {{question}}\n"
        for i, expert_name in enumerate(expert_names):
            summarizer_user_prompt += f"\nExpert {i + 1} Answer:\n{{{{{expert_name}}}}}"
        summarizer_context = [
            {
                "role": "system",
                "content": summarizer_system_prompt,
            },
            {
                "role": "user",
                "content": summarizer_user_prompt,
            },
            {
                "role": "assistant",
                "content": "{{summary}}",
            },
        ]

        kwargs = kwargs.copy()
        del kwargs["answer"]
        kwargs.update({expert_name: P.Input for expert_name in expert_names})
        kwargs["summary"] = P.Output(sampling_config)

        summarizer = parrot_semantic_function(
            vm, "summarizer", model, summarizer_context, **kwargs
        )
        semantic_functions["summarizer"] = summarizer

        return semantic_functions

    def _create_semantic_variable(
        self,
        semantic_functions: dict[str, SemanticFunction],
        num_agents: int,
        context_prompt: SemanticVariable | None,
        question_prompt: SemanticVariable,
    ) -> SemanticVariable:
        kwargs = {"question": question_prompt}
        if context_prompt is not None:
            kwargs["context"] = context_prompt
        expert_names = [f"expert_{i}" for i in range(num_agents)]
        expert_answers = [
            parrot_semantic_variable(semantic_functions[expert_name], **kwargs)
            for expert_name in expert_names
        ]

        kwargs = kwargs.copy()
        kwargs.update(
            {
                expert_name: answer
                for expert_name, answer in zip(expert_names, expert_answers)
            }
        )

        answer_summary = parrot_semantic_variable(
            semantic_functions["summarizer"], **kwargs
        )

        return answer_summary
