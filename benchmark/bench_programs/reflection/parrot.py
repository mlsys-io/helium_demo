import asyncio

from bench_programs.reflection.base import ReflectionProgram
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


class ParrotReflectionProgram(ParrotMixin, ReflectionProgram):
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
        with self.get_vm() as vm:
            if generation_config is None:
                generation_config = GenerationConfig.from_env()

            # Start benchmarking
            parrot_start_benchmark(vm)

            # Prepare inputs
            flattened = self.flatten_inputs(contexts, context_questions)

            # Start generation
            self.start_timer("generate")

            self.start_timer("prepare")
            sampling_config = parrot_sampling_config(generation_config)
            semantic_functions = self._create_semantic_functions(
                vm,
                system_prompts,
                financial_analyst_fmt,
                extraction_critic_fmt,
                calculation_critic_fmt,
                final_answer_fmt,
                generation_config.model,
                sampling_config,
            )

            context_prompt_cache: dict[int, SemanticVariable] = {}
            input_semantic_vars: list[
                tuple[int, int, SemanticVariable, SemanticVariable]
            ] = []
            for context_idx, question_idx, context, question in flattened:
                context_var = context_prompt_cache.get(context_idx)
                if context_var is None:
                    context_var = P.variable(content=context)
                    context_prompt_cache[context_idx] = context_var
                question_var = P.variable(content=question)
                input_semantic_vars.append(
                    (context_idx, question_idx, context_var, question_var)
                )
            semantic_variables = [
                (
                    context_idx,
                    question_idx,
                    self._create_semantic_variable(
                        semantic_functions, context_var, question_var
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
        system_prompts: tuple[str, str, str],
        financial_analyst_fmt: str,
        extraction_critic_fmt: str,
        calculation_critic_fmt: str,
        final_answer_fmt: str,
        model: str,
        sampling_config: P.SamplingConfig,
    ) -> dict[str, SemanticFunction]:
        semantic_functions: dict[str, SemanticFunction] = {}

        fa_system_prompt, ec_system_prompt, cc_system_prompt = system_prompts
        context_question_kwargs: dict[str, type[P.Input]] = {
            "context": P.Input,
            "question": P.Input,
        }

        # Financial Analyst
        financial_analyst_history: list[dict[str, str]] = [
            {"role": "system", "content": fa_system_prompt},
            {
                "role": "user",
                "content": financial_analyst_fmt.format(
                    context="{{context}}", question="{{question}}"
                ),
            },
            {"role": "assistant", "content": "{{answer}}"},
        ]
        answer = parrot_semantic_function(
            vm,
            "financial_analyst",
            model,
            financial_analyst_history,
            **context_question_kwargs,
            answer=P.Output(sampling_config),
        )
        semantic_functions["financial_analyst"] = answer

        # Extraction Critic
        messages = [
            {"role": "system", "content": ec_system_prompt},
            {
                "role": "user",
                "content": extraction_critic_fmt.format(
                    context="{{context}}",
                    question="{{question}}",
                    response="{{response}}",
                ),
            },
            {"role": "assistant", "content": "{{critic}}"},
        ]
        extraction_critic = parrot_semantic_function(
            vm,
            "extraction_critic",
            model,
            messages,
            **context_question_kwargs,
            response=P.Input,
            critic=P.Output(sampling_config),
        )
        semantic_functions["extraction_critic"] = extraction_critic

        # Calculation Critic
        messages = [
            {"role": "system", "content": cc_system_prompt},
            {
                "role": "user",
                "content": calculation_critic_fmt.format(
                    context="{{context}}",
                    question="{{question}}",
                    response="{{response}}",
                    critic="{{critic}}",
                ),
            },
            {"role": "assistant", "content": "{{calculation_critic}}"},
        ]
        calculation_critic = parrot_semantic_function(
            vm,
            "calculation_critic",
            model,
            messages,
            **context_question_kwargs,
            response=P.Input,
            critic=P.Input,
            calculation_critic=P.Output(sampling_config),
        )
        semantic_functions["calculation_critic"] = calculation_critic

        # Final Answer
        financial_analyst_history.extend(
            [
                {
                    "role": "user",
                    "content": final_answer_fmt.format(
                        extraction_critic="{{extraction_critic}}",
                        calculation_critic="{{calculation_critic}}",
                    ),
                },
                {"role": "assistant", "content": "{{final_answer}}"},
            ]
        )
        final_answer = parrot_semantic_function(
            vm,
            "final_answer",
            model,
            financial_analyst_history,
            **context_question_kwargs,
            answer=P.Input,
            extraction_critic=P.Input,
            calculation_critic=P.Input,
            final_answer=P.Output(sampling_config),
        )
        semantic_functions["final_answer"] = final_answer

        return semantic_functions

    def _create_semantic_variable(
        self,
        semantic_functions: dict[str, SemanticFunction],
        context: SemanticVariable,
        question: SemanticVariable,
    ) -> SemanticVariable:
        context_question_kwargs = {
            "context": context,
            "question": question,
        }

        # Financial Analyst
        answer = parrot_semantic_variable(
            semantic_functions["financial_analyst"], **context_question_kwargs
        )

        # Extraction Critic
        extraction_critic = parrot_semantic_variable(
            semantic_functions["extraction_critic"],
            **context_question_kwargs,
            response=answer,
        )

        # Calculation Critic
        calculation_critic = parrot_semantic_variable(
            semantic_functions["calculation_critic"],
            **context_question_kwargs,
            response=answer,
            critic=extraction_critic,
        )

        # Final Answer
        final_answer = parrot_semantic_variable(
            semantic_functions["final_answer"],
            **context_question_kwargs,
            answer=answer,
            extraction_critic=extraction_critic,
            calculation_critic=calculation_critic,
        )

        return final_answer
