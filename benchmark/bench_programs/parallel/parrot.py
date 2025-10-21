import asyncio

from bench_programs.parallel.base import ParallelProgram
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


class ParrotParallelProgram(ParrotMixin, ParallelProgram):
    async def _run(
        self,
        expert_system_prompts: list[str],
        writer_system_prompt: str,
        extraction_instruction_fmt: str,
        summary_instruction_fmt: str,
        report_instruction_fmt: str,
        item_descriptions: list[str],
        review_chunks: tuple[list[str], ...],
        generation_config: GenerationConfig | None,
        **kwargs,
    ) -> tuple[list[str], HeliumSystemProfile]:
        with self.get_vm() as vm:
            if generation_config is None:
                generation_config = GenerationConfig.from_env()

            # Start benchmarking
            parrot_start_benchmark(vm)

            # Start generation
            self.start_timer("generate")

            # Prepare inputs
            flattened = self.flatten_inputs(item_descriptions, review_chunks)

            self.start_timer("prepare")
            sampling_config = parrot_sampling_config(generation_config)
            semantic_functions = self._create_semantic_functions(
                vm,
                expert_system_prompts,
                writer_system_prompt,
                extraction_instruction_fmt,
                summary_instruction_fmt,
                report_instruction_fmt,
                len(review_chunks),
                generation_config.model,
                sampling_config,
            )

            input_semantic_vars: list[
                tuple[int, SemanticVariable, list[SemanticVariable]]
            ] = []
            for index, item_description, chunks in flattened:
                item_var = P.variable(content=item_description)
                chunk_vars = [P.variable(content=chunk) for chunk in chunks]
                input_semantic_vars.append((index, item_var, chunk_vars))
            semantic_variables = [
                (
                    index,
                    self._create_semantic_variable(
                        semantic_functions,
                        len(expert_system_prompts),
                        item_var,
                        chunk_vars,
                    ),
                )
                for index, item_var, chunk_vars in input_semantic_vars
            ]
            self.stop_timer()

            outputs = await asyncio.gather(
                *[variable.aget() for _, variable in semantic_variables]
            )

            # Stop generation
            self.stop_timer()

            # Stop benchmarking
            system_profile = parrot_stop_benchmark(vm)

        output_builder = self.OutputBuilder()
        for (index, _), output in zip(semantic_variables, outputs):
            output_builder.add(index, output)

        return output_builder.build(), system_profile

    def _create_semantic_functions(
        self,
        vm: P.VirtualMachine,
        expert_system_prompts: list[str],
        writer_system_prompt: str,
        extraction_instruction_fmt: str,
        summary_instruction_fmt: str,
        report_instruction_fmt: str,
        num_review_chunks_per_item: int,
        model: str,
        sampling_config: P.SamplingConfig,
    ) -> dict[str, SemanticFunction]:
        semantic_functions: dict[str, SemanticFunction] = {}

        expert_summary_names = [
            f"expert_{expert_i}_summary"
            for expert_i in range(len(expert_system_prompts))
        ]
        chunk_names = [
            f"chunk_{chunk_i}" for chunk_i in range(num_review_chunks_per_item)
        ]
        for expert_i, (expert_summary_name, system_prompt) in enumerate(
            zip(expert_summary_names, expert_system_prompts)
        ):
            system_message = {"role": "system", "content": system_prompt}
            # Extract insights
            for chunk_i in range(num_review_chunks_per_item):
                expert_extraction_name = f"expert_{expert_i}_chunk_{chunk_i}"
                messages = [
                    system_message,
                    {
                        "role": "user",
                        "content": extraction_instruction_fmt.format(
                            "{{item}}", "{{chunk}}"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": "{{extracted_insights}}",
                    },
                ]
                semantic_functions[expert_extraction_name] = parrot_semantic_function(
                    vm,
                    expert_extraction_name,
                    model,
                    messages,
                    item=P.Input,
                    chunk=P.Input,
                    extracted_insights=P.Output(sampling_config),
                )
            # Summarize insights
            messages = [
                system_message,
                {
                    "role": "user",
                    "content": summary_instruction_fmt.format(
                        "{{item}}",
                        *[f"{{{{{chunk_name}}}}}" for chunk_name in chunk_names],
                    ),
                },
                {"role": "assistant", "content": "{{summary}}"},
            ]
            semantic_functions[expert_summary_name] = parrot_semantic_function(
                vm,
                expert_summary_name,
                model,
                messages,
                item=P.Input,
                **{chunk_name: P.Input for chunk_name in chunk_names},
                summary=P.Output(sampling_config),
            )

        # Generate report
        messages = [
            {"role": "system", "content": writer_system_prompt},
            {
                "role": "user",
                "content": report_instruction_fmt.format(
                    "{{item}}",
                    *[f"{{{{{name}}}}}" for name in expert_summary_names],
                ),
            },
            {"role": "assistant", "content": "{{report}}"},
        ]
        report_name = "report"
        semantic_functions[report_name] = parrot_semantic_function(
            vm,
            report_name,
            model,
            messages,
            item=P.Input,
            **{name: P.Input for name in expert_summary_names},
            report=P.Output(sampling_config),
        )

        return semantic_functions

    def _create_semantic_variable(
        self,
        semantic_functions: dict[str, SemanticFunction],
        num_experts: int,
        item_description: SemanticVariable,
        review_chunks: list[SemanticVariable],
    ) -> SemanticVariable:
        expert_summary_names = [
            f"expert_{expert_i}_summary" for expert_i in range(num_experts)
        ]
        chunk_names = [f"chunk_{i}" for i in range(len(review_chunks))]

        insight_summaries: list[SemanticVariable] = []
        for expert_i, expert_summary_name in enumerate(expert_summary_names):
            # Extract insights
            extracted_insights: list[SemanticVariable] = []
            for chunk_i, chunk in enumerate(review_chunks):
                expert_extraction_name = f"expert_{expert_i}_chunk_{chunk_i}"
                extracted_insight = parrot_semantic_variable(
                    semantic_functions[expert_extraction_name],
                    item=item_description,
                    chunk=chunk,
                )
                extracted_insights.append(extracted_insight)
            # Summarize insights
            insight_summary = parrot_semantic_variable(
                semantic_functions[expert_summary_name],
                item=item_description,
                **{
                    chunk_name: extracted_insight
                    for chunk_name, extracted_insight in zip(
                        chunk_names, extracted_insights
                    )
                },
            )
            insight_summaries.append(insight_summary)

        # Generate report
        insight_report = parrot_semantic_variable(
            semantic_functions["report"],
            item=item_description,
            **{
                name: summary
                for name, summary in zip(expert_summary_names, insight_summaries)
            },
        )

        return insight_report
