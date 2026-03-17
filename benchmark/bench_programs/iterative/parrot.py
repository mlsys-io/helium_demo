import asyncio

from bench_programs.iterative.base import IterativeProgram
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


class ParrotIterativeProgram(ParrotMixin, IterativeProgram):
    async def _run(
        self,
        system_prompt: str,
        first_prompt_fmt: str,
        subsequent_prompt_fmt: str,
        document_chunks: tuple[list[str], ...],
        generation_config: GenerationConfig | None,
        **kwargs,
    ) -> tuple[list[str], HeliumSystemProfile]:
        with self.get_vm() as vm:
            if generation_config is None:
                generation_config = GenerationConfig.from_env()

            # Start benchmarking
            parrot_start_benchmark(vm)

            # Prepare inputs
            flattened = self.flatten_inputs(document_chunks)

            # Start generation
            self.start_timer("generate")

            self.start_timer("prepare")
            sampling_config = parrot_sampling_config(generation_config)
            semantic_functions = self._create_semantic_functions(
                vm,
                system_prompt,
                first_prompt_fmt,
                subsequent_prompt_fmt,
                len(document_chunks),
                generation_config.model,
                sampling_config,
            )

            semantic_variables = [
                (
                    index,
                    self._create_semantic_variable(
                        semantic_functions,
                        [P.variable(content=chunk) for chunk in chunks],
                    ),
                )
                for index, chunks in flattened
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
        system_prompt: str,
        first_prompt_fmt: str,
        subsequent_prompt_fmt: str,
        num_chunks: int,
        model: str,
        sampling_config: P.SamplingConfig,
    ) -> dict[str, SemanticFunction]:
        semantic_functions: dict[str, SemanticFunction] = {}
        system_message = {"role": "system", "content": system_prompt}

        # First chunk
        messages = [
            system_message,
            {
                "role": "user",
                "content": first_prompt_fmt.format("{{chunk}}"),
            },
            {"role": "assistant", "content": "{{summary}}"},
        ]
        summary_name = self._get_summary_name(0)
        summary = parrot_semantic_function(
            vm,
            summary_name,
            model,
            messages,
            chunk=P.Input,
            summary=P.Output(sampling_config),
        )
        semantic_functions[summary_name] = summary

        # Remaining chunks
        for chunk_i in range(1, num_chunks):
            summary_name = self._get_summary_name(chunk_i)
            messages = [
                system_message,
                {
                    "role": "user",
                    "content": subsequent_prompt_fmt.format(
                        "{{prev_summary}}", "{{chunk}}"
                    ),
                },
                {"role": "assistant", "content": "{{summary}}"},
            ]
            summary = parrot_semantic_function(
                vm,
                summary_name,
                model,
                messages,
                prev_summary=P.Input,
                chunk=P.Input,
                summary=P.Output(sampling_config),
            )
            semantic_functions[summary_name] = summary

        return semantic_functions

    def _create_semantic_variable(
        self,
        semantic_functions: dict[str, SemanticFunction],
        chunks: list[SemanticVariable],
    ) -> SemanticVariable:
        first_chunk, *remaining_chunks = chunks

        # First chunk
        summary_name = self._get_summary_name(0)
        summary = parrot_semantic_variable(
            semantic_functions[summary_name], chunk=first_chunk
        )

        # Remaining chunks
        for i, chunk in enumerate(remaining_chunks, start=1):
            summary_name = self._get_summary_name(i)
            summary = parrot_semantic_variable(
                semantic_functions[summary_name],
                prev_summary=summary,
                chunk=chunk,
            )

        return summary

    def _get_summary_name(self, chunk_i: int) -> str:
        return f"summary_{chunk_i}"
