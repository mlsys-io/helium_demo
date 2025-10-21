from typing import Any

from bench_programs.iterative.base import IterativeProgram
from bench_programs.utils.common import try_start_benchmark, try_stop_benchmark
from bench_programs.utils.openai import openai_generate_async, prepare_openai
from bench_programs.utils.opwise import WorkflowDAG
from openai import AsyncOpenAI

from helium.common import GenerationConfig, Message
from helium.runtime.protocol import HeliumSystemProfile


class OpWiseIterativeProgram(IterativeProgram):
    async def _run(
        self,
        system_prompt: str,
        first_prompt_fmt: str,
        subsequent_prompt_fmt: str,
        document_chunks: tuple[list[str], ...],
        generation_config: GenerationConfig | None,
        **kwargs,
    ) -> tuple[list[str], HeliumSystemProfile]:
        client, generation_config = prepare_openai(generation_config)
        generation_kwargs = generation_config.openai_kwargs()
        base_url = generation_config.base_url

        # Prepare inputs
        flattened = self.flatten_inputs(document_chunks)
        inputs = [{"index": index, "chunks": chunks} for index, chunks in flattened]

        workflow = self._build_workflow(
            client,
            system_prompt,
            first_prompt_fmt,
            subsequent_prompt_fmt,
            len(document_chunks),
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
            output_builder.add(state["index"], state["summary"])

        return output_builder.build(), system_profile

    def _build_workflow(
        self,
        client: AsyncOpenAI,
        system_prompt: str,
        first_prompt_fmt: str,
        subsequent_prompt_fmt: str,
        num_chunks: int,
        generation_kwargs: dict[str, Any],
    ) -> WorkflowDAG:
        def summary(chunk_idx: int):
            async def summarize(state: dict) -> dict:
                chunk = state["chunks"][chunk_idx]
                if chunk_idx == 0:
                    prompt = first_prompt_fmt.format(chunk)
                else:
                    summary = state["summary"]
                    prompt = subsequent_prompt_fmt.format(summary, chunk)
                messages = [Message("system", system_prompt), Message("user", prompt)]
                summary = await openai_generate_async(
                    client, messages, generation_kwargs
                )
                state["summary"] = summary
                return state

            return summarize

        workflow = WorkflowDAG()

        prev_node: str | None = None
        for chunk_idx in range(num_chunks):
            node_name = f"chunk_{chunk_idx}"
            workflow.add_node(node_name, summary(chunk_idx))
            if prev_node is not None:
                workflow.add_edge(prev_node, node_name)
            prev_node = node_name

        return workflow
