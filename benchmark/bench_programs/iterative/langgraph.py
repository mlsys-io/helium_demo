from typing import Any, TypedDict

from bench_programs.iterative.base import IterativeProgram
from bench_programs.utils.common import try_start_benchmark, try_stop_benchmark
from bench_programs.utils.langgraph import get_langgraph_openai_client
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph

from helium.common import GenerationConfig
from helium.runtime.protocol import HeliumSystemProfile


class State(TypedDict):
    index: int
    chunks: tuple[str, ...]
    summary: list[str]


class LangGraphIterativeProgram(IterativeProgram):
    async def _run(
        self,
        system_prompt: str,
        first_prompt_fmt: str,
        subsequent_prompt_fmt: str,
        document_chunks: tuple[list[str], ...],
        generation_config: GenerationConfig | None,
        **kwargs,
    ) -> tuple[list[str], HeliumSystemProfile]:
        if generation_config is None:
            generation_config = GenerationConfig.from_env()

        base_url = generation_config.base_url

        workflow = self._build_workflow(
            system_prompt,
            first_prompt_fmt,
            subsequent_prompt_fmt,
            len(document_chunks),
            generation_config,
        ).compile()

        # Prepare inputs
        flattened = self.flatten_inputs(document_chunks)
        inputs = [
            {"index": index, "chunks": chunks, "summary": []}
            for index, chunks in flattened
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
            output_builder.add(output["index"], output["summary"][-1])

        return output_builder.build(), system_profile

    def _build_workflow(
        self,
        system_prompt: str,
        first_prompt_fmt: str,
        subsequent_prompt_fmt: str,
        num_chunks: int,
        generation_config: GenerationConfig,
    ) -> StateGraph:
        llm = get_langgraph_openai_client(generation_config)

        def summary(chunk_idx: int):
            def summarize(state: State) -> dict[str, Any]:
                chunk = state["chunks"][chunk_idx]
                if chunk_idx == 0:
                    prompt = first_prompt_fmt.format(chunk)
                else:
                    summary = state["summary"][-1]
                    prompt = subsequent_prompt_fmt.format(summary, chunk)
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=prompt),
                ]
                summary = llm.invoke(messages)
                return {"summary": [summary.content]}

            return summarize

        workflow_builder = StateGraph(State)

        prev_node = START
        for chunk_idx in range(num_chunks):
            node_name = f"chunk_{chunk_idx}"
            workflow_builder.add_node(node_name, summary(chunk_idx))
            workflow_builder.add_edge(prev_node, node_name)
            prev_node = node_name

        return workflow_builder
