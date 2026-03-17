from bench_programs.iterative.base import IterativeProgram
from bench_programs.utils.agentscope import (
    AgentScopeAgent,
    FormatMsg,
    Msg,
    PlaceholderMsg,
    RpcObject,
    agentscope_call_agent,
    agentscope_reinit_from_config,
)
from bench_programs.utils.common import try_start_benchmark, try_stop_benchmark

from helium.common import GenerationConfig
from helium.runtime.protocol import HeliumSystemProfile


class ASIterativeProgram(IterativeProgram):
    async def _run(
        self,
        system_prompt: str,
        first_prompt_fmt: str,
        subsequent_prompt_fmt: str,
        document_chunks: tuple[list[str], ...],
        generation_config: GenerationConfig | None,
        **kwargs,
    ) -> tuple[list[str], HeliumSystemProfile]:
        self.start_timer("prepare")
        llm_config = agentscope_reinit_from_config(generation_config)
        base_url = llm_config.base_url

        agent = AgentScopeAgent.dist("agent", system_prompt)
        self.stop_timer()

        # Prepare inputs
        flattened = self.flatten_inputs(document_chunks)

        # Start benchmarking
        try_start_benchmark(base_url)

        self.start_timer("generate")
        outputs = [
            self._run_chunks(
                index,
                agent,
                first_prompt_fmt,
                subsequent_prompt_fmt,
                chunks,
                llm_config,
            )
            for index, chunks in flattened
        ]
        output_builder = self.OutputBuilder()
        for index, output in outputs:
            output_builder.add(index, output.content)
        self.stop_timer()

        # Stop benchmarking
        system_profile = try_stop_benchmark(base_url)

        AgentScopeAgent.stop_all([agent])

        return output_builder.build(), system_profile

    def _run_chunks(
        self,
        index: int,
        agent: RpcObject,
        first_prompt_fmt: str,
        subsequent_prompt_fmt: str,
        chunks: tuple[str, ...],
        generation_config: GenerationConfig,
    ) -> tuple[int, PlaceholderMsg]:
        first_chunk, *remaining_chunks = chunks
        message = [Msg("user", first_prompt_fmt.format(first_chunk), "user")]
        summary = agentscope_call_agent(agent, message, generation_config)
        for chunk in remaining_chunks:
            message = [FormatMsg("user", subsequent_prompt_fmt, summary, chunk)]
            summary = agentscope_call_agent(agent, message, generation_config)
        return index, summary
