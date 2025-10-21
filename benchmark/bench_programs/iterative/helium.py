from collections.abc import Sequence
from typing import Literal

from bench_programs.iterative.base import IterativeProgram, OutputType
from bench_programs.utils.common import random_shuffle

from helium import ops
from helium.common import GenerationConfig
from helium.frontend.agents import Agent
from helium.frontend.programs import Program as HeliumProgram
from helium.runtime import HeliumServerConfig
from helium.runtime.protocol import (
    HeliumRequestConfig,
    HeliumResponse,
    HeliumSystemProfile,
)


class IterativeAgent(Agent):
    def __init__(
        self,
        system_prompt: str,
        first_prompt_fmt: str,
        subsequent_prompt_fmt: str,
        chunk_ops: Sequence[ops.Op],
        server_config: HeliumServerConfig | None,
        generation_config: GenerationConfig | None,
    ) -> None:
        super().__init__(
            server_config=server_config,
            system_prompt=system_prompt,
            first_prompt_fmt=first_prompt_fmt,
            subsequent_prompt_fmt=subsequent_prompt_fmt,
            chunk_ops=chunk_ops,
            generation_config=generation_config,
        )

    def build_ops(
        self,
        system_prompt: str,
        first_prompt_fmt: str,
        subsequent_prompt_fmt: str,
        chunk_ops: list[ops.Op],
        generation_config: GenerationConfig | None,
    ) -> list[ops.OutputOp]:
        first_chunk, *remaining_chunks = chunk_ops
        first_prompt = ops.format_op(first_prompt_fmt, first_chunk)
        first_message = [
            ops.OpMessage(role="system", content=system_prompt),
            ops.OpMessage(role="user", content=first_prompt),
        ]
        summary = ops.llm_chat(first_message, generation_config, return_history=False)
        for chunk in remaining_chunks:
            subsequent_prompt = ops.format_op(subsequent_prompt_fmt, summary, chunk)
            message = [
                ops.OpMessage(role="system", content=system_prompt),
                ops.OpMessage(role="user", content=subsequent_prompt),
            ]
            summary = ops.llm_chat(message, generation_config, return_history=False)
        output = ops.as_output("summary", summary)
        return [output]


class HeliumIterativeProgram(IterativeProgram, HeliumProgram):
    def __init__(
        self,
        request_config: HeliumRequestConfig | None = None,
        server_config: HeliumServerConfig | None = None,
        iterative_agent: IterativeAgent | None = None,
    ) -> None:
        IterativeProgram.__init__(self)
        HeliumProgram.__init__(self, server_config=server_config)
        self.iterative_agent = iterative_agent
        self.request_config = request_config

    def create_agent(
        self,
        system_prompt: str,
        first_prompt_fmt: str,
        subsequent_prompt_fmt: str,
        document_chunks: tuple[list[str], ...],
        generation_config: GenerationConfig | None,
        **_,
    ) -> IterativeAgent:
        chunk_ops = [ops.InputOp(f"chunk_{i}") for i in range(len(document_chunks))]

        if self.iterative_agent is None:
            iterative_agent = IterativeAgent(
                system_prompt=system_prompt,
                first_prompt_fmt=first_prompt_fmt,
                subsequent_prompt_fmt=subsequent_prompt_fmt,
                chunk_ops=chunk_ops,
                server_config=self.server_config,
                generation_config=generation_config,
            )
        else:
            iterative_agent = self.iterative_agent
            # Replace LLM ops' generation config
            for op in iterative_agent.graph.iter_ops(ops.LLMOp):
                op.config = generation_config or GenerationConfig.from_env()

        inputs = {
            chunk_op.name: documents
            for chunk_op, documents in zip(chunk_ops, document_chunks)
        }

        iterative_agent.compile(**inputs)
        return iterative_agent

    async def _run(
        self,
        system_prompt: str,
        first_prompt_fmt: str,
        subsequent_prompt_fmt: str,
        document_chunks: tuple[list[str], ...],
        generation_config: GenerationConfig | None,
        **kwargs,
    ) -> tuple[OutputType, HeliumSystemProfile]:
        # Prepare inputs
        indices = list(range(len(document_chunks[0])))
        document_chunks = tuple(
            random_shuffle(chunks, inplace=False) for chunks in document_chunks
        )

        agent = self.create_agent(
            system_prompt=system_prompt,
            first_prompt_fmt=first_prompt_fmt,
            subsequent_prompt_fmt=subsequent_prompt_fmt,
            document_chunks=document_chunks,
            generation_config=generation_config,
        )

        self.start_timer("generate")
        response = await agent.run_async(self.request_config)
        self.stop_timer()

        outputs = response.outputs["summary"]
        summary = self.OutputBuilder().update(zip(indices, outputs)).build()

        return summary, response.system_profile

    async def _precompute(
        self,
        system_prompt: str,
        first_prompt_fmt: str,
        subsequent_prompt_fmt: str,
        document_chunks: tuple[list[str], ...],
        generation_config: GenerationConfig | None,
        precompute_mode: Literal["none", "only", "both"],
        **kwargs,
    ) -> HeliumResponse:
        # Prepare inputs
        document_chunks = tuple(
            random_shuffle(chunks, inplace=False) for chunks in document_chunks
        )

        agent = self.create_agent(
            system_prompt=system_prompt,
            first_prompt_fmt=first_prompt_fmt,
            subsequent_prompt_fmt=subsequent_prompt_fmt,
            document_chunks=document_chunks,
            generation_config=generation_config,
        )

        request_config = (
            HeliumRequestConfig()
            if self.request_config is None
            else self.request_config.model_copy()
        )
        request_config.precompute_mode = precompute_mode
        return await agent.run_async(request_config)
