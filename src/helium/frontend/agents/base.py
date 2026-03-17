from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

import dill

from helium.graphs import CompiledGraph, Graph
from helium.ops import OutputOp
from helium.runtime import HeliumServer, HeliumServerConfig
from helium.runtime.protocol import (
    HeliumQueryProfile,
    HeliumRequest,
    HeliumRequestConfig,
    HeliumResponse,
)
from helium.utils import http_client, run_coroutine_blocking, unique_id


class Agent(ABC):
    def __init__(
        self,
        *args,
        name: str | None = None,
        server_config: HeliumServerConfig | None = None,
        **kwargs,
    ) -> None:
        self.name = unique_id() if name is None else name
        self.server_config = server_config or HeliumServerConfig()
        self._ops = self.build_ops(*args, **kwargs)
        self._graph = Graph.from_ops(self._ops)
        self._compiled_graph: CompiledGraph | None = None

    @property
    def graph(self) -> Graph:
        return self._graph

    @property
    def output_ops(self) -> list[OutputOp]:
        return list(self._graph.output_ops.values())

    @abstractmethod
    def build_ops(self, *args, **kwargs) -> list[OutputOp]:
        raise NotImplementedError()

    def compile(self, *_, **inputs: list[str]) -> None:
        if self._compiled_graph is None:
            self._compiled_graph = self._graph.compile(**inputs)

    def get_and_reset_compiled_graph(self) -> CompiledGraph:
        if self._compiled_graph is None:
            raise ValueError("Graph has not been compiled.")
        ret = self._compiled_graph
        self._compiled_graph = None
        return ret

    def serialize(self) -> dict[str, Any]:
        if self._compiled_graph is None:
            raise ValueError("Graph has not been compiled.")
        return self._compiled_graph.serialize()

    def compile_and_serialize(self, *_, **inputs: list[str]) -> dict[str, Any]:
        self.compile(**inputs)
        return self.serialize()

    async def _run_local(
        self, inputs: dict[str, list[str]], config: HeliumRequestConfig | None = None
    ) -> HeliumResponse:
        server = HeliumServer.get_started_instance(self.server_config)
        self.compile(**inputs)
        graphs = {self.name: self.get_and_reset_compiled_graph()}
        response = await server.execute(graphs, config)
        response.outputs = response.outputs[self.name]
        return response

    def run(
        self, config: HeliumRequestConfig | None = None, *_, **inputs: list[str]
    ) -> HeliumResponse:
        if self.server_config.is_local:
            response = run_coroutine_blocking(self._run_local(inputs, config))
        else:
            query = {self.name: self.compile_and_serialize(**inputs)}
            request = HeliumRequest(query=query, config=config).model_dump(mode="json")
            resp = http_client.post(f"{self.server_config.url}/request", json=request)
            resp.raise_for_status()
            response = HeliumResponse.model_validate(resp.json())
            response.outputs = response.outputs[self.name]
        return response

    async def run_async(
        self, config: HeliumRequestConfig | None = None, *_, **inputs: list[str]
    ) -> HeliumResponse:
        if self.server_config.is_local:
            response = await self._run_local(inputs, config)
        else:
            query = {self.name: self.compile_and_serialize(**inputs)}
            request = HeliumRequest(query=query, config=config).model_dump(mode="json")
            json_obj = await http_client.apost_json(
                f"{self.server_config.url}/request", json=request
            )
            response = HeliumResponse.model_validate(json_obj)
            response.outputs = response.outputs[self.name]
        return response

    def save(
        self,
        path: str | Path,
        query_profile: HeliumQueryProfile | None,
    ) -> None:
        compiled_graph = (
            None
            if self._compiled_graph is None
            else self.get_and_reset_compiled_graph()
        )
        obj = {"agent": self, "query_profile": query_profile}
        with open(path, "wb") as f:
            dill.dump(obj, f)
        self._compiled_graph = compiled_graph

    @classmethod
    def load(cls, path: str | Path) -> tuple[Self, HeliumQueryProfile | None]:
        with open(path, "rb") as f:
            obj = dill.load(f)
        if not isinstance(obj, dict) or "agent" not in obj:
            raise ValueError("Invalid file format")
        return obj["agent"], obj.get("query_profile", None)


class WrapperAgent(Agent):
    def __init__(
        self,
        agents: Sequence[Agent],
        name: str | None = None,
        server_config: HeliumServerConfig | None = None,
    ) -> None:
        super().__init__(name=name, server_config=server_config, agents=agents)

    def build_ops(self, agents: list[Agent]) -> list[OutputOp]:
        return [op for agent in agents for op in agent.output_ops]
