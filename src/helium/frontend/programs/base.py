from abc import abstractmethod
from collections.abc import Sequence
from typing import Any

from helium.frontend.agents import Agent
from helium.runtime import HeliumServer, HeliumServerConfig
from helium.runtime.protocol import HeliumRequest, HeliumRequestConfig, HeliumResponse
from helium.utils import http_client, run_coroutine_blocking


class Program:
    def __init__(
        self,
        name: str | None = None,
        server_config: HeliumServerConfig | None = None,
    ):
        self.name = name or self.__class__.__name__
        self.server_config = server_config or HeliumServerConfig()

    async def _run_local(
        self, agents: Sequence[Agent], config: HeliumRequestConfig | None = None
    ) -> HeliumResponse:
        server = HeliumServer.get_started_instance(self.server_config)
        graphs = {agent.name: agent.get_and_reset_compiled_graph() for agent in agents}
        response = await server.execute(graphs, config)
        return response

    def run_agents(
        self, agents: Sequence[Agent], config: HeliumRequestConfig | None = None
    ) -> HeliumResponse:
        if self.server_config.is_local:
            response = run_coroutine_blocking(self._run_local(agents, config))
        else:
            query = {agent.name: agent.serialize() for agent in agents}
            request = HeliumRequest(query=query, config=config).model_dump(mode="json")
            resp = http_client.post(f"{self.server_config.url}/request", json=request)
            resp.raise_for_status()
            response = HeliumResponse.model_validate(resp.json())
        return response

    async def run_agents_async(
        self, agents: Sequence[Agent], config: HeliumRequestConfig | None = None
    ) -> HeliumResponse:
        if self.server_config.is_local:
            response = await self._run_local(agents, config)
        else:
            query = {agent.name: agent.serialize() for agent in agents}
            request = HeliumRequest(query=query, config=config).model_dump(mode="json")
            json_obj = await http_client.apost_json(
                f"{self.server_config.url}/request", json=request
            )
            response = HeliumResponse.model_validate(json_obj)
        return response

    def run(self, *args, **kwargs) -> Any:
        return run_coroutine_blocking(self.run_async(*args, **kwargs))

    @abstractmethod
    async def run_async(self, *args, **kwargs) -> Any:
        raise NotImplementedError()
