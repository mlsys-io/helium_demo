from dataclasses import dataclass, field
from typing import Literal

from helium.graphs import CompiledGraph
from helium.runtime.protocol import (
    HeliumQueryProfile,
    HeliumRequestConfig,
    HeliumResponse,
    QueryProfilingConfig,
    SystemProfilingConfig,
)
from helium.runtime.utils.queue import TSQueue
from helium.utils import unique_id


@dataclass(slots=True)
class RequestInfo:
    request_id: str

    query_graphs: dict[str, CompiledGraph]

    enable_cache_aware_scheduling: bool
    enable_runtime_adjustment: bool
    precompute_mode: Literal["none", "only", "both"]
    precompute_cacheable_inputs: bool

    query_profiling_config: QueryProfilingConfig | None
    system_profiling_config: SystemProfilingConfig | None

    # These are to be set and modified by the optimizer and processor
    compiled_graph: CompiledGraph = field(init=False)
    disjoint_graphs: list[CompiledGraph] = field(init=False)
    llm_service_map: dict[str, str] = field(init=False)
    llm_partition_counts: dict[str, int] = field(init=False)
    query_profile: HeliumQueryProfile | None = field(init=False)


class RequestHandler:
    def __init__(
        self,
        query: dict[str, CompiledGraph],
        config: HeliumRequestConfig | None = None,
    ) -> None:
        self._request_id = "req-" + unique_id()
        self.query = query
        self.config = HeliumRequestConfig() if config is None else config
        self._result_queue: TSQueue[HeliumResponse] = TSQueue()

    @property
    def request_id(self) -> str:
        return self._request_id

    async def put_result(self, result: HeliumResponse) -> None:
        await self._result_queue.put(result)

    async def get_result(self) -> HeliumResponse:
        return await self._result_queue.get()
