import pytest

from helium import helium
from helium.runtime import HeliumServer
from helium.runtime.protocol import (
    HeliumRequestConfig,
    QueryProfilingConfig,
    SystemProfilingConfig,
)
from tests.utils import english_writing


@pytest.mark.timeout(3)
@pytest.mark.parametrize("enable_cache_aware_scheduling", [True, False])
@pytest.mark.parametrize("enable_runtime_adjustment", [True, False])
@pytest.mark.parametrize("system_profiling_config", [None, SystemProfilingConfig()])
@pytest.mark.parametrize("query_profiling_config", [None, QueryProfilingConfig()])
@pytest.mark.parametrize("num_generations", [100])
def test_english_writing(
    mock_helium_server: HeliumServer,
    enable_cache_aware_scheduling: bool,
    enable_runtime_adjustment: bool,
    system_profiling_config: SystemProfilingConfig | None,
    query_profiling_config: QueryProfilingConfig | None,
    num_generations: int,
):
    request_config = HeliumRequestConfig(
        enable_cache_aware_scheduling=enable_cache_aware_scheduling,
        enable_runtime_adjustment=enable_runtime_adjustment,
        system_profiling_config=system_profiling_config,
        query_profiling_config=query_profiling_config,
    )
    graph = english_writing.build_graph(num_generations)
    outputs = helium.invoke(graph, request_config)
    english_writing.check_outputs(outputs, num_generations)
