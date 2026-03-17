import pytest

from helium.frontend.agents.examples import ChatAgent
from helium.runtime import HeliumServer
from helium.runtime.protocol import (
    HeliumRequestConfig,
    QueryProfilingConfig,
    SystemProfilingConfig,
)


def run_chat_agent(
    mock_helium_server: HeliumServer,
    request_config: HeliumRequestConfig,
    num_prompts: int,
):
    agent = ChatAgent(
        server_config=mock_helium_server.config,
        system_prompt="You are a helpful assistant.",
    )
    response = agent.run(
        request_config, inputs=[f"TEST{i}" for i in range(1, num_prompts + 1)]
    )

    outputs = response.outputs
    assert len(outputs) == 1, "Output length mismatch"

    expected = ["MOCK"] * num_prompts
    for out_values in outputs.values():
        assert out_values == expected, f"Expected {expected}, got {out_values}"


@pytest.mark.timeout(3)
@pytest.mark.parametrize("enable_cache_aware_scheduling", [True, False])
@pytest.mark.parametrize("enable_runtime_adjustment", [True, False])
@pytest.mark.parametrize("system_profiling_config", [None, SystemProfilingConfig()])
@pytest.mark.parametrize("query_profiling_config", [None, QueryProfilingConfig()])
@pytest.mark.parametrize("num_prompts", [100])
def test_chat_agent(
    mock_helium_server: HeliumServer,
    enable_cache_aware_scheduling: bool,
    enable_runtime_adjustment: bool,
    system_profiling_config: SystemProfilingConfig | None,
    query_profiling_config: QueryProfilingConfig | None,
    num_prompts: int,
):
    request_config = HeliumRequestConfig(
        enable_cache_aware_scheduling=enable_cache_aware_scheduling,
        enable_runtime_adjustment=enable_runtime_adjustment,
        system_profiling_config=system_profiling_config,
        query_profiling_config=query_profiling_config,
    )
    run_chat_agent(mock_helium_server, request_config, num_prompts)
