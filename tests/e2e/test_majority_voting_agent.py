import pytest

from helium.frontend.programs.examples import MCQAProgram
from helium.runtime import HeliumServer
from helium.runtime.protocol import (
    HeliumRequestConfig,
    QueryProfilingConfig,
    SystemProfilingConfig,
)


def run_majority_voting_agent(
    mock_helium_server: HeliumServer,
    request_config: HeliumRequestConfig,
    num_questions: int,
):
    program = MCQAProgram(
        num_agents=3, method="direct", server_config=mock_helium_server.config
    )
    agent = program.create_program_agent("user_inputs")
    agent.compile(user_inputs=[f"TEST{i}" for i in range(1, num_questions + 1)])
    response = agent.run(request_config)

    output = response.outputs
    assert len(output) == 3, "Output length mismatch"

    expected = ["MOCK"] * num_questions
    for out_values in output.values():
        assert out_values == expected, f"Expected {expected}, got {out_values}"


@pytest.mark.timeout(3)
@pytest.mark.parametrize("enable_cache_aware_scheduling", [True, False])
@pytest.mark.parametrize("enable_runtime_adjustment", [True, False])
@pytest.mark.parametrize("system_profiling_config", [None, SystemProfilingConfig()])
@pytest.mark.parametrize("query_profiling_config", [None, QueryProfilingConfig()])
@pytest.mark.parametrize("num_questions", [100])
def test_majority_voting_agent(
    mock_helium_server: HeliumServer,
    enable_cache_aware_scheduling: bool,
    enable_runtime_adjustment: bool,
    system_profiling_config: SystemProfilingConfig | None,
    query_profiling_config: QueryProfilingConfig | None,
    num_questions: int,
):
    request_config = HeliumRequestConfig(
        enable_cache_aware_scheduling=enable_cache_aware_scheduling,
        enable_runtime_adjustment=enable_runtime_adjustment,
        system_profiling_config=system_profiling_config,
        query_profiling_config=query_profiling_config,
    )
    run_majority_voting_agent(mock_helium_server, request_config, num_questions)
