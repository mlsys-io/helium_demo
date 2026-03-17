import pytest

from helium.frontend.programs.examples import MCQAProgram
from helium.runtime import HeliumServer
from helium.runtime.protocol import HeliumRequestConfig


@pytest.mark.timeout(3)
def test_graph_optimization(mock_helium_server: HeliumServer):
    mcqa_program = MCQAProgram(num_agents=5, method="direct")
    agent = mcqa_program.create_program_agent(input_name="user_inputs")

    assert agent.graph.node_count == 16, "Initial node count mismatch"

    # 1. Compile the agent's compute graph
    agent.compile(user_inputs=["test"])

    # 2. Optimize the compiled graph
    query = {"graph": agent.get_and_reset_compiled_graph()}
    config = HeliumRequestConfig()
    info = mock_helium_server._prepare_request_info("0", query, config)
    mock_helium_server.optimizer.initial_rewrite(info)

    # 3. Plot the optimized graph
    graph = info.compiled_graph.graph
    assert graph.node_count == 12, "Optimized node count mismatch"
