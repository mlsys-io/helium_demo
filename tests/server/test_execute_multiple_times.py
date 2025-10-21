import pytest

from helium import helium, ops
from helium.runtime import HeliumServer
from helium.runtime.protocol import HeliumRequestConfig, SystemProfilingConfig
from tests.utils import guessing_game, while_loop


@pytest.mark.timeout(5)
def test_no_while_loop(mock_helium_server: HeliumServer):
    graph = guessing_game.build_graph()
    for _ in range(3):
        outputs = helium.invoke(
            graph,
            HeliumRequestConfig(
                enable_cache_aware_scheduling=True,
                system_profiling_config=SystemProfilingConfig(),
            ),
        )
        guessing_game.check_outputs(outputs)


def run_while_loop(loop_op: ops.Op):
    out = helium.invoke(loop_op)
    expected = ["Hell", "", "Hiasdfasdfasdfasdfasfdd"]

    assert out == expected, f"Expected {expected}, got {out}"


@pytest.mark.timeout(5)
def test_with_while_loop(mock_helium_server: HeliumServer):
    graph = while_loop.build_graph()
    for _ in range(3):
        outputs = helium.invoke(graph)
        while_loop.check_outputs(outputs)
