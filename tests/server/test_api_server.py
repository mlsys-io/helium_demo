import pytest
from httpx import AsyncClient

from helium.runtime.protocol import HeliumRequestConfig, SystemProfilingConfig
from tests.utils import english_writing, guessing_game
from tests.utils.api_server import invoke_api_server


@pytest.mark.asyncio
@pytest.mark.timeout(5)
@pytest.mark.parametrize("num_generations", [100])
async def test_english_writing(
    mock_api_server_client: AsyncClient, num_generations: int
):
    graph = english_writing.build_graph(num_generations)
    request_config = HeliumRequestConfig(
        enable_cache_aware_scheduling=True,
        system_profiling_config=SystemProfilingConfig(),
    )
    response = await invoke_api_server(mock_api_server_client, graph, request_config)
    assert not response.error_info, "An error occurred"
    english_writing.check_outputs(response.outputs, num_generations)


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_guessing_game(mock_api_server_client: AsyncClient):
    graph = guessing_game.build_graph()
    request_config = HeliumRequestConfig(
        enable_cache_aware_scheduling=True,
        system_profiling_config=SystemProfilingConfig(),
    )
    with pytest.raises(ValueError):
        await invoke_api_server(mock_api_server_client, graph, request_config)
