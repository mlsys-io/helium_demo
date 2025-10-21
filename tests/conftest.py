from collections.abc import AsyncGenerator, Generator
from contextlib import contextmanager

import pytest
import pytest_asyncio
from httpx import AsyncClient

from helium import helium
from helium.runtime import HeliumServer, HeliumServerConfig
from helium.runtime.cache_manager import CacheManagerConfig
from helium.runtime.llm import LLMServiceConfig, LLMServiceInfo
from tests.utils.api_server import api_server_client


@pytest.fixture(scope="function")
def started_helium_server() -> Generator[HeliumServer, None, None]:
    config = HeliumServerConfig(
        is_local=True,
        llm_service_configs=[
            LLMServiceConfig(
                name="vllm-local",
                args={
                    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
                    "device": "cuda:0",
                    "enable_prefix_caching": True,
                    "num_gpu_blocks_override": None,
                },
            )
        ],
    )
    with helium.serve_instance(config=config) as server:
        yield server


@contextmanager
def helium_server_context(
    benchmarking: bool, num_llm_workers: int
) -> Generator[HeliumServer, None, None]:
    config = HeliumServerConfig(
        is_local=True,
        benchmarking=benchmarking,
        llm_service_configs=[
            LLMServiceConfig(
                name="mock",
                args={"verbose": False},
                info=LLMServiceInfo(accumulation_window=0, max_accumulation_time=0),
            )
        ]
        * num_llm_workers,
        cache_manager_config=CacheManagerConfig(
            enable_proactive_kv_cache=False, enable_prompt_cache=False
        ),
    )
    with helium.serve_instance(config=config) as server:
        yield server


@pytest.fixture(scope="function")
def non_benchmarking_server_1() -> Generator[HeliumServer, None, None]:
    with helium_server_context(benchmarking=False, num_llm_workers=1) as server:
        yield server


@pytest.fixture(scope="function")
def non_benchmarking_server_2() -> Generator[HeliumServer, None, None]:
    with helium_server_context(benchmarking=False, num_llm_workers=2) as server:
        yield server


@pytest.fixture(scope="function")
def benchmarking_server_1() -> Generator[HeliumServer, None, None]:
    with helium_server_context(benchmarking=True, num_llm_workers=1) as server:
        yield server


@pytest.fixture(scope="function")
def benchmarking_server_2() -> Generator[HeliumServer, None, None]:
    with helium_server_context(benchmarking=True, num_llm_workers=2) as server:
        yield server


@pytest.fixture(
    scope="function",
    params=[
        "non_benchmarking_server_1",
        "non_benchmarking_server_2",
        "benchmarking_server_1",
        "benchmarking_server_2",
    ],
)
def mock_helium_server(request) -> Generator[HeliumServer, None, None]:
    return request.getfixturevalue(request.param)


@pytest_asyncio.fixture(scope="function")
async def mock_api_server_client() -> AsyncGenerator[AsyncClient, None]:
    with helium_server_context(benchmarking=False, num_llm_workers=1) as server:
        async with api_server_client(server) as client:
            yield client
