import random
from typing import TypeVar

import httpx

from helium.runtime.protocol import HeliumSystemProfile
from helium.runtime.utils.vllm.utils import (
    get_metric_values,
    request_metrics,
    request_start_benchmark,
    request_stop_benchmark,
)

RANDOM_SEED = 42
UNLIMITED: int = 9999999
DEFAULT_TIMEOUT = httpx.Timeout(timeout=6000, connect=60)
DEFAULT_LIMITS = httpx.Limits(
    max_connections=UNLIMITED, max_keepalive_connections=UNLIMITED
)

T = TypeVar("T")


def random_shuffle(
    lst: list[T], inplace: bool = True, seed: int | None = RANDOM_SEED
) -> list[T]:
    if not inplace:
        lst = lst.copy()
    random.Random(seed).shuffle(lst)
    return lst


def try_start_benchmark(base_url: str) -> None:
    try:
        request_start_benchmark(base_url)
    except Exception:
        pass


def _try_request_metrics(base_url: str) -> list:
    try:
        return request_metrics(base_url)
    except Exception:
        return []


def try_stop_benchmark(base_url: str) -> HeliumSystemProfile:
    # Stop benchmarking
    try:
        bench = request_stop_benchmark(base_url)
    except Exception:
        bench = {}

    # Fetch engine metrics
    metrics = get_metric_values(_try_request_metrics(base_url))

    # Create profiling results
    inner = bench | metrics
    system_profile: HeliumSystemProfile = {"llm_benchmark": {"openai": {"none": inner}}}

    return system_profile
