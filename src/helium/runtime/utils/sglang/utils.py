from typing import Any

from httpx import Response
from openai import AsyncOpenAI, OpenAI
from prometheus_client import Metric, parser

from helium.utils import http_client

SUPPORTED_ENGINE_METRICS: list[str] = [
    "total_retracted_reqs",
    "cache_hit_rate",
    "prompt_tokens",
    "cached_tokens",
]


def _parse_metric_response(
    res: Response | http_client.SyncHTTPResponse,
) -> list[Metric]:
    res.raise_for_status()
    metrics = list(parser.text_string_to_metric_families(res.text))
    return metrics


async def _parse_metric_response_async(
    res: http_client.AsyncHTTPResponse,
) -> list[Metric]:
    res.raise_for_status()
    text = await res.text()
    metrics = list(parser.text_string_to_metric_families(text))
    return metrics


def strip_v1_suffix(base_url: str) -> str:
    base_url = base_url.strip().rstrip("/").removesuffix("/v1")
    return base_url


def request_metrics(base_url: str) -> list[Metric]:
    base_url = strip_v1_suffix(base_url)
    res = http_client.get(f"{base_url}/metrics")
    return _parse_metric_response(res)


def request_metrics_openai(client: OpenAI) -> list[Metric]:
    res = client.get("/metrics", cast_to=Response)
    return _parse_metric_response(res)


async def async_request_metrics(base_url: str) -> list[Metric]:
    base_url = strip_v1_suffix(base_url)
    res = await http_client.aget(f"{base_url}/metrics")
    return await _parse_metric_response_async(res)


async def async_request_metrics_openai(client: AsyncOpenAI) -> list[Metric]:
    res = await client.get("/metrics", cast_to=Response)
    return _parse_metric_response(res)


def get_metric_values(
    engine_metrics: list[Metric], sglang_metrics: list[str] | None = None
) -> dict[str, Any]:
    if sglang_metrics is None:
        sglang_metrics = SUPPORTED_ENGINE_METRICS

    sglang_prefix = "sglang:"
    metric_map = {
        m.name[len(sglang_prefix) :]: m
        for m in engine_metrics
        if m.name.startswith(sglang_prefix)
    }

    def _get_value(name: str) -> float | None:
        if name not in metric_map:
            return None
        samples = metric_map[name].samples
        if not samples:
            return None
        return float(samples[0].value)

    ret: dict[str, Any] = {}

    # Prefix-cache related:
    prompt_tokens_total = _get_value("prompt_tokens")
    cached_tokens_total = _get_value("cached_tokens")
    if prompt_tokens_total is not None:
        ret["gpu_prefix_cache_queries"] = prompt_tokens_total
    if cached_tokens_total is not None:
        ret["gpu_prefix_cache_hits"] = cached_tokens_total
    if prompt_tokens_total is not None and prompt_tokens_total > 0:
        if cached_tokens_total is None:
            cached_tokens_total = 0.0
        ret["gpu_prefix_cache_hit_rate"] = cached_tokens_total / prompt_tokens_total
    else:
        cache_hit_rate = _get_value("cache_hit_rate")
        if cache_hit_rate is not None:
            ret["gpu_prefix_cache_hit_rate"] = cache_hit_rate

    # Preemption-like metric:
    total_retracted_reqs = _get_value("total_retracted_reqs")
    if total_retracted_reqs is not None:
        ret["num_preemptions"] = total_retracted_reqs
        ret["num_preemptions_total"] = total_retracted_reqs

    return ret


def request_start_benchmark(base_url: str) -> None:
    base_url = strip_v1_suffix(base_url)
    res = http_client.get(f"{base_url}/start_benchmark")
    res.raise_for_status()


def request_start_benchmark_openai(client: OpenAI) -> None:
    res = client.get("/start_benchmark", cast_to=Response)
    res.raise_for_status()


async def async_request_start_benchmark(base_url: str) -> None:
    base_url = strip_v1_suffix(base_url)
    res = await http_client.aget(f"{base_url}/start_benchmark")
    res.raise_for_status()


async def async_request_start_benchmark_openai(client: AsyncOpenAI) -> None:
    res = await client.get("/start_benchmark", cast_to=Response)
    res.raise_for_status()


def request_stop_benchmark(base_url: str) -> dict[str, Any]:
    base_url = strip_v1_suffix(base_url)
    res = http_client.get(f"{base_url}/stop_benchmark")
    res.raise_for_status()
    return res.json()


def request_stop_benchmark_openai(client: OpenAI) -> dict[str, Any]:
    res = client.get("/stop_benchmark", cast_to=Response)
    res.raise_for_status()
    return res.json()


async def async_request_stop_benchmark(base_url: str) -> dict[str, Any]:
    base_url = strip_v1_suffix(base_url)
    return await http_client.aget_json(f"{base_url}/stop_benchmark")


async def async_request_stop_benchmark_openai(client: AsyncOpenAI) -> dict[str, Any]:
    res = await client.get("/stop_benchmark", cast_to=Response)
    res.raise_for_status()
    return res.json()


def request_flush_cache(base_url: str) -> None:
    base_url = strip_v1_suffix(base_url)
    res = http_client.post(f"{base_url}/flush_cache")
    res.raise_for_status()


def request_flush_cache_openai(client: OpenAI) -> None:
    res = client.post("/flush_cache", cast_to=Response)
    res.raise_for_status()


async def async_request_flush_cache(base_url: str) -> None:
    base_url = strip_v1_suffix(base_url)
    res = await http_client.apost(f"{base_url}/flush_cache")
    res.raise_for_status()


async def async_request_flush_cache_openai(client: AsyncOpenAI) -> None:
    res = await client.post("/flush_cache", cast_to=Response)
    res.raise_for_status()


def request_abort_all(base_url: str) -> None:
    base_url = strip_v1_suffix(base_url)
    res = http_client.post(
        f"{base_url}/abort_request",
        json={"rid": "", "abort_all": True},
    )
    res.raise_for_status()


async def async_request_abort_all(base_url: str) -> None:
    base_url = strip_v1_suffix(base_url)
    res = await http_client.apost(
        f"{base_url}/abort_request",
        json={"rid": "", "abort_all": True},
    )
    res.raise_for_status()


def request_update_agent_step_graph(
    base_url: str,
    agent_data: dict[str, Any],
    timestep_data: dict[int, list[str]],
    timestep_cnt: int,
) -> None:
    base_url = base_url.strip().rstrip("/")
    res = http_client.post(
        f"{base_url}/update",
        json={
            "agent_data": agent_data,
            "timestep_data": timestep_data,
            "timestep_cnt": timestep_cnt,
        },
    )
    res.raise_for_status()


async def async_request_update_agent_step_graph(
    base_url: str,
    agent_data: dict[str, Any],
    timestep_data: dict[int, list[str]],
    timestep_cnt: int,
) -> None:
    base_url = base_url.strip().rstrip("/")
    res = await http_client.apost(
        f"{base_url}/update",
        json={
            "agent_data": agent_data,
            "timestep_data": timestep_data,
            "timestep_cnt": timestep_cnt,
        },
    )
    res.raise_for_status()
