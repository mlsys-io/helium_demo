from typing import Any

from httpx import Response
from openai import AsyncOpenAI, OpenAI
from prometheus_client import Metric, parser

from helium.utils import http_client

SUPPORTED_ENGINE_METRICS: list[str] = [
    "num_preemptions",
    "num_preemptions_total",
    "gpu_prefix_cache_queries",
    "gpu_prefix_cache_hits",
    "gpu_prefix_cache_hit_rate",
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
    engine_metrics: list[Metric], vllm_metrics: list[str] | None = None
) -> dict[str, Any]:
    if vllm_metrics is None:
        vllm_metrics = SUPPORTED_ENGINE_METRICS

    vllm_prefix = "vllm:"
    metric_map = {
        m.name[len(vllm_prefix) :]: m
        for m in engine_metrics
        if m.name.startswith(vllm_prefix)
    }

    ret: dict[str, Any] = {}
    for metric in vllm_metrics:
        if metric not in metric_map:
            continue
        samples = metric_map[metric].samples
        if len(samples) > 0:
            ret[metric] = samples[0].value
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


def request_reset_prefix_cache(base_url: str) -> None:
    base_url = strip_v1_suffix(base_url)
    res = http_client.post(f"{base_url}/reset_prefix_cache")
    res.raise_for_status()


def request_reset_prefix_cache_openai(client: OpenAI) -> None:
    res = client.post("/reset_prefix_cache", cast_to=Response)
    res.raise_for_status()


async def async_request_reset_prefix_cache(base_url: str) -> None:
    base_url = strip_v1_suffix(base_url)
    res = await http_client.apost(f"{base_url}/reset_prefix_cache")
    res.raise_for_status()


async def async_request_reset_prefix_cache_openai(client: AsyncOpenAI) -> None:
    res = await client.post("/reset_prefix_cache", cast_to=Response)
    res.raise_for_status()
