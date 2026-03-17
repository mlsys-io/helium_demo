import httpx
from bench_programs.utils.common import DEFAULT_LIMITS, DEFAULT_TIMEOUT
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from helium.common import GenerationConfig


def get_langgraph_openai_client(generation_config: GenerationConfig) -> ChatOpenAI:
    http_client = httpx.Client(
        base_url=generation_config.base_url,
        timeout=DEFAULT_TIMEOUT,
        limits=DEFAULT_LIMITS,
    )
    http_async_client = httpx.AsyncClient(
        base_url=generation_config.base_url,
        timeout=DEFAULT_TIMEOUT,
        limits=DEFAULT_LIMITS,
    )
    return ChatOpenAI(
        http_client=http_client,
        http_async_client=http_async_client,
        model=generation_config.model,
        base_url=generation_config.base_url,
        api_key=SecretStr(generation_config.api_key),
        temperature=generation_config.temperature,
        presence_penalty=generation_config.presence_penalty,
        frequency_penalty=generation_config.frequency_penalty,
        seed=generation_config.seed,
        logprobs=bool(generation_config.logprobs),
        streaming=bool(generation_config.stream),
        n=generation_config.n,
        top_p=generation_config.top_p,
        max_completion_tokens=generation_config.max_tokens,
        stop_sequences=generation_config.stop,
    )
