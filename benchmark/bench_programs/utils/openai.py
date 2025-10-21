from collections.abc import Iterable
from typing import Any, TypeVar

import backoff
import httpx
from bench_programs.utils.common import DEFAULT_LIMITS, DEFAULT_TIMEOUT
from openai import APITimeoutError, AsyncOpenAI
from openai.types.chat import ChatCompletion

from helium.common import GenerationConfig, Message
from helium.utils import iter_batch

T = TypeVar("T")

BATCH_SIZE: int | None = 1


def prepare_openai(
    generation_config: GenerationConfig | None,
) -> tuple[AsyncOpenAI, GenerationConfig]:
    if generation_config is None:
        generation_config = GenerationConfig.from_env()

    http_client = httpx.AsyncClient(
        base_url=generation_config.base_url,
        timeout=DEFAULT_TIMEOUT,
        limits=DEFAULT_LIMITS,
    )
    client = AsyncOpenAI(
        api_key=generation_config.api_key,
        base_url=generation_config.base_url,
        http_client=http_client,
    )

    return client, generation_config


@backoff.on_exception(
    backoff.constant, exception=APITimeoutError, jitter=None, interval=0
)
async def openai_generate_async(
    client: AsyncOpenAI, messages: list[Message], generation_kwargs: dict[str, Any]
) -> str:
    response: ChatCompletion = await client.chat.completions.create(
        messages=[msg.to_dict() for msg in messages], **generation_kwargs  # type: ignore
    )
    output = response.choices[0].message.content
    assert isinstance(output, str)
    return output


def openai_iter_batch(
    inputs: Iterable[T], batch_size: int | None
) -> Iterable[Iterable[T]]:
    if batch_size is None:
        yield inputs
    else:
        for batch in iter_batch(inputs, batch_size):
            yield batch
