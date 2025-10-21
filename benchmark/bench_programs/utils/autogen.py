from typing import Any

from autogen import AssistantAgent

from helium.common import GenerationConfig

_reply_func_to_exclude = [
    AssistantAgent.check_termination_and_human_reply,
    AssistantAgent.a_check_termination_and_human_reply,
]


async def autogen_generate_async(
    agent: AssistantAgent, messages: list[dict[str, str]]
) -> str:
    reply = await agent.a_generate_reply(messages, exclude=_reply_func_to_exclude)
    if isinstance(reply, str):
        content = reply
    elif isinstance(reply, dict):
        content = reply["content"]
    else:
        raise ValueError("Invalid agent reply type")
    return content


def autogen_get_llm_config(
    generation_config: GenerationConfig | None,
) -> tuple[str, dict[str, Any]]:
    """
    Returns
    -------
    base_url: str
        The base URL for the LLM service.
    config_list: list[dict[str, Any]]
        A list of configuration dictionaries for the LLM.
    """
    if generation_config is None:
        generation_config = GenerationConfig.from_env()

    base_url = generation_config.base_url
    if base_url is None:
        raise ValueError("Base URL is not set")

    config_list = [
        {
            "model": generation_config.model,
            "base_url": generation_config.base_url,
            "api_type": "openai",
            "api_key": generation_config.api_key,
            "temperature": generation_config.temperature or 0,
            "max_tokens": generation_config.max_tokens,
            "max_completion_tokens": generation_config.max_tokens,
            "price": [0, 0],
        }
    ]
    llm_config = {"cache_seed": None, "config_list": config_list}

    return base_url, llm_config
