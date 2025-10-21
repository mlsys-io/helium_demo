from functools import lru_cache

from helium import envs
from helium.common import Message
from vllm.vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.vllm.transformers_utils.tokenizer import get_tokenizer as vllm_get_tokenizer

_OLLAMA_TO_HF_MODELS = {
    # Llama 3
    "llama3:8b": "meta-llama/Meta-Llama-3-8B",
    "llama3:70b": "meta-llama/Meta-Llama-3-70B",
    # Llama 3.1
    "llama3.1:8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.1:70b": "meta-llama/Llama-3.1-70B-Instruct",
    "llama3.1:405b": "meta-llama/Llama-3.1-405B-Instruct",
    # Llama 3.2
    "llama3.2:1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
    # Qwen 2.5
    "qwen2.5:0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen2.5:3b": "Qwen/Qwen2.5-3B",
    "qwen2.5:7b": "Qwen/Qwen2.5-7B",
    "qwen2.5:14b": "Qwen/Qwen2.5-14B",
    "qwen2.5:32b": "Qwen/Qwen2.5-32B",
    "qwen2.5:72b": "Qwen/Qwen2.5-72B",
    "qwen2.5-coder:0.5b": "Qwen/Qwen2.5-Coder-0.5B",
    "qwen2.5-coder:1.5b": "Qwen/Qwen2.5-Coder-1.5B",
    "qwen2.5-coder:3b": "Qwen/Qwen2.5-Coder-3B",
    "qwen2.5-coder:7b": "Qwen/Qwen2.5-Coder-7B",
    "qwen2.5-coder:14b": "Qwen/Qwen2.5-Coder-14B",
    "qwen2.5-coder:32b": "Qwen/Qwen2.5-Coder-32B",
    # Qwen 3
    "qwen3:0.6b": "Qwen/Qwen3-0.6B",
    "qwen3:1.7b": "Qwen/Qwen3-1.7B",
    "qwen3:4b": "Qwen/Qwen3-4B",
    "qwen3:8b": "Qwen/Qwen3-8B",
    "qwen3:14b": "Qwen/Qwen3-14B",
    "qwen3:30b": "Qwen/Qwen3-30B",
    "qwen3:32b": "Qwen/Qwen3-32B",
    "qwen3:235b": "Qwen/Qwen3-235B",
}


@lru_cache
def get_tokenizer(model_name: str) -> AnyTokenizer:
    if "/" not in model_name:
        # Handle Ollama's model name
        if model_name in _OLLAMA_TO_HF_MODELS:
            model_name = _OLLAMA_TO_HF_MODELS[model_name]
        else:
            raise ValueError("Unknown model name: {}".format(model_name))
    return vllm_get_tokenizer(model_name)


def apply_chat_template(
    tokenizer_or_name: str | AnyTokenizer,
    messages_list: list[list[Message]],
    strip_begin_of_text: bool = True,
) -> list[str]:
    tokenizer = (
        get_tokenizer(tokenizer_or_name)
        if isinstance(tokenizer_or_name, str)
        else tokenizer_or_name
    )
    tokenized_list: list[str] = tokenizer.apply_chat_template(  # type: ignore
        [[message.to_dict() for message in messages] for messages in messages_list],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
        enable_thinking=envs.HELIUM_VLLM_ENABLE_THINKING,
    )
    if strip_begin_of_text:
        bos_token = tokenizer.special_tokens_map.get("bos_token")
        if bos_token is not None:
            assert isinstance(bos_token, str)
            for i, formatted_messages in enumerate(tokenized_list):
                tokenized_list[i] = formatted_messages.removeprefix(bos_token)
    return tokenized_list
