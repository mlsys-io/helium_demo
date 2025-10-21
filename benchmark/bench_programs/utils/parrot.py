from collections.abc import Generator
from contextlib import contextmanager
from inspect import Parameter, Signature
from typing import Any, TypeVar

import requests
from bench_programs.utils.common import try_start_benchmark
from parrot import P
from parrot.frontend.pfunc.interface import SemanticFunction, SemanticVariable
from prometheus_client import parser

from helium import envs
from helium.common import GenerationConfig
from helium.runtime.llm.utils import get_tokenizer
from helium.runtime.protocol import HeliumSystemProfile
from helium.runtime.utils.vllm.utils import (
    get_metric_values,
    request_stop_benchmark,
    strip_v1_suffix,
)

T = TypeVar("T")


class ParrotMixin:
    def __init__(self, host: str, port: int) -> None:
        super().__init__()
        self._vm = P.VirtualMachine(
            core_http_addr=f"http://{host}:{port}", mode="release"
        )

    @contextmanager
    def get_vm(self) -> Generator[P.VirtualMachine]:
        try:
            self._vm.set_global_env()
            yield self._vm
        finally:
            self._vm.unset_global_env()


def parrot_sampling_config(generation_config: GenerationConfig) -> P.SamplingConfig:
    def get_or_default(value: T | None, default: T) -> T:
        return value if value is not None else default

    return P.SamplingConfig(
        temperature=get_or_default(generation_config.temperature, 1.0),
        top_p=get_or_default(generation_config.top_p, 1.0),
        max_gen_length=get_or_default(generation_config.max_tokens, 512),
        ignore_tokenizer_eos=generation_config.ignore_eos,
        stop_str=(
            generation_config.stop if isinstance(generation_config.stop, str) else None
        ),
        presence_penalty=get_or_default(generation_config.presence_penalty, 0.0),
        frequency_penalty=get_or_default(generation_config.frequency_penalty, 0.0),
        n=get_or_default(generation_config.n, 1),
        logit_bias=generation_config.logit_bias,
    )


class _DummyFunc:
    def __init__(self, name: str, doc: str, signature: Signature):
        self.__name__ = name
        self.__doc__ = doc
        self.__signature__ = signature

    def __call__(self, *args, **kwargs) -> None:
        pass


def parrot_semantic_function(
    vm: P.VirtualMachine,
    func_name: str,
    model: str,
    messages: list[dict[str, str]],
    **kwargs: Any,
) -> SemanticFunction:
    tokenizer = get_tokenizer(model)
    formatted_messages = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=False,
        enable_thinking=envs.HELIUM_VLLM_ENABLE_THINKING,
    )
    assert isinstance(formatted_messages, str)
    if model.startswith("Qwen/"):
        # Remove thinking tokens
        last_message = messages[-1]
        if last_message["role"] == "assistant":
            i = formatted_messages.rfind("</think>")
            if i >= 0:
                formatted_messages = formatted_messages[: i + len("</think>\n\n")]
            formatted_messages += last_message["content"]
    else:
        bos_token = tokenizer.special_tokens_map.get("bos_token")
        if bos_token is not None:
            assert isinstance(bos_token, str)
            formatted_messages = formatted_messages.removeprefix(bos_token)

    parameters = [
        Parameter(
            name=name, kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=annotation
        )
        for name, annotation in kwargs.items()
    ]

    dummy_func = _DummyFunc(
        name=func_name, doc=formatted_messages, signature=Signature(parameters)
    )
    parrot_func = P.semantic_function(formatter=None, model_type="text")(dummy_func)

    vm.register_function_handler(parrot_func)

    return parrot_func


def parrot_semantic_variable(
    func: SemanticFunction, **kwargs: SemanticVariable
) -> SemanticVariable:
    ret = func(**kwargs)  # type: ignore
    assert isinstance(ret, SemanticVariable)
    return ret


def parrot_start_benchmark(vm: P.VirtualMachine) -> None:
    try_start_benchmark(vm.core_http_addr)


def parrot_stop_benchmark(vm: P.VirtualMachine) -> HeliumSystemProfile:
    base_url = strip_v1_suffix(vm.core_http_addr)
    # Stop benchmarking
    bench: dict[str, dict[str, Any]]
    try:
        bench = request_stop_benchmark(base_url)
    except Exception:
        bench = {}

    # Fetch engine metrics
    engine_metrics: dict[str, str]
    try:
        res = requests.get(f"{base_url}/metrics")
        res.raise_for_status()
        engine_metrics = res.json()
    except Exception:
        engine_metrics = {}

    # Parse engine metrics
    for key, metric_str in engine_metrics.items():
        parsed = list(parser.text_string_to_metric_families(metric_str))
        metrics = get_metric_values(parsed)
        if key in bench:
            bench[key].update(metrics)
        else:
            bench[key] = metrics

    # Create profiling results
    system_profile: HeliumSystemProfile = {"llm_benchmark": {"parrot": bench}}

    return system_profile
