from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from helium import envs
from helium.constants import DEFAULT_MAX_NUM_BATCHED_TOKENS, DEFAULT_MAX_NUM_SEQS
from vllm.vllm.config import KVTransferConfig
from vllm.vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.vllm.utils import FlexibleArgumentParser


class CompiledServerConfig:
    __slots__ = ("args", "inner")

    def __init__(self, config: "VLLMServerConfig", args: Any = None):
        self.inner = config
        self.args = config.parse_args(args)


@dataclass
class VLLMServerConfig:
    model: str = envs.HELIUM_VLLM_MODEL
    host: str = envs.HELIUM_VLLM_HOST
    port: int = envs.HELIUM_VLLM_PORT
    lmcache_config_file: Path | None = None

    root_dir: Path = Path.cwd()
    device: str = envs.HELIUM_VLLM_DEVICE
    uvicorn_log_level: str = "warning"
    log_level: str = envs.HELIUM_VLLM_LOGGING_LEVEL
    reuse_addr: bool = True

    profiling: bool = False
    benchmarking: bool = False
    use_v1: bool = True
    mock: bool = False

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def trace_dir(self) -> Path:
        return self.root_dir / "traces"

    @property
    def cuda_device(self) -> str | None:
        if self.device == "cuda":
            return envs.CUDA_VISIBLE_DEVICES
        if self.device.startswith("cuda:"):
            return self.device.removeprefix("cuda:")
        return None

    @property
    def non_engine_args(self) -> list[str]:
        """Arguments that should not be passed to vLLM's LLMEngine"""
        return [
            # Helium args
            "host",
            "port",
            "uvicorn_log_level",
            "log_level",
            "reuse_addr",
            "profiling",
            "benchmarking",
            "use_v1",
            "mock",
            # vLLM server args
            "allow_credentials",
            "allowed_origins",
            "allowed_methods",
            "allowed_headers",
            "api_key",
            "lora_modules",
            "prompt_adapters",
            "chat_template",
            "chat_template_content_format",
            "response_role",
            "ssl_keyfile",
            "ssl_certfile",
            "ssl_ca_certs",
            "enable_ssl_refresh",
            "ssl_cert_reqs",
            "root_path",
            "middleware",
            "return_tokens_as_token_ids",
            "disable_frontend_multiprocessing",
            "enable_request_id_headers",
            "enable_auto_tool_choice",
            "tool_call_parser",
            "tool_parser_plugin",
            "disable_log_requests",
            "max_log_len",
            "disable_fastapi_docs",
            "enable_prompt_tokens_details",
            "enable_server_load_tracking",
        ]

    def default_args(self) -> Namespace:
        return Namespace(**self.default_args_dict())

    def default_args_dict(self) -> dict[str, Any]:
        return dict(
            model=self.model,
            host=self.host,
            port=self.port,
            device=self.device.split(":")[0],
            uvicorn_log_level=self.uvicorn_log_level,
            use_v1=self.use_v1,
        )

    def engine_args(self) -> dict[str, Any]:
        server_args = self.parse_args([]).__dict__
        for k in ["log_level", "profiling", "reuse_addr"]:
            server_args.pop(k, None)
        return server_args

    def parse_args(self, args: Any = None) -> Namespace:
        parser = FlexibleArgumentParser(
            description="vLLM OpenAI-Compatible RESTful API server."
        )
        parser = make_arg_parser(parser)

        parser.add_argument(
            "--log-level", default=self.log_level, help="Logging level."
        )
        parser.add_argument(
            "--profiling",
            action="store_true",
            default=self.profiling,
            help="Start the server in profiling mode. # Traces can be viewed in `https://ui.perfetto.dev/`",
        )
        parser.add_argument(
            "--benchmarking",
            action="store_true",
            default=self.benchmarking,
            help="Start the server in benchmarking mode.",
        )
        parser.add_argument(
            "--reuse-addr",
            action="store_true",
            default=self.reuse_addr,
            help="Enable SO_REUSEADDR for the server.",
        )

        return parser.parse_args(args, self.default_args())

    def compile(self, args: Any = None) -> CompiledServerConfig:
        return CompiledServerConfig(self, args)


@dataclass
class BenchVLLMServerConfig(VLLMServerConfig):
    log_file: Path | None = None

    max_model_len: int | None = 8192

    enable_prefix_caching: bool = False
    enable_chunked_prefill: bool = False
    kv_transfer_config: KVTransferConfig | None = None

    block_size: int | None = 16
    num_gpu_blocks_override: int | None = None

    # For reproducibility
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS
    max_num_batched_tokens: int = DEFAULT_MAX_NUM_BATCHED_TOKENS

    # For context length extension
    hf_overrides: dict[str, Any] | None = None

    @property
    def non_engine_args(self) -> list[str]:
        return super().non_engine_args + ["log_file"]

    def default_args_dict(self) -> dict[str, Any]:
        additional_args = dict(
            log_file=self.log_file,
            max_model_len=self.max_model_len,
            enable_prefix_caching=self.enable_prefix_caching,
            enable_chunked_prefill=self.enable_chunked_prefill,
            kv_transfer_config=self.kv_transfer_config,
            num_gpu_blocks_override=self.num_gpu_blocks_override,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            hf_overrides=self.hf_overrides,
        )
        if self.block_size is not None:
            additional_args["block_size"] = self.block_size
        return super().default_args_dict() | additional_args


@dataclass
class LocalVLLMServerConfig(BenchVLLMServerConfig):
    disable_log_requests: bool = True
    disable_log_stats: bool = True

    @property
    def non_engine_args(self) -> list[str]:
        return super().non_engine_args + ["disable_log_requests", "disable_log_stats"]

    def default_args_dict(self) -> dict[str, Any]:
        additional_args = dict(
            disable_log_requests=self.disable_log_requests,
            disable_log_stats=self.disable_log_stats and not self.benchmarking,
        )
        return super().default_args_dict() | additional_args
