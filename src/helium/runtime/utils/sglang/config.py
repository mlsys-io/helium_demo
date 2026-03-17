from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal


@dataclass
class SGLangServerConfig:
    model: str
    host: str = "127.0.0.1"
    port: int = 30000

    device: str | None = None
    tp_size: int | None = None
    pp_size: int | None = None
    dp_size: int | None = None
    base_gpu_id: int | None = None
    gpu_id_step: int | None = None

    api_key: str | None = None
    enable_metrics: bool | None = None
    log_level: str | None = None
    log_level_http: str | None = None

    context_length: int | None = None
    max_running_requests: int | None = None
    chunked_prefill_size: int | None = None
    max_prefill_tokens: int | None = None
    disable_radix_cache: bool | None = None

    schedule_policy: Literal["lpm", "random", "fcfs", "dfs-weight", "lof"] | None = None
    enable_hierarchical_cache: bool | None = None
    hicache_ratio: float | None = None
    hicache_size: int | None = None
    hicache_write_policy: (
        Literal["write_back", "write_through", "write_through_selective"] | None
    ) = None

    disable_prefetch: bool | None = None
    disable_lr_pf: bool | None = None
    disable_kv_pf: bool | None = None
    kv_pf_reserve_tokens: int | None = None

    extra_args: list[str] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def to_cli_args(self) -> list[str]:
        args: list[str] = [
            "--model-path",
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]

        def _add(flag: str, value: Any) -> None:
            if value is None:
                return
            args.extend([flag, str(value)])

        _add("--device", self.device)
        _add("--tp-size", self.tp_size)
        _add("--pp-size", self.pp_size)
        _add("--dp-size", self.dp_size)
        _add("--base-gpu-id", self.base_gpu_id)
        _add("--gpu-id-step", self.gpu_id_step)
        _add("--api-key", self.api_key)
        _add("--log-level", self.log_level)
        _add("--log-level-http", self.log_level_http)
        _add("--context-length", self.context_length)
        _add("--max-running-requests", self.max_running_requests)
        _add("--chunked-prefill-size", self.chunked_prefill_size)
        _add("--max-prefill-tokens", self.max_prefill_tokens)
        _add("--schedule-policy", self.schedule_policy)
        _add("--hicache-ratio", self.hicache_ratio)
        _add("--hicache-size", self.hicache_size)
        _add("--hicache-write-policy", self.hicache_write_policy)
        _add("--kv-pf-reserve-tokens", self.kv_pf_reserve_tokens)

        if self.enable_metrics:
            args.append("--enable-metrics")
        if self.disable_radix_cache:
            args.append("--disable-radix-cache")
        if self.enable_hierarchical_cache:
            args.append("--enable-hierarchical-cache")
        if self.disable_prefetch:
            args.append("--disable-prefetch")
        if self.disable_lr_pf:
            args.append("--disable-lr-pf")
        if self.disable_kv_pf:
            args.append("--disable-kv-pf")

        args.extend(self.extra_args)
        return args


@dataclass
class SGLangRouterConfig:
    host: str = "127.0.0.1"
    port: int = 30000
    policy: str | None = None
    worker_urls: list[str] = field(default_factory=list)
    api_key: str | None = None
    log_level: str | None = None

    extra_args: list[str] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def replace(self, **kwargs: Any) -> "SGLangRouterConfig":
        return replace(self, **kwargs)

    def to_cli_args(self) -> list[str]:
        args: list[str] = ["--host", self.host, "--port", str(self.port)]
        if self.worker_urls:
            args.extend(["--worker-urls", *self.worker_urls])
        if self.policy is not None:
            args.extend(["--policy", self.policy])
        if self.api_key is not None:
            args.extend(["--api-key", self.api_key])
        if self.log_level is not None:
            args.extend(["--log-level", self.log_level])
        args.extend(self.extra_args)
        return args


@dataclass
class SGLangWorkerProcessConfig:
    server: SGLangServerConfig
    env: dict[str, str] = field(default_factory=dict)
    log_file: Path | None = None
