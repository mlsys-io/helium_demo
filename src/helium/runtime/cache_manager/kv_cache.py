import multiprocessing as mp
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import TYPE_CHECKING

import uvicorn
from lmcache.v1.api_server.__main__ import create_app as create_controller_app

if TYPE_CHECKING:
    from lmcache.config import LMCacheEngineConfig as LMCacheConfig
else:
    from lmcache.v1.config import LMCacheEngineConfig as LMCacheConfig

from helium import envs
from helium.runtime.utils.logger import Logger, LogLevel, init_child_logger
from helium.utils import http_client


def read_lmcache_config(config_file: str | Path | None) -> LMCacheConfig:
    if config_file is None:
        return LMCacheConfig.from_env()
    if isinstance(config_file, str):
        return LMCacheConfig.from_file(config_file)
    return LMCacheConfig.from_file(str(config_file))


def _parse_controller_url(url: str) -> tuple[str, int, int]:
    host, monitor_port_str = url.split(":")
    monitor_port = int(monitor_port_str)
    port = monitor_port - 1
    return host, port, monitor_port


def _run_lmcache_controller(
    host: str, port: int, monitor_port: int, ready_event: Event
) -> None:
    app = create_controller_app(f"{host}:{monitor_port}")
    ready_event.set()
    uvicorn.run(app, host=host, port=port, log_level=envs.LMCACHE_LOG_LEVEL.lower())


class KVCacheManager:
    def __init__(
        self,
        config_file: str | None,
        use_lmcache: bool,
        name: str = "KVCacheManager",
        logger: Logger | None = None,
        log_level: LogLevel | None = None,
    ) -> None:
        self.name = name
        self.logger = init_child_logger(name, logger, log_level)

        self._use_lmcache = use_lmcache
        self._config = read_lmcache_config(config_file)
        self._controller_url: str = self._config.controller_url  # type: ignore
        assert isinstance(self._controller_url, str)

        self._controller_client = None
        self._controller_proc = None

    def start(self):
        if self.is_started:
            return

        controller_host, controller_port, monitor_port = _parse_controller_url(
            self._controller_url
        )
        self._controller_client = KVCacheClient(
            controller_host, controller_port, not self._use_lmcache
        )

        if self._use_lmcache:
            ready_event = mp.Event()
            # Start the LMCache controller
            self._controller_proc = mp.Process(
                target=_run_lmcache_controller,
                args=(controller_host, controller_port, monitor_port, ready_event),
                daemon=True,
            )
            self._controller_proc.start()
            ready_event.wait()

        self.logger.debug("%s started", self.name)

    def stop(self) -> None:
        if not self.is_started:
            self.logger.warning("%s is not started", self.name)
            return

        self._controller_client = None

        if self._controller_proc is not None:
            self._controller_proc.terminate()
            self._controller_proc.join()
            self._controller_proc = None

        self.logger.debug("%s terminated", self.name)

    @property
    def is_started(self) -> bool:
        return self._controller_client is not None

    @property
    def controller_client(self) -> "KVCacheClient":
        if self._controller_client is None:
            raise RuntimeError(f"{self.name} is not started")
        return self._controller_client

    async def clear_cache(self, instance_id: str) -> None:
        await self.controller_client.clear(instance_id)


class KVCacheClient:
    def __init__(self, host: str, port: int, disabled: bool) -> None:
        self._url = f"http://{host}:{port}"
        self._disabled = disabled

    async def pin(
        self, instance_id: str, token_ids: list[int], location: str = "LocalCPUBackend"
    ) -> None:
        if self._disabled:
            return

        json_obj = {
            "tokens": token_ids,
            "instance_id": instance_id,
            "location": location,
        }
        await http_client.apost_json(self._url + "/pin", json=json_obj)

    async def clear(self, instance_id: str, location: str = "LocalCPUBackend") -> None:
        if self._disabled:
            return

        json_obj = {
            "instance_id": instance_id,
            "location": location,
        }
        await http_client.apost_json(self._url + "/clear", json=json_obj)
