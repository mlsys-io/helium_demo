import os
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cloudpickle as pickle
import httpx
from agentscope.agents import AgentBase
from agentscope.manager import ASManager
from agentscope.message import Msg
from agentscope.rpc import RpcObject
from agentscope.rpc.rpc_async import AsyncResult
from agentscope.rpc.rpc_client import RpcClient
from agentscope.rpc.rpc_config import DistConf
from agentscope.server import RpcAgentServerLauncher
from agentscope.server.async_result_pool import RedisPool
from bench_programs.utils.common import DEFAULT_LIMITS, DEFAULT_TIMEOUT, UNLIMITED
from openai import OpenAI

from helium.common import GenerationConfig

_AS_CACHE_DIR = str(
    Path(os.environ.get("AS_HOME_PATH", Path.home())) / ".cache" / "agentscope"
)
_AS_REDIS_URL = os.environ.get("AGENTSCOPE_REDIS_URL", "redis://localhost:6379")
redis_pool: RedisPool | None = None


def agentscope_model_config(
    model_name: str,
    api_key: str,
    base_url: str,
    model_type: str = "openai_chat",
    config_name: str = "vllm",
) -> dict[str, Any]:
    return {
        "model_type": model_type,
        "config_name": config_name,
        "model_name": model_name,
        "api_key": api_key,
        "client_args": {"base_url": base_url},
    }


def agentscope_reinit(model_configs: list[dict[str, Any]]) -> None:
    global redis_pool

    manager = ASManager.get_instance()
    manager.flush()
    manager.initialize(
        model_configs=model_configs,
        project=None,
        name=None,
        disable_saving=True,
        save_dir="./runs",
        save_log=False,
        save_code=False,
        save_api_invoke=False,
        cache_dir=_AS_CACHE_DIR,
        use_monitor=False,
        logger_level="WARNING",
        run_id=None,
        studio_url=None,
    )

    if redis_pool is None:
        redis_pool = RedisPool(_AS_REDIS_URL, max_expire=UNLIMITED)


def agentscope_reinit_from_config(config: GenerationConfig | None) -> GenerationConfig:
    if config is None:
        config = GenerationConfig.from_env()

    model_config = agentscope_model_config(
        config.model, config.api_key, config.base_url
    )
    agentscope_reinit([model_config])
    return config


class PlaceholderMsg:
    def __init__(self, result: AsyncResult) -> None:
        self._host = result._host
        self._port = result._port
        self._retry = result._retry
        task_id = result._task_id
        if task_id is None:
            task_id = int(result._get_task_id())
        self._task_id = task_id

        self._ready = result._ready
        self._role: str
        self._content: str

    def _fetch_result(self) -> None:
        assert redis_pool is not None
        result = redis_pool.get(self._task_id, timeout=UNLIMITED)
        data = pickle.loads(result)
        assert isinstance(data, Msg)
        assert isinstance(data.content, str)
        self._role = data.role
        self._content = data.content
        self._ready = True

    @property
    def role(self) -> str:
        if not self._ready:
            self._fetch_result()
        return self._role

    @property
    def content(self) -> str:
        if not self._ready:
            self._fetch_result()
        return self._content


class FormatMsg:
    def __init__(
        self,
        role: str,
        format_str: str,
        *args: str | PlaceholderMsg,
        **kwargs: str | PlaceholderMsg,
    ) -> None:
        self.role = role
        self.format_str = format_str
        self.args = args
        self.kwargs = kwargs
        self._content: str | None = None

    @property
    def content(self) -> str:
        if self._content is not None:
            return self._content

        arg: list[str] = [
            arg.content if isinstance(arg, PlaceholderMsg) else arg for arg in self.args
        ]
        kwargs: dict[str, str] = {
            k: (v.content if isinstance(v, PlaceholderMsg) else v)
            for k, v in self.kwargs.items()
        }
        content = self.format_str.format(*arg, **kwargs)
        self._content = content
        return content


class AgentScopeAgent(AgentBase):
    def __init__(
        self, name: str, sys_prompt: str, model_config_name: str = "vllm"
    ) -> None:
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
            use_memory=False,
        )
        client = self.model.client  # type: ignore
        assert isinstance(client, OpenAI)
        client._client = httpx.Client(
            base_url=client.base_url, timeout=DEFAULT_TIMEOUT, limits=DEFAULT_LIMITS
        )

    @classmethod
    def dist(
        cls,
        name: str,
        sys_prompt: str,
        model_config_name: str = "vllm",
        config: DistConf | None = None,
        max_retries: int = 3,
    ) -> RpcObject:
        # NOTE: This is a workaround to avoid typing issues due to the __call__ method
        agent: Any = cls(name, sys_prompt, model_config_name)  # type: ignore[args]
        host = "localhost"
        for _ in range(max_retries):
            # Launch the agent server
            launcher = RpcAgentServerLauncher(
                host=host,
                capacity=32,
                pool_type="redis",
                redis_url=_AS_REDIS_URL,
                max_expire_time=UNLIMITED,
                max_timeout_seconds=UNLIMITED,
                local_mode=True,
                custom_agent_classes=[cls],
            )
            launcher.launch()
            time.sleep(0.2)  # This prevents connection issues

            # Check if the server is alive
            client = RpcClient(host, launcher.port)
            if client.is_alive():
                break
            launcher.shutdown()
            launcher.wait_until_terminate()
        else:
            raise Exception("Failed to connect to RPC server")

        # Create the distributed agent
        if config is None:
            config = DistConf(
                host=host,
                port=launcher.port,
                max_expire_time=UNLIMITED,
                max_timeout_seconds=UNLIMITED,
            )
        obj: RpcObject = agent.to_dist(**config)
        obj.server_launcher = launcher
        obj._check_created()
        return obj

    def reply(
        self,
        messages: Sequence[Msg | FormatMsg | PlaceholderMsg],
        config: GenerationConfig,
    ) -> Msg:
        formatted = self._format(messages)
        response = self.model(formatted, **config.openai_kwargs())
        msg = Msg(self.name, response.text, role="assistant")
        return msg

    def _format(
        self, messages: Sequence[Msg | FormatMsg | PlaceholderMsg]
    ) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = (
            [{"role": "system", "content": self.sys_prompt}] if self.sys_prompt else []
        )
        formatted.extend(
            [{"role": msg.role, "content": msg.content} for msg in messages]
        )
        return formatted

    @classmethod
    def stop_all(cls, agents: Iterable[RpcObject]) -> None:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(agent.stop) for agent in agents]
            for future in futures:
                future.result()
            for agent in agents:
                if agent.server_launcher is not None:
                    agent.server_launcher.wait_until_terminate()
        # Clear the DB after every run
        assert redis_pool is not None
        redis_pool.pool.flushdb()


def agentscope_call_agent(
    agent: RpcObject,
    messages: Sequence[Msg | FormatMsg | PlaceholderMsg],
    generation_config: GenerationConfig,
) -> PlaceholderMsg:
    return PlaceholderMsg(agent(messages, generation_config))
