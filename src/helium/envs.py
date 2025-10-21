import os
from typing import Literal

from dotenv import find_dotenv, load_dotenv

env_path = find_dotenv()
if env_path:
    load_dotenv(env_path)
else:
    raise FileNotFoundError("No .env file found")


def _get_deployment_mode() -> Literal["dev", "prod"]:
    mode = os.environ.get("HELIUM_DEPLOYMENT_MODE", "dev").lower()
    if mode not in ("dev", "prod"):
        raise ValueError("HELIUM_DEPLOYMENT_MODE must be 'dev' or 'prod'")
    return mode  # type: ignore[return-value]


HELIUM_LOG_LEVEL: str = os.environ.get("HELIUM_LOG_LEVEL", "INFO")
HELIUM_DEPLOYMENT_MODE: Literal["dev", "prod"] = _get_deployment_mode()

HELIUM_SERVER_HOST: str = os.environ["HELIUM_SERVER_HOST"]
HELIUM_SERVER_PORT: int = int(os.environ["HELIUM_SERVER_PORT"])
HELIUM_LLM_SERVICES: list[str] = os.environ["HELIUM_LLM_SERVICES"].split(",")

LLM_SERVICE: str | None = os.environ.get("LLM_SERVICE")
LLM_API_KEY: str = os.environ.get("LLM_API_KEY", "EMPTY")
LLM_BASE_URL: str = os.environ["LLM_BASE_URL"]
LLM_MODEL: str = os.environ["LLM_MODEL"]
LLM_MAX_TOKENS: int = int(os.environ.get("LLM_MAX_TOKENS", 300))

HELIUM_VLLM_MODEL: str = os.environ["HELIUM_VLLM_MODEL"]
HELIUM_VLLM_HOST: str = os.environ["HELIUM_VLLM_HOST"]
HELIUM_VLLM_PORT: int = int(os.environ["HELIUM_VLLM_PORT"])
HELIUM_VLLM_DEVICE: str = os.environ["HELIUM_VLLM_DEVICE"]
HELIUM_VLLM_LOGGING_LEVEL: str = os.environ["HELIUM_VLLM_LOGGING_LEVEL"]
HELIUM_VLLM_ENABLE_THINKING: bool = bool(
    int(os.environ.get("HELIUM_VLLM_ENABLE_THINKING", "0"))
)

CUDA_VISIBLE_DEVICES: str | None = os.environ.get("CUDA_VISIBLE_DEVICES")

LMCACHE_CONFIG_FILE: str = os.environ.get("LMCACHE_CONFIG_FILE", "configs/lmcache.yaml")
LMCACHE_LOG_LEVEL: str = os.environ.get("LMCACHE_LOG_LEVEL", "WARNING")

DEBUG_MODE: bool = "DEBUG_MODE" in os.environ
DEBUG_MOCK_LLM_ONLY: bool = False
DEBUG_MOCK_LLM_VERBOSE: bool = True
