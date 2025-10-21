from dataclasses import dataclass

from helium import envs
from helium.runtime.cache_manager.kv_cache import KVCacheManager
from helium.runtime.cache_manager.prompt_cache import PromptCacheManager


@dataclass
class CacheManagerConfig:
    # KV cache config
    enable_proactive_kv_cache: bool = True
    kv_cache_config_file: str | None = envs.LMCACHE_CONFIG_FILE
    kv_cache_manager: KVCacheManager | None = None
    # Prompt cache config
    enable_prompt_cache: bool = True
    prompt_cache_manager: PromptCacheManager | None = None
