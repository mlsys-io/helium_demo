from dataclasses import field
from typing import Any, Literal

from pydantic import BaseModel
from typing_extensions import TypedDict

from helium.common import Message
from helium.runtime.llm import LLMProfilingInfo


class HeliumSystemProfile(TypedDict, total=False):
    llm_benchmark: dict[str, Any]
    task_profile: dict[str, Any]
    range_profile: dict[str, Any]
    request_profile: dict[str, Any]


class HeliumQueryProfile(BaseModel):
    llm_profiling_info: dict[str, LLMProfilingInfo | None]
    """Profiling information for each LLM op in the query, if available."""


class SystemProfilingConfig(BaseModel):
    llm_service_info: dict[str, list[tuple[str | None, str | None]]] | None = None
    """Mapping from LLM service name to (API key, base URL) tuple"""


class QueryProfilingConfig(BaseModel):
    only_profile: bool = False
    """Whether to only profile the query without executing it"""
    sampling_ratio: float = 0.1
    """Sampling ratio for profiling"""
    min_sampling_size: int = 1
    """Minimum data size for profiling"""
    max_sampling_size: int = 30
    """Maximum data size for profiling"""
    query_profile_map: dict[str, HeliumQueryProfile] | None = None
    """Query profiling info. Query profiling will be skipped if this is provided."""


def _default_system_profile() -> HeliumSystemProfile:
    return {}


class HeliumRequestConfig(BaseModel):
    """
    Configuration object associated with a Helium request.
    """

    enable_cache_aware_scheduling: bool = True
    """Whether to enable cache-aware scheduling"""
    enable_runtime_adjustment: bool = True
    """Whether to enable runtime adjustment for CAS"""
    precompute_mode: Literal["none", "only", "both"] = "none"
    """Whether to only precompute the KV cache without executing the graph"""
    system_profiling_config: SystemProfilingConfig | None = None
    """Configuration for system profiling"""
    query_profiling_config: QueryProfilingConfig | None = field(
        default_factory=QueryProfilingConfig
    )
    """Configuration for query profiling"""


class HeliumRequest(BaseModel):
    """
    Request object to be processed by the Helium server.
    """

    query: dict[str, Any]
    """Query for the request that contains one or more compute graphs"""
    config: HeliumRequestConfig | None = None
    """Configuration for the request"""


PrefixKey = tuple[str, str, str, str]  # (llm_service, model, base_url, api_key)
PrefixMap = dict[PrefixKey, list[str | list[Message]]]


class HeliumResponse(BaseModel):
    """
    Response object returned by the Helium server.
    """

    outputs: dict[str, Any] = field(default_factory=dict)
    """Outputs of the request"""
    system_profile: HeliumSystemProfile = field(default_factory=_default_system_profile)
    """System profiling results of the request"""
    query_profile_map: dict[str, HeliumQueryProfile] | None = None
    """Query profiling results of the request"""
    static_prefix_map: PrefixMap | None = None
    """Mapping from prefix key to static prefixes used in the request"""
    dynamic_prefix_map: dict[int, PrefixMap] | None = None
    """Mapping from query index to prefix key to dynamic prefixes used in the request"""
    error_info: list[dict[str, Any]] | None = None
    """Information of errors encountered during processing the request"""
