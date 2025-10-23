# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Optional
from asyncio import Event

from parrot.constants import NONE_CONTEXT_ID

from .engine import ExecutionEngine


class Context:
    """Context represents the KV cache of SemanticVariable in one single engine.

    Note context is highly related to:
    - The engine that it is associated with.
    - The parent context. (Due to the property of the KV cache, a context can not be isolated from its
      parent context.)

    If B wants to continue generating based on A's context, the lifecycle is:
    - B forks a context based on A's context.
    - B generates tokens in this context.
    - When B's job finish, we free the memory taken by B's context. This will not
      affect A's context.

    Contexts are naturally organized in a tree structure by the forking relationship.
    """

    def __init__(
        self,
        context_id: int,
        engine: ExecutionEngine,
        parent_context: Optional["Context"] = None,
    ):
        self.context_id = context_id
        self.engine = engine
        self.parent_context = parent_context

        # Events.
        self.start_event = (
            Event()
        )  # Start event indicates a request has been sent to fill this context.
        self.ready_event = (
            Event()
        )  # Ready event indicates the context has been filled in the backend.

        # The number of tokens this context (don't include its parent) holds.
        self.tokens_num = 0

    @property
    def has_parent_context(self) -> bool:
        return self.parent_context is not None

    @property
    def parent_context_id(self) -> int:
        return (
            self.parent_context.context_id
            if self.has_parent_context
            else NONE_CONTEXT_ID
        )

    @property
    def memory_usage(self) -> float:
        num_cached_tokens = self.engine.get_num_cached_tokens()
        memory_per_token = (
            self.engine.get_cache_mem() / num_cached_tokens
            if num_cached_tokens > 0
            else 0
        )

        return memory_per_token * self.tokens_num

    @property
    def engine_url(self) -> str:
        return self.engine.http_address
