# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from enum import Enum
from typing import List, Dict, Optional
from asyncio import Event

from parrot.exceptions import parrot_assert

from parrot.utils import get_logger

from parrot.serve.backend_repr import ExecutionEngine, Context
from parrot.serve.graph import CompletionChain

from .schedule_annotation import ScheduleAnnotation


class TaskStatus(Enum):
    CREATED = 0
    INQUEUE = 1
    EXECUTING = 2
    FINISHED = 3
    ERROR = 4


class CompletionTask:
    """ScheduleUnit wraps CompletionChain."""

    def __init__(
        self,
        task_id: int,
        chain: CompletionChain,
        schedule_annotation: ScheduleAnnotation = ScheduleAnnotation(),
    ):
        self.task_id = task_id
        self.chain = chain
        self.status = TaskStatus.CREATED

        # Tokenized result
        # Map from tokenizer name to tokenized result
        # A tokenized result is a List of token ids, i.e. List[List[int]]
        self.tokenized_result: Optional[Dict[str, List[List[int]]]] = None

        # Context bound to the task
        # A list of contexts that are bound to the task
        self.contexts: List[Context] = []

        # Scheduling
        self._scheduled_event: Event = Event()
        self.schedule_annotation = schedule_annotation
        self.engine: Optional[ExecutionEngine] = None

    @property
    def is_tokenized(self) -> bool:
        return self.tokenized_result is not None

    @property
    def context_bound(self) -> bool:
        return len(self.contexts) > 0

    @property
    def is_scheduled(self) -> bool:
        return self._scheduled_event.is_set()

    def schedule_to(
        self, engine: ExecutionEngine, update_engine_info: bool = True
    ) -> None:
        """Schedule the task to the engine."""

        self.engine = engine
        self._scheduled_event.set()

        if update_engine_info:
            self.engine.update_servelayer_runtime_info_add_task(self)

    async def wait_scheduled(self) -> None:
        """Wait until the task is scheduled."""

        await self._scheduled_event.wait()

    def leave_scheduled(self) -> None:
        """Leave the scheduled status."""

        self.engine.update_servelayer_runtime_info_remove_task(self)

    def tokenize_chain(self, tokenizers_wrapper: "TokenizersWrapper") -> None:
        """Tokenize the chain using the tokenizers in the wrapper."""

        parrot_assert(not self.is_tokenized, "Tokenized result is already available.")
        parrot_assert(self.chain.sv_created, "SVs are not created yet.")

        self.tokenized_result = {}
        for fill_node in self.chain.iter_fill():
            tokenized_result: Dict = tokenizers_wrapper.tokenize_all(fill_node.get())
            for key, value in tokenized_result.items():
                if key not in self.tokenized_result:
                    self.tokenized_result[key] = []
                self.tokenized_result[key].append(value)

    def get_token_nums(self, tokenizer_name: str) -> int:
        """Get the number of tokens in the tokenized result."""

        parrot_assert(self.is_tokenized, "Tokenized result is not available.")
        tokens_num = 0
        # Add the number of tokens in Fill part.
        for token_ids in self.tokenized_result[tokenizer_name]:
            tokens_num += len(token_ids)
        # Add the number of tokens in Gen part.
        tokens_num += self.chain.gen_node.sampling_config.max_gen_length
        return tokens_num

    def __str__(self):
        return f"CompletionTask(chain={self.chain})"
