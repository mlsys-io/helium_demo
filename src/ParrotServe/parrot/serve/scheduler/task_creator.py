# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Dict

from parrot.exceptions import parrot_assert
from parrot.utils import get_logger, RecyclePool

from parrot.serve.graph import CompletionChain, PerformanceCriteria

from .completion_task import CompletionTask
from .schedule_annotation import ScheduleAnnotation


logger = get_logger("TaskCreator")


class TaskCreator:
    """TaskCreator creates a CompletionTask object for the CompletionChain."""

    def __init__(self) -> None:
        self._task_id_pool = RecyclePool("TaskIDPool", debug_mode=True)

    def _lower_criteria(self, criteria: PerformanceCriteria) -> ScheduleAnnotation:
        if criteria == PerformanceCriteria.LATENCY:
            return ScheduleAnnotation(
                tasks_num_upperbound=4,
                tokens_num_upperbound=4096,
            )
        elif criteria == PerformanceCriteria.THROUGHPUT:
            return ScheduleAnnotation(
                tasks_num_upperbound=99999,
                tokens_num_upperbound=9999999999999,
            )
        else:
            raise NotImplementedError(
                f"PerformanceCriteria {criteria} is not supported."
            )

    def create_task(self, completion_chain: CompletionChain) -> CompletionTask:
        """Create a Task object for the CompletionChain.

        Args:
            completion_chain: CompletionChain.

        Returns:
            CompletionTask. The Task object created for the CompletionChain.
        """

        parrot_assert(
            completion_chain.gen_node is not None,
            "The CompletionChain does not have a GenNode.",
        )

        var = completion_chain.gen_node.sv
        parrot_assert(var.is_activated, "The chain is not activated.")

        # Create a new Task
        task_id = self._task_id_pool.allocate()
        schedule_annotation = self._lower_criteria(var.criteria)

        logger.debug(
            f"Create Task(task_id={task_id}) for CompletionChain(request_id={completion_chain.request_id},"
            f" session_id={completion_chain.session_id})."
        )

        return CompletionTask(
            task_id=task_id,
            chain=completion_chain,
            schedule_annotation=schedule_annotation,
        )

    def free_task(self, task: CompletionTask) -> None:
        """Free the CompletionTask.

        Args:
            task: CompletionTask. The task to be freed.
        """

        self._task_id_pool.free(task.task_id)

        # Remove from the engine
        task.leave_scheduled()
        return
