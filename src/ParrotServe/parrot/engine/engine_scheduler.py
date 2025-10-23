# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import List, Dict
import time

from parrot.exceptions import parrot_assert
from parrot.utils import get_logger, time_counter_in_nanoseconds

from .primitive_job import PrimitiveJob, Fill, Generate
from .config import SchedulerConfig


logger = get_logger("Scheduler")


class EngineScheduler:
    """EngineScheduler is the scheduler for a engine.

    Different from "scheduler/dispatcher" (which is actually a cluster scheduler) in serve layer,
    the scheduler in a engine/LLM is for deciding the order of jobs/sequences to be executed in the
    next batch.
    """

    def __init__(self, config: SchedulerConfig) -> None:
        self.max_batch_size = config.max_batch_size
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_total_tokens = config.max_total_tokens

        self.waiting_jobs: List[PrimitiveJob] = []
        self.running_jobs: List[PrimitiveJob] = []

        self.policy = config.policy

        # Use context id as key. Different jobs with the same context id can't
        # present at the same time.
        self.job_arrival_time: Dict[int, float] = {}

        # task_id as key.
        self.task_arrival_time: Dict[int, float] = {}

    def add_job(self, job: PrimitiveJob) -> None:
        """Add a job to the scheduler."""

        self.waiting_jobs.append(job)
        cur_time = time_counter_in_nanoseconds()
        self.job_arrival_time[job.context_id] = cur_time
        if job.task_id not in self.task_arrival_time:
            self.task_arrival_time[job.task_id] = cur_time

    def remove_job(self, job: PrimitiveJob) -> None:
        """Remove a job from the scheduler."""

        # self.running_jobs.remove(job)
        self.job_arrival_time.pop(job.context_id)
        if job.end_flag:
            self.task_arrival_time.pop(job.task_id)

    @property
    def num_running_jobs(self) -> int:
        """Get the number of running jobs."""

        return len(self.running_jobs)

    @property
    def num_total_jobs(self) -> int:
        """Get the number of total jobs."""

        return len(self.waiting_jobs) + len(self.running_jobs)

    @property
    def is_empty(self) -> bool:
        """Whether the scheduler is empty."""

        # print(f"Waiting: {len(self.waiting_jobs)} Running: {len(self.running_jobs)}")
        # return len(self.waiting_jobs) == 0 and len(self.running_jobs) == 0
        return self.num_total_jobs == 0

    def schedule(self) -> List[PrimitiveJob]:
        """Schedule jobs."""

        # TGI-style scheduling: Fill and Gen jobs are scheduled separately.
        if self.policy == "tgi":
            cur_tokens_sum = 0
            cur_num_jobs = 0
            fill_running_jobs = []

            for job in self.waiting_jobs:
                if not isinstance(job, Fill):
                    continue

                job_num_tokens = len(job.token_ids) if job.token_ids else 0

                if cur_tokens_sum + job_num_tokens > self.max_num_batched_tokens:
                    break

                fill_running_jobs.append(job)
                if job.start_time == -1:
                    job.start_time = time_counter_in_nanoseconds()
                cur_tokens_sum += job_num_tokens
                cur_num_jobs += 1

            if len(fill_running_jobs) > 0:
                # Remove all fill_running_jobs from waiting_jobs.
                self.waiting_jobs = [
                    job for job in self.waiting_jobs if job not in fill_running_jobs
                ]

                # Preempte all running Generation jobs.
                self.waiting_jobs = self.running_jobs + self.waiting_jobs  # FIFO
                self.running_jobs = fill_running_jobs
                ret = fill_running_jobs.copy()

            assert False  # TODO: fix
        elif self.policy == "fifo_v1":
            cur_num_jobs = len(self.running_jobs)
            cur_num_batched_tokens = len(
                self.running_jobs
            )  # Note: running jobs must be all Gen jobs.
            cur_total_tokens = sum(
                [job.context.get_context_len() for job in self.running_jobs]
            )

            # print(
            #     f"Scheduling: Waiting: {len(self.waiting_jobs)} Running: {len(self.running_jobs)}"
            # )

            while self.waiting_jobs:
                job = self.waiting_jobs[0]

                job_num_tokens = (
                    1
                    if isinstance(job, Generate) or job.token_ids is None
                    else len(job.token_ids)
                )
                # NOTE(chaofan): In shared prefix mode, we should only count the prefix context once.
                job_total_tokens = job.context.get_context_len()

                # Constraints
                if cur_num_jobs + 1 > self.max_batch_size:
                    break
                if (
                    cur_num_batched_tokens + job_num_tokens
                    > self.max_num_batched_tokens
                ):
                    break
                if cur_total_tokens + job_total_tokens > self.max_total_tokens:
                    break

                self.running_jobs.append(job)
                if job.start_time == -1:
                    job.start_time = time_counter_in_nanoseconds()
                self.waiting_jobs.pop(0)

                # Update
                cur_num_jobs += 1
                cur_num_batched_tokens += job_num_tokens
                cur_total_tokens += job_total_tokens

            # NOTE(chaofan): Use copy() to avoid list modification.
            ret = self.running_jobs.copy()
        else:
            cur_num_jobs = len(self.running_jobs)
            cur_num_batched_tokens = len(
                self.running_jobs
            )  # Note: running jobs must be all Gen jobs.

            # print(
            #     f"Scheduling: Waiting: {len(self.waiting_jobs)} Running: {len(self.running_jobs)}"
            # )

            while self.waiting_jobs:
                job = self.waiting_jobs[0]

                job_num_tokens = (
                    1
                    if isinstance(job, Generate) or job.token_ids is None
                    else len(job.token_ids)
                )
                # Constraints
                if cur_num_jobs + 1 > self.max_batch_size:
                    break
                if (
                    cur_num_batched_tokens + job_num_tokens
                    > self.max_num_batched_tokens
                ):
                    break

                self.running_jobs.append(job)
                if job.start_time == -1:
                    job.start_time = time.perf_counter_ns()
                self.waiting_jobs.pop(0)

                # Update
                cur_num_jobs += 1
                cur_num_batched_tokens += job_num_tokens

            # Check total tokens constraint and do preemption

            # This is to avoid compute the same context multiple times.
            # TODO(chaofan): Only do this in shared prefix mode.
            # visited_context_ids = set()
            # if ctx.context_id not in visited_context_ids:
            #     cur_total_tokens += ctx.get_this_context_len()
            #     visited_context_ids.add(ctx.context_id)
            # parent_ctx = ctx.parent_context
            # if parent_ctx and parent_ctx.context_id not in visited_context_ids:
            #     cur_total_tokens += parent_ctx.get_this_context_len()
            #     visited_context_ids.add(parent_ctx.context_id)

            # For normal mode, we repeatly count prefix because it's repeated loaded.

            self.running_jobs.sort(
                key=lambda job: (
                    self.task_arrival_time[job.task_id],
                    self.job_arrival_time[job.context_id],
                )
            )

            # print(f"Running jobs: {self.running_jobs}")

            new_running: List[PrimitiveJob] = []
            cur_total_tokens = 0
            preempted = False
            for job in self.running_jobs:
                if preempted:
                    self._preempt(job)
                    continue

                # NOTE(chaofan): In shared prefix mode, we should only count the prefix context once.
                job_tokens = job.context.get_context_len()
                if cur_total_tokens + job_tokens > self.max_total_tokens:
                    preempted = True
                    self._preempt(job)
                    continue

                new_running.append(job)
                cur_total_tokens += job_tokens

            self.running_jobs = new_running

            # NOTE(chaofan): Use copy() to avoid list modification.
            ret = self.running_jobs.copy()

        logger.debug(
            f"Schedule {len(ret)} jobs. cur_num_jobs={cur_num_jobs}, cur_num_batched_tokens={cur_num_batched_tokens}, "
            f"cur_total_tokens={cur_total_tokens}"
        )

        return ret

    def _preempt(self, job) -> None:
        self.waiting_jobs.insert(0, job)
        # logger.debug(f"Job {job} preempted.")

    def finish(self) -> None:
        """Finish jobs."""

        new_running: List[PrimitiveJob] = []
        for job in self.running_jobs:
            if not job.finish_event.is_set():
                new_running.append(job)
            else:
                self.remove_job(job)
                job.end_time = time_counter_in_nanoseconds()
                logger.debug(
                    f"Job {job} finished. Latency: {(job.end_time - job.start_time) / 1e6} ms"
                )

        self.running_jobs = new_running
