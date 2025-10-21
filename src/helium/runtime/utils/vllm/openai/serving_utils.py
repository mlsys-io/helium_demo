import time
from dataclasses import asdict, dataclass, field
from typing import Any, cast

import numpy as np

from vllm.vllm.outputs import CompletionOutput, RequestOutput
from vllm.vllm.sequence import RequestMetrics


def add_request_output_delta(
    prev_res: RequestOutput, res: RequestOutput
) -> RequestOutput:
    if prev_res.request_id != res.request_id:
        raise ValueError(
            f"Request ID mismatch: {prev_res.request_id} != {res.request_id}"
        )

    new_out_tmp: list[CompletionOutput | None] = [None] * len(prev_res.outputs)
    for i, (prev_outputs, outputs) in enumerate(zip(prev_res.outputs, res.outputs)):
        new_outputs = CompletionOutput(
            index=prev_outputs.index,
            text=prev_outputs.text + outputs.text,
            token_ids=[*prev_outputs.token_ids, *outputs.token_ids],
            cumulative_logprob=None,
            logprobs=(
                None
                if prev_outputs.logprobs is None or outputs.logprobs is None
                else prev_outputs.logprobs + outputs.logprobs
            ),
            finish_reason=outputs.finish_reason,
            stop_reason=outputs.stop_reason,
            lora_request=outputs.lora_request,
        )
        new_out_tmp[i] = new_outputs
    new_out = cast(list[CompletionOutput], new_out_tmp)

    prev_metrics = prev_res.metrics
    metrics = res.metrics
    if not (prev_metrics is None or metrics is None):
        if metrics.time_in_queue is None:
            new_time_in_queue = prev_metrics.time_in_queue
        else:
            new_time_in_queue = (
                prev_metrics.time_in_queue or 0
            ) + metrics.time_in_queue
        if metrics.scheduler_time is None:
            new_scheduler_time = prev_metrics.scheduler_time
        else:
            new_scheduler_time = (
                prev_metrics.scheduler_time or 0
            ) + metrics.scheduler_time
        if metrics.model_forward_time is None:
            new_model_forward_time = prev_metrics.model_forward_time
        else:
            new_model_forward_time = (
                prev_metrics.model_forward_time or 0
            ) + metrics.model_forward_time
        if metrics.model_execute_time is None:
            new_model_execute_time = prev_metrics.model_execute_time
        else:
            new_model_execute_time = (
                prev_metrics.model_execute_time or 0
            ) + metrics.model_execute_time

        new_metrics = RequestMetrics(
            arrival_time=prev_metrics.arrival_time,
            last_token_time=metrics.last_token_time,
            first_scheduled_time=prev_metrics.first_scheduled_time,
            first_token_time=prev_metrics.first_token_time,
            time_in_queue=new_time_in_queue,
            finished_time=metrics.finished_time,
            scheduler_time=new_scheduler_time,
            model_forward_time=new_model_forward_time,
            model_execute_time=new_model_execute_time,
        )
    else:
        new_metrics = None

    prompt = res.prompt if prev_res.prompt is None else prev_res.prompt
    prompt_token_ids = (
        res.prompt_token_ids
        if prev_res.prompt_token_ids is None
        else prev_res.prompt_token_ids
    )
    prompt_logprobs = (
        res.prompt_logprobs
        if prev_res.prompt_logprobs is None
        else prev_res.prompt_logprobs
    )

    new_output = RequestOutput(
        request_id=prev_res.request_id,
        prompt=prompt,
        prompt_token_ids=prompt_token_ids,
        prompt_logprobs=prompt_logprobs,
        outputs=new_out,
        finished=res.finished,
        metrics=new_metrics,
        lora_request=prev_res.lora_request,
        encoder_prompt=prev_res.encoder_prompt,
        encoder_prompt_token_ids=prev_res.encoder_prompt_token_ids,
        num_cached_tokens=res.num_cached_tokens,
    )
    return new_output


@dataclass
class BenchmarkMetrics:
    latency: float = -1
    ttft: float = 0.0  # Time to first token
    itl: list[float] = field(default_factory=list)  # list of inter-token latencies
    tpot: float | None = None  # avg next-token latencies
    prompt_len: int = -1
    output_len: int = 0


class BenchmarkRequestTracker:
    def __init__(self) -> None:
        self.start_time = time.perf_counter()
        self.recent_time = self.start_time
        self._metrics = BenchmarkMetrics()
        self.not_received = True

    def update(self, request_output: RequestOutput) -> None:
        metrics = self._metrics
        now = time.perf_counter()
        # Prompt length
        if metrics.prompt_len < 0 and request_output.prompt_token_ids is not None:
            metrics.prompt_len = len(request_output.prompt_token_ids)
        # TTFT
        if self.not_received:
            metrics.ttft = now - self.start_time
            self.not_received = False
        # ITL
        metrics.itl.append(now - self.recent_time)
        # Output length
        metrics.output_len += len(request_output.outputs[0].token_ids)
        # Recent time
        self.recent_time = now

    def finish(self) -> BenchmarkMetrics:
        if self._metrics.prompt_len < 0:
            raise ValueError("Prompt length not set")
        now = time.perf_counter()
        metrics = self._metrics
        metrics.latency = now - self.start_time
        if metrics.output_len > 1:
            metrics.tpot = (metrics.latency - metrics.ttft) / (metrics.output_len - 1)
        return metrics


@dataclass
class BenchmarkResults:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: list[tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: list[tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: list[tuple[float, float]]
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: list[tuple[float, float]]


class VLLMBenchmarker:
    def __init__(self, selected_percentiles: list[float] | None = None) -> None:
        self.metrics: list[BenchmarkMetrics] = []
        self.start_time: float | None = None
        self.selected_percentiles = (
            [99.0] if selected_percentiles is None else selected_percentiles
        )

    def start_benchmark(self) -> None:
        self.start_time = time.perf_counter()

    def add_metrics(self, metrics: BenchmarkMetrics | None) -> None:
        if self.start_time is None or metrics is None:
            return
        self.metrics.append(metrics)

    def stop_benchmark(self) -> dict[str, Any]:
        if self.start_time is None:
            raise ValueError("Benchmark not started")
        dur_s = time.perf_counter() - self.start_time
        self.start_time = None

        actual_output_lens: list[int] = []
        total_input = 0
        good_completed = 0
        itls: list[float] = []
        tpots: list[float] = []
        ttfts: list[float] = []
        e2els: list[float] = []
        outputs = self.metrics
        completed = len(outputs)
        for i in range(completed):
            metrics = outputs[i]
            actual_output_lens.append(metrics.output_len)
            total_input += metrics.prompt_len
            if metrics.tpot is not None:
                tpots.append(metrics.tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            completed += 1

        results = BenchmarkResults(
            completed=completed,
            total_input=total_input,
            total_output=sum(actual_output_lens),
            request_throughput=completed / dur_s,
            request_goodput=good_completed / dur_s,
            output_throughput=sum(actual_output_lens) / dur_s,
            total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
            mean_ttft_ms=float(
                np.mean(ttfts or 0) * 1000
            ),  # ttfts is empty if streaming is not supported by backend
            std_ttft_ms=float(np.std(ttfts or 0) * 1000),
            median_ttft_ms=float(np.median(ttfts or 0) * 1000),
            percentiles_ttft_ms=[
                (p, float(np.percentile(ttfts or 0, p) * 1000))
                for p in self.selected_percentiles
            ],
            mean_tpot_ms=float(np.mean(tpots or 0) * 1000),
            std_tpot_ms=float(np.std(tpots or 0) * 1000),
            median_tpot_ms=float(np.median(tpots or 0) * 1000),
            percentiles_tpot_ms=[
                (p, float(np.percentile(tpots or 0, p) * 1000))
                for p in self.selected_percentiles
            ],
            mean_itl_ms=float(np.mean(itls or 0) * 1000),
            std_itl_ms=float(np.std(itls or 0) * 1000),
            median_itl_ms=float(np.median(itls or 0) * 1000),
            percentiles_itl_ms=[
                (p, float(np.percentile(itls or 0, p) * 1000))
                for p in self.selected_percentiles
            ],
            mean_e2el_ms=float(np.mean(e2els or 0) * 1000),
            std_e2el_ms=float(np.std(e2els or 0) * 1000),
            median_e2el_ms=float(np.median(e2els or 0) * 1000),
            percentiles_e2el_ms=[
                (p, float(np.percentile(e2els or 0, p) * 1000))
                for p in self.selected_percentiles
            ],
        )

        self.metrics.clear()

        return asdict(results)


benchmarker = VLLMBenchmarker()
