import asyncio
import sys
from argparse import ArgumentParser, Namespace
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


async def main(
    num_trials: int,
    max_workload_scale: int,
    cuda_device_name: str | None = None,
    server_log_file: str | None = None,
    gpu_util_log_dir: Path | None = None,
    result_dir: Path | None = None,
    llm_server_host: str | None = None,
    llm_server_port: str | None = None,
    verbose: bool = False,
    test: bool = False,
) -> None:
    from bench import generate_tasks

    if result_dir is not None:
        result_dir.mkdir(exist_ok=True)

    bench_results: dict[str, dict[str, pd.DataFrame]] = {}

    # Parse LLM server hosts
    llm_server_hosts: str | list[str] | None
    if llm_server_host is None or "," not in llm_server_host:
        llm_server_hosts = llm_server_host
    else:
        llm_server_hosts = llm_server_host.split(",")

    # Parse LLM server ports
    llm_server_ports: int | list[int] | None
    if llm_server_port is None:
        llm_server_ports = None
    elif "," in llm_server_port:
        llm_server_ports = [int(port) for port in llm_server_port.split(",")]
    else:
        llm_server_ports = int(llm_server_port)

    # Parse server log files
    server_log_files: Path | list[Path | None] | None
    if server_log_file is None:
        server_log_files = None
    elif "," in server_log_file:
        server_log_files = [Path(file) for file in server_log_file.split(",")]
    else:
        server_log_files = Path(server_log_file)

    result_suffix = "_test.csv" if test else ".csv"
    async for task in generate_tasks(
        num_trials,
        max_workload_scale,
        cuda_device_name,
        llm_server_hosts,
        llm_server_ports,
        server_log_files,
        gpu_util_log_dir,
        verbose,
        helium_profiling=False,
    ):
        summary_df_dict = await task.run()
        if result_dir is not None:
            workload_name = task.workload
            if workload_name in bench_results:
                task_results = bench_results[workload_name]
            else:
                task_results = {}
                bench_results[workload_name] = task_results
            for name, summary_df in summary_df_dict.items():
                if name in task_results:
                    task_results[name] = pd.concat([task_results[name], summary_df])
                else:
                    task_results[name] = summary_df

            # Checkpoint results after each task
            for name, summary_df in task_results.items():
                result_path = result_dir / (f"{workload_name}_{name}" + result_suffix)
                summary_df.to_csv(result_path, index=False)
                if verbose:
                    print(f"Results saved to '{result_path}'")


def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    parser.add_argument("--silent", "-s", action="store_false", dest="verbose")
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-workload-scale", type=int, default=4)
    parser.add_argument("--cuda-device-name", type=str, default=None)
    parser.add_argument("--benchmark-log-file", type=Path, default=None)
    parser.add_argument("--server-log-file", type=str, default=None)
    parser.add_argument("--result-dir", type=Path, default=Path("results"))
    parser.add_argument("--gpu-util-log-dir", type=Path, default=None)
    parser.add_argument("--no-result", action="store_true", default=False)
    parser.add_argument("--llm-server-host", type=str, required=False, default=None)
    parser.add_argument("--llm-server-port", type=str, required=False, default=None)
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run small benchmarks for development testing",
    )

    return parser.parse_args()


@contextmanager
def configure_logging(log_file: Path | None) -> Iterator[None]:
    if log_file is None:
        yield
        return

    # Get available log file name
    available_log_file = log_file
    counter = 0
    while available_log_file.exists():
        counter += 1
        available_log_file = available_log_file.with_stem(f"{log_file.stem}_{counter}")

    with available_log_file.open("w") as f:
        stdout = sys.stdout
        stderr = sys.stderr
        sys.stdout = f
        sys.stderr = f
        yield
    sys.stdout = stdout
    sys.stderr = stderr


if __name__ == "__main__":
    args = parse_args()
    with configure_logging(args.benchmark_log_file):
        asyncio.run(
            main(
                num_trials=args.num_trials,
                max_workload_scale=args.max_workload_scale,
                cuda_device_name=args.cuda_device_name,
                server_log_file=args.server_log_file,
                gpu_util_log_dir=args.gpu_util_log_dir,
                result_dir=None if args.no_result else args.result_dir,
                llm_server_host=args.llm_server_host,
                llm_server_port=args.llm_server_port,
                verbose=args.verbose,
                test=args.test,
            )
        )
