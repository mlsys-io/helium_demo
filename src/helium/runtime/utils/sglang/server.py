import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

from helium.runtime.utils.sglang.config import SGLangRouterConfig, SGLangServerConfig

# NOTE: Set this manually to the Python interpreter that has `sglang` + `sglang-router`
# installed. This should be an absolute path or a repo-relative path.
SGLANG_PYTHON = Path("src/sglang/.venv/bin/python")


def resolve_sglang_python() -> str:
    return str(SGLANG_PYTHON.absolute())


def resolve_project_python() -> str:
    return str(Path(sys.executable).absolute())


def _popen(
    cmd: list[str],
    env: dict[str, str] | None = None,
    log_file: Path | None = None,
) -> subprocess.Popen[bytes]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    stdout = None
    stderr = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        f = open(log_file, "wb", buffering=0)
        stdout = f
        stderr = f

    # Start a new process group so we can terminate the whole tree.
    return subprocess.Popen(
        cmd,
        env=merged_env,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )


def terminate_process_group(
    proc: subprocess.Popen[bytes], timeout_s: float = 60
) -> None:
    if proc.poll() is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.1)

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait(timeout=5)


def wait_for_health(host: str, port: int, timeout_s: float = 600) -> None:
    url = f"http://{host}:{port}/health"
    start = time.perf_counter()
    with requests.Session() as session:
        while time.perf_counter() - start < timeout_s:
            try:
                res = session.get(url, timeout=5)
                if res.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.5)
    raise TimeoutError(f"SGLang server failed to become healthy: {url}")


def popen_sglang_server(
    config: SGLangServerConfig,
    env: dict[str, str] | None = None,
    log_file: Path | None = None,
) -> subprocess.Popen[bytes]:
    python = resolve_sglang_python()
    cmd = [python, "-m", "sglang.launch_server", *config.to_cli_args()]
    return _popen(cmd, env=env, log_file=log_file)


def popen_sglang_router(
    config: SGLangRouterConfig,
    env: dict[str, str] | None = None,
    log_file: Path | None = None,
) -> subprocess.Popen[bytes]:
    python = resolve_sglang_python()
    cmd = [python, "-m", "sglang_router.launch_router", *config.to_cli_args()]
    return _popen(cmd, env=env, log_file=log_file)


def popen_py_router(
    config: SGLangRouterConfig,
    env: dict[str, str] | None = None,
    log_file: Path | None = None,
) -> subprocess.Popen[bytes]:
    python = resolve_project_python()
    cmd = [python, "-m", "helium.runtime.utils.sglang.router", *config.to_cli_args()]
    return _popen(cmd, env=env, log_file=log_file)
