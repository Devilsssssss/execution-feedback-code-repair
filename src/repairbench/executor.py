from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .schemas import ExecutionResult, TestFailure


class CodeExecutor(Protocol):
    def execute(self, code: str, tests: Sequence[str], setup_code: str = "") -> ExecutionResult:
        ...


def _runner_path() -> Path:
    return Path(__file__).with_name("executor_runner.py").resolve()


def _decode_result(
    process: subprocess.CompletedProcess[str], duration: float, test_count: int
) -> ExecutionResult:
    if process.returncode != 0:
        return ExecutionResult(
            status="runner_error",
            passed=0,
            total=test_count,
            failures=(
                TestFailure(
                    test="",
                    error_type="RunnerProcessError",
                    message=f"runner exited with code {process.returncode}",
                ),
            ),
            stdout=process.stdout[-2000:],
            stderr=process.stderr[-2000:],
            duration_seconds=duration,
        )
    try:
        value = json.loads(process.stdout)
        value["duration_seconds"] = duration
        return ExecutionResult.from_dict(value)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return ExecutionResult(
            status="runner_error",
            passed=0,
            total=test_count,
            failures=(
                TestFailure(test="", error_type="InvalidRunnerOutput", message=str(exc)),
            ),
            stdout=process.stdout[-2000:],
            stderr=process.stderr[-2000:],
            duration_seconds=duration,
        )


@dataclass
class LocalSubprocessExecutor:
    """Development-only executor. This is not a security sandbox."""

    timeout_seconds: float = 3.0
    allow_unsafe: bool = False

    def execute(self, code: str, tests: Sequence[str], setup_code: str = "") -> ExecutionResult:
        if not self.allow_unsafe:
            raise RuntimeError(
                "local execution is unsafe; pass allow_unsafe=True only for reviewed code"
            )
        payload = json.dumps({"code": code, "tests": list(tests), "setup_code": setup_code})
        started = time.perf_counter()
        try:
            process = subprocess.run(
                [sys.executable, "-I", str(_runner_path())],
                input=payload,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                cwd=str(_runner_path().parent),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                status="timeout",
                passed=0,
                total=len(tests),
                failures=(
                    TestFailure(
                        test="",
                        error_type="TimeoutError",
                        message=f"execution exceeded {self.timeout_seconds:.1f}s",
                    ),
                ),
                stdout=(exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
                stderr=(exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
                duration_seconds=time.perf_counter() - started,
            )
        return _decode_result(process, time.perf_counter() - started, len(tests))


@dataclass
class DockerExecutor:
    timeout_seconds: float = 5.0
    image: str = "python:3.11-slim"
    memory: str = "256m"
    cpus: str = "1.0"
    # RLIMIT_NPROC counts every thread owned by the Unix user, including the
    # model inference process outside the sandbox.  Keep this above typical
    # BLAS/CUDA thread counts while still preventing unbounded forking.
    pids_limit: int = 4096

    def execute(self, code: str, tests: Sequence[str], setup_code: str = "") -> ExecutionResult:
        project_root = _runner_path().parents[2]
        runner_in_container = "/app/src/repairbench/executor_runner.py"
        payload = json.dumps({"code": code, "tests": list(tests), "setup_code": setup_code})
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--memory",
            self.memory,
            "--cpus",
            self.cpus,
            "--pids-limit",
            str(self.pids_limit),
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "-i",
            "-v",
            f"{project_root}:/app:ro",
            "-w",
            "/app",
            self.image,
            "python",
            "-I",
            runner_in_container,
        ]
        started = time.perf_counter()
        try:
            process = subprocess.run(
                command,
                input=payload,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Docker executable was not found") from exc
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                status="timeout",
                passed=0,
                total=len(tests),
                failures=(
                    TestFailure(
                        test="",
                        error_type="TimeoutError",
                        message=f"execution exceeded {self.timeout_seconds:.1f}s",
                    ),
                ),
                stdout=(exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
                stderr=(exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
                duration_seconds=time.perf_counter() - started,
            )
        return _decode_result(process, time.perf_counter() - started, len(tests))


@dataclass
class BubblewrapExecutor:
    """Linux sandbox using unprivileged namespaces and process resource limits."""

    timeout_seconds: float = 5.0
    memory_bytes: int = 512 * 1024 * 1024
    # RLIMIT_NPROC is charged to the host user, not just the sandbox. Model
    # runtimes may already own many worker threads before bwrap starts.
    pids_limit: int = 4096
    python_executable: str = "/usr/bin/python3"
    startup_retries: int = 3
    retry_delay_seconds: float = 0.2

    def execute(self, code: str, tests: Sequence[str], setup_code: str = "") -> ExecutionResult:
        payload = json.dumps({"code": code, "tests": list(tests), "setup_code": setup_code})
        command = [
            "prlimit",
            f"--as={self.memory_bytes}",
            f"--cpu={max(1, math.ceil(self.timeout_seconds))}",
            f"--nproc={self.pids_limit}",
            "--nofile=64",
            "--",
            "bwrap",
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--clearenv",
            "--setenv",
            "PATH",
            "/usr/bin",
            "--ro-bind",
            "/usr",
            "/usr",
        ]
        for system_path in ("/lib", "/lib64"):
            if Path(system_path).exists():
                command.extend(("--ro-bind", system_path, system_path))
        command.extend(
            (
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
                "--dir",
                "/app",
                "--ro-bind",
                str(_runner_path()),
                "/app/executor_runner.py",
                "--chdir",
                "/tmp",
                self.python_executable,
                "-I",
                "/app/executor_runner.py",
            )
        )
        started = time.perf_counter()
        for attempt in range(self.startup_retries + 1):
            try:
                process = subprocess.run(
                    command,
                    input=payload,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise RuntimeError("Bubblewrap or prlimit executable was not found") from exc
            except subprocess.TimeoutExpired as exc:
                return ExecutionResult(
                    status="timeout",
                    passed=0,
                    total=len(tests),
                    failures=(
                        TestFailure(
                            test="",
                            error_type="TimeoutError",
                            message=f"execution exceeded {self.timeout_seconds:.1f}s",
                        ),
                    ),
                    stdout=(exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
                    stderr=(exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
                    duration_seconds=time.perf_counter() - started,
                )
            transient_namespace_error = (
                process.returncode != 0
                and "Creating new namespace failed: Resource temporarily unavailable"
                in process.stderr
            )
            if not transient_namespace_error or attempt == self.startup_retries:
                break
            time.sleep(self.retry_delay_seconds * (attempt + 1))
        return _decode_result(process, time.perf_counter() - started, len(tests))
