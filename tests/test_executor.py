import shutil
import subprocess
import sys

import pytest

from repairbench.executor import BubblewrapExecutor, LocalSubprocessExecutor


def executor() -> LocalSubprocessExecutor:
    return LocalSubprocessExecutor(timeout_seconds=1.0, allow_unsafe=True)


def test_executor_extracts_expected_and_actual() -> None:
    result = executor().execute("def add_one(x):\n    return x\n", ["assert add_one(2) == 3"])

    assert not result.all_passed
    assert result.failures[0].actual == "2"
    assert result.failures[0].expected == "3"


def test_executor_extracts_values_with_test_preamble() -> None:
    result = executor().execute(
        "def add_one(value):\n    return value - 1\n",
        ["import math\nassert add_one(1) == 2"],
    )

    assert not result.all_passed
    assert result.failures[0].expected == "2"
    assert result.failures[0].actual == "0"


def test_executor_reports_runtime_error() -> None:
    result = executor().execute("def broken(x):\n    return missing + x\n", ["assert broken(1) == 1"])

    assert result.failures[0].error_type == "NameError"


def test_executor_enforces_timeout() -> None:
    result = LocalSubprocessExecutor(timeout_seconds=0.2, allow_unsafe=True).execute(
        "while True:\n    pass\n", ["assert True"]
    )

    assert result.status == "timeout"


def test_bubblewrap_retries_transient_namespace_failure(monkeypatch) -> None:
    attempts = 0

    def fake_run(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return subprocess.CompletedProcess(
                args=args[0],
                returncode=1,
                stdout="",
                stderr="bwrap: Creating new namespace failed: Resource temporarily unavailable",
            )
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout='{"status":"completed","passed":1,"total":1,"failures":[]}',
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = BubblewrapExecutor(retry_delay_seconds=0).execute("", ["assert True"])

    assert attempts == 2
    assert result.all_passed


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("bwrap") is None or shutil.which("prlimit") is None,
    reason="Bubblewrap executor requires Linux, bwrap, and prlimit",
)
def test_bubblewrap_executor_runs_in_linux_sandbox() -> None:
    result = BubblewrapExecutor(timeout_seconds=2.0).execute(
        "def add_one(x):\n    return x + 1\n", ["assert add_one(2) == 3"]
    )

    assert result.all_passed
