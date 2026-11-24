import json
from argparse import Namespace
from collections.abc import Iterable
from pathlib import Path

import pytest

from repairbench import cli
from repairbench.evaluation import EvaluationRunner
from repairbench.feedback import FeedbackCondition
from repairbench.io import write_jsonl
from repairbench.mutations import Mutation
from repairbench.schemas import BugCase, ExecutionResult, RepairTask
from repairbench.schemas import TestFailure as FailureRecord
from repairbench.validation import execute_checked, require_candidate_fails


def _task() -> RepairTask:
    return RepairTask(
        task_id="t1",
        prompt="Return x + 1.",
        entry_point="add_one",
        canonical_solution="def add_one(x):\n    return x + 1\n",
        public_tests=("assert add_one(1) == 2",),
        hidden_tests=("assert add_one(2) == 3",),
    )


def _bug() -> BugCase:
    return BugCase(
        bug_id="b1",
        task_id="t1",
        buggy_code="def add_one(x):\n    return x\n",
        bug_type="logic_error",
        mutation="manual",
    )


def _result(*, passed: int, total: int = 1, status: str = "completed") -> ExecutionResult:
    failures = () if passed == total else (FailureRecord("test", "AssertionError", "failed"),)
    return ExecutionResult(status=status, passed=passed, total=total, failures=failures)


class ResultExecutor:
    def __init__(self, results: Iterable[ExecutionResult]) -> None:
        self._results = iter(results)

    def execute(self, code: str, tests, setup_code: str = "") -> ExecutionResult:
        return next(self._results)


def _cli_args(tmp_path: Path, *, tasks: Path, bugs: Path | None = None) -> Namespace:
    return Namespace(
        tasks=str(tasks),
        bugs=str(bugs) if bugs else None,
        output=str(tmp_path / "output.jsonl"),
        per_task=1,
        executor="local",
        timeout=1,
        allow_unsafe_local=True,
    )


def test_generate_bugs_keeps_only_mutations_failing_public_and_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_path = tmp_path / "tasks.jsonl"
    write_jsonl(task_path, [_task().to_dict()])
    executor = ResultExecutor(
        [
            _result(passed=1),  # reference public
            _result(passed=1),  # reference hidden
            _result(passed=0),  # mutation public
            _result(passed=1),  # mutation hidden: mutation is not benchmark-valid
        ]
    )
    monkeypatch.setattr(cli, "_executor", lambda args: executor)
    monkeypatch.setattr(
        cli,
        "generate_mutations",
        lambda code: [Mutation("manual", "logic_error", 0, _bug().buggy_code)],
    )

    assert cli.command_generate_bugs(_cli_args(tmp_path, tasks=task_path)) == 0
    assert (tmp_path / "output.jsonl").read_text(encoding="utf-8") == ""


def test_generate_bugs_does_not_export_hidden_failure_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_path = tmp_path / "tasks.jsonl"
    write_jsonl(task_path, [_task().to_dict()])
    executor = ResultExecutor(
        [
            _result(passed=1),  # reference public
            _result(passed=1),  # reference hidden
            _result(passed=0),  # mutation public
            _result(passed=0),  # mutation hidden: valid failing sample
        ]
    )
    monkeypatch.setattr(cli, "_executor", lambda args: executor)
    monkeypatch.setattr(
        cli,
        "generate_mutations",
        lambda code: [Mutation("manual", "logic_error", 0, _bug().buggy_code)],
    )

    assert cli.command_generate_bugs(_cli_args(tmp_path, tasks=task_path)) == 0
    [record] = [
        json.loads(line)
        for line in (tmp_path / "output.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert set(record["metadata"]) == {"public_failure"}


def test_generate_bugs_aborts_on_executor_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task_path = tmp_path / "tasks.jsonl"
    write_jsonl(task_path, [_task().to_dict()])
    executor = ResultExecutor(
        [
            _result(passed=1),
            _result(passed=1),
            _result(passed=0, status="runner_error"),
        ]
    )
    monkeypatch.setattr(cli, "_executor", lambda args: executor)
    monkeypatch.setattr(
        cli,
        "generate_mutations",
        lambda code: [Mutation("manual", "logic_error", 0, _bug().buggy_code)],
    )

    with pytest.raises(RuntimeError, match="infrastructure failed"):
        cli.command_generate_bugs(_cli_args(tmp_path, tasks=task_path))


def test_validate_rejects_bug_that_passes_hidden_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    task_path = tmp_path / "tasks.jsonl"
    bug_path = tmp_path / "bugs.jsonl"
    write_jsonl(task_path, [_task().to_dict()])
    write_jsonl(bug_path, [_bug().to_dict()])
    executor = ResultExecutor(
        [
            _result(passed=1),  # reference public
            _result(passed=1),  # reference hidden
            _result(passed=0),  # bug public
            _result(passed=1),  # bug hidden
        ]
    )
    monkeypatch.setattr(cli, "_executor", lambda args: executor)

    args = _cli_args(tmp_path, tasks=task_path, bugs=bug_path)
    assert cli.command_validate(args) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] is False
    assert any("does not fail" in failure and "hidden" in failure for failure in report["failures"])


def test_validate_reports_runner_error_instead_of_accepting_it_as_bug(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    task_path = tmp_path / "tasks.jsonl"
    bug_path = tmp_path / "bugs.jsonl"
    write_jsonl(task_path, [_task().to_dict()])
    write_jsonl(bug_path, [_bug().to_dict()])
    executor = ResultExecutor(
        [
            _result(passed=1),
            _result(passed=1),
            _result(passed=0, status="runner_error"),
            _result(passed=0),
        ]
    )
    monkeypatch.setattr(cli, "_executor", lambda args: executor)

    args = _cli_args(tmp_path, tasks=task_path, bugs=bug_path)
    assert cli.command_validate(args) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] is False
    assert any("infrastructure failed" in failure for failure in report["failures"])


def test_build_sft_rejects_runner_error_on_hidden_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_path = tmp_path / "tasks.jsonl"
    bug_path = tmp_path / "bugs.jsonl"
    write_jsonl(task_path, [_task().to_dict()])
    write_jsonl(bug_path, [_bug().to_dict()])
    executor = ResultExecutor(
        [
            _result(passed=1),  # reference public
            _result(passed=1),  # reference hidden
            _result(passed=0),  # bug public
            _result(passed=0, status="runner_error"),  # bug hidden
        ]
    )
    monkeypatch.setattr(cli, "_executor", lambda args: executor)

    with pytest.raises(RuntimeError, match="infrastructure failed"):
        cli.command_build_sft(_cli_args(tmp_path, tasks=task_path, bugs=bug_path))
    assert not (tmp_path / "output.jsonl").exists()


def test_candidate_timeout_is_a_valid_failure_but_runner_error_is_not() -> None:
    timed_out = execute_checked(
        ResultExecutor([_result(passed=0, status="timeout")]),
        "while True: pass",
        ["assert True"],
        stage="candidate",
    )
    require_candidate_fails(timed_out, "candidate public tests")

    with pytest.raises(RuntimeError, match="infrastructure failed"):
        execute_checked(
            ResultExecutor([_result(passed=0, status="runner_error")]),
            "def f(): pass",
            ["assert True"],
            stage="candidate",
        )


def test_evaluation_rejects_bug_that_already_passes_public_tests() -> None:
    task = _task()
    runner = EvaluationRunner(
        model=type("Model", (), {"repair": lambda self, prompt: task.canonical_solution})(),
        executor=ResultExecutor([_result(passed=1)]),
        condition=FeedbackCondition.NONE,
        max_repairs=1,
    )

    with pytest.raises(ValueError, match="bug b1 public tests"):
        runner.evaluate_case(task, _bug())
