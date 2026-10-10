from dataclasses import dataclass

import pytest

from repairbench.evaluation import (
    EvaluationRunner,
    FeedbackOverride,
    build_feedback_derangement,
    summarize,
)
from repairbench.executor import LocalSubprocessExecutor
from repairbench.feedback import FeedbackCondition
from repairbench.schemas import BugCase, ExecutionResult, RepairTask


@dataclass
class FixedModel:
    code: str

    def repair(self, prompt: str) -> str:
        assert "hidden" not in prompt.lower()
        return self.code


def test_evaluation_uses_hidden_test_only_for_final_score() -> None:
    task = RepairTask(
        task_id="t1",
        prompt="Return x + 1.",
        entry_point="add_one",
        canonical_solution="def add_one(x):\n    return x + 1\n",
        public_tests=("assert add_one(1) == 2",),
        hidden_tests=("assert add_one(-1) == 0",),
    )
    bug = BugCase(
        bug_id="b1",
        task_id="t1",
        buggy_code="def add_one(x):\n    return x\n",
        bug_type="logic_error",
        mutation="manual",
    )
    runner = EvaluationRunner(
        model=FixedModel(task.canonical_solution),
        executor=LocalSubprocessExecutor(allow_unsafe=True),
        condition=FeedbackCondition.STRUCTURED,
    )

    record = runner.evaluate_case(task, bug)
    summary = summarize([record])

    assert record["repair_at_1"] is True
    assert record["repair_at_k"] is True
    assert summary["repair_at_k"] == 1.0


def test_evaluation_does_not_count_runner_failure_as_model_failure() -> None:
    task = RepairTask(
        task_id="t1",
        prompt="Return one.",
        entry_point="one",
        canonical_solution="def one():\n    return 1\n",
        public_tests=("assert one() == 1",),
        hidden_tests=("assert one() == 1",),
    )
    bug = BugCase(
        bug_id="b1",
        task_id="t1",
        buggy_code="def one():\n    return 0\n",
        bug_type="logic_error",
        mutation="manual",
    )
    runner = EvaluationRunner(
        model=FixedModel("import os\nos._exit(1)\n"),
        executor=LocalSubprocessExecutor(allow_unsafe=True),
        condition=FeedbackCondition.STRUCTURED,
        max_repairs=1,
    )

    with pytest.raises(RuntimeError, match="infrastructure failed"):
        runner.evaluate_case(task, bug)


def test_feedback_derangement_is_reproducible_and_has_no_fixed_points() -> None:
    executions = {
        f"b{index}": ExecutionResult(status="completed", passed=0, total=index + 1)
        for index in range(4)
    }

    first = build_feedback_derangement(executions, seed=42)
    second = build_feedback_derangement(executions, seed=42)

    assert first == second
    assert all(target != override.source_bug_id for target, override in first.items())
    assert {override.source_bug_id for override in first.values()} == set(executions)


def test_evaluation_can_use_feedback_from_another_case() -> None:
    task = RepairTask(
        task_id="t1",
        prompt="Return x + 1.",
        entry_point="add_one",
        canonical_solution="def add_one(x):\n    return x + 1\n",
        public_tests=("assert add_one(1) == 2",),
        hidden_tests=("assert add_one(-1) == 0",),
    )
    bug = BugCase(
        bug_id="b1",
        task_id="t1",
        buggy_code="def add_one(x):\n    return x\n",
        bug_type="logic_error",
        mutation="manual",
    )
    foreign_feedback = ExecutionResult(
        status="completed",
        passed=0,
        total=1,
        stderr="foreign feedback marker",
    )
    runner = EvaluationRunner(
        model=FixedModel(task.canonical_solution),
        executor=LocalSubprocessExecutor(allow_unsafe=True),
        condition=FeedbackCondition.RAW,
        max_repairs=1,
        feedback_overrides={"b1": FeedbackOverride("b2", foreign_feedback)},
    )

    record = runner.evaluate_case(task, bug)

    assert record["feedback_permuted"] is True
    assert record["trajectory"][0]["feedback_source_bug_id"] == "b2"
    assert "foreign feedback marker" in record["trajectory"][0]["prompt"]
