from dataclasses import dataclass

import pytest

from repairbench.evaluation import EvaluationRunner, summarize
from repairbench.executor import LocalSubprocessExecutor
from repairbench.feedback import FeedbackCondition
from repairbench.schemas import BugCase, RepairTask


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
