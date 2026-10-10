from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .executor import CodeExecutor
from .feedback import FeedbackCondition
from .models import RepairModel
from .prompts import build_repair_prompt
from .schemas import BugCase, RepairTask


@dataclass
class EvaluationRunner:
    model: RepairModel
    executor: CodeExecutor
    condition: FeedbackCondition
    max_repairs: int = 3

    def _execute(self, code: str, tests: tuple[str, ...], setup_code: str, stage: str):
        result = self.executor.execute(code, tests, setup_code=setup_code)
        if result.status == "runner_error":
            detail = result.failures[0].message if result.failures else "unknown runner error"
            if result.stderr.strip():
                detail = f"{detail}; stderr={result.stderr.strip()[-500:]}"
            raise RuntimeError(f"evaluation infrastructure failed during {stage}: {detail}")
        return result

    def evaluate_case(self, task: RepairTask, bug: BugCase) -> dict[str, Any]:
        if task.task_id != bug.task_id:
            raise ValueError("task and bug IDs do not match")

        current_code = bug.buggy_code
        trajectory: list[dict[str, Any]] = []
        generated_characters = 0
        first_repair_success = False

        for repair_index in range(1, self.max_repairs + 1):
            public_before = self._execute(
                current_code, task.public_tests, task.setup_code, "public_before"
            )
            prompt = build_repair_prompt(task, current_code, public_before, self.condition)
            repaired_code = self.model.repair(prompt)
            generated_characters += len(repaired_code)
            public_after = self._execute(
                repaired_code, task.public_tests, task.setup_code, "public_after"
            )
            trajectory.append(
                {
                    "repair_index": repair_index,
                    "prompt": prompt,
                    "candidate_code": repaired_code,
                    "public_before": public_before.to_dict(),
                    "public_after": public_after.to_dict(),
                }
            )
            current_code = repaired_code
            if public_after.all_passed:
                first_repair_success = repair_index == 1
                break

        final_public = self._execute(
            current_code, task.public_tests, task.setup_code, "final_public"
        )
        # Hidden tests are evaluated exactly once and are never included in any prompt.
        final_hidden = self._execute(
            current_code, task.hidden_tests, task.setup_code, "final_hidden"
        )
        repair_success = final_public.all_passed and final_hidden.all_passed
        return {
            "task_id": task.task_id,
            "bug_id": bug.bug_id,
            "bug_type": bug.bug_type,
            "condition": self.condition.value,
            "repair_at_1": first_repair_success and final_hidden.all_passed,
            "repair_at_k": repair_success,
            "repairs_used": len(trajectory),
            "generated_characters": generated_characters,
            "final_public": final_public.to_dict(),
            "final_hidden": final_hidden.to_dict(),
            "trajectory": trajectory,
        }


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        return {"cases": 0, "repair_at_1": 0.0, "repair_at_k": 0.0, "by_bug_type": {}}

    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_type.setdefault(str(row["bug_type"]), []).append(row)

    def rate(items: list[dict[str, Any]], key: str) -> float:
        return sum(bool(item[key]) for item in items) / len(items)

    return {
        "cases": len(rows),
        "repair_at_1": rate(rows, "repair_at_1"),
        "repair_at_k": rate(rows, "repair_at_k"),
        "mean_repairs_used": sum(int(row["repairs_used"]) for row in rows) / len(rows),
        "mean_generated_characters": sum(int(row["generated_characters"]) for row in rows)
        / len(rows),
        "by_bug_type": {
            bug_type: {
                "cases": len(items),
                "repair_at_1": rate(items, "repair_at_1"),
                "repair_at_k": rate(items, "repair_at_k"),
            }
            for bug_type, items in sorted(by_type.items())
        },
    }
