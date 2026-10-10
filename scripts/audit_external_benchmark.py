from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from repairbench.executor import BubblewrapExecutor, DockerExecutor, LocalSubprocessExecutor
from repairbench.io import index_tasks, load_bugs, load_tasks, write_jsonl
from repairbench.schemas import ExecutionResult


def build_executor(args: argparse.Namespace):
    if args.executor == "local":
        return LocalSubprocessExecutor(
            timeout_seconds=args.timeout,
            allow_unsafe=args.allow_unsafe_local,
        )
    if args.executor == "bwrap":
        return BubblewrapExecutor(timeout_seconds=args.timeout)
    return DockerExecutor(timeout_seconds=args.timeout)


def _require_runner(result: ExecutionResult, stage: str) -> None:
    if result.status != "runner_error":
        return
    detail = result.failures[0].message if result.failures else "unknown runner error"
    raise RuntimeError(f"benchmark infrastructure failed during {stage}: {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Keep external benchmark cases with valid references and detectable bugs."
    )
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--bugs", required=True)
    parser.add_argument("--tasks-output", required=True)
    parser.add_argument("--bugs-output", required=True)
    parser.add_argument("--rejected-output", required=True)
    parser.add_argument("--executor", choices=("docker", "bwrap", "local"), default="docker")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--allow-unsafe-local", action="store_true")
    args = parser.parse_args()

    tasks = index_tasks(load_tasks(args.tasks))
    bugs = load_bugs(args.bugs)
    executor = build_executor(args)
    reference_results: dict[str, tuple[ExecutionResult, ExecutionResult]] = {}
    accepted_bugs: list[dict[str, Any]] = []
    accepted_task_ids: set[str] = set()
    rejected: list[dict[str, Any]] = []
    rejection_reasons: Counter[str] = Counter()

    for bug in bugs:
        task = tasks.get(bug.task_id)
        if task is None:
            reason = "missing_task"
            rejected.append({"bug": bug.to_dict(), "reason": reason})
            rejection_reasons[reason] += 1
            continue

        if task.task_id not in reference_results:
            reference_public = executor.execute(
                task.canonical_solution, task.public_tests, task.setup_code
            )
            reference_hidden = executor.execute(
                task.canonical_solution, task.hidden_tests, task.setup_code
            )
            _require_runner(reference_public, f"{task.task_id}:reference_public")
            _require_runner(reference_hidden, f"{task.task_id}:reference_hidden")
            reference_results[task.task_id] = (reference_public, reference_hidden)
        reference_public, reference_hidden = reference_results[task.task_id]

        bug_public = executor.execute(bug.buggy_code, task.public_tests, task.setup_code)
        bug_hidden = executor.execute(bug.buggy_code, task.hidden_tests, task.setup_code)
        _require_runner(bug_public, f"{bug.bug_id}:bug_public")
        _require_runner(bug_hidden, f"{bug.bug_id}:bug_hidden")

        reasons = []
        if not reference_public.all_passed:
            reasons.append("reference_public_failed")
        if not reference_hidden.all_passed:
            reasons.append("reference_hidden_failed")
        if bug_public.all_passed:
            reasons.append("bug_not_detected_by_public_tests")
        if bug_hidden.all_passed:
            reasons.append("bug_not_detected_by_hidden_tests")
        if reasons:
            rejection_reasons.update(reasons)
            rejected.append(
                {
                    "bug": bug.to_dict(),
                    "task": task.to_dict(),
                    "reasons": reasons,
                    "reference_public": reference_public.to_dict(),
                    "reference_hidden": reference_hidden.to_dict(),
                    "bug_public": bug_public.to_dict(),
                    "bug_hidden": bug_hidden.to_dict(),
                }
            )
            continue

        accepted_task_ids.add(task.task_id)
        accepted_bugs.append(bug.to_dict())

    accepted_tasks = [
        task.to_dict() for task_id, task in tasks.items() if task_id in accepted_task_ids
    ]
    write_jsonl(args.tasks_output, accepted_tasks)
    write_jsonl(args.bugs_output, accepted_bugs)
    write_jsonl(args.rejected_output, rejected)
    report = {
        "input_tasks": len(tasks),
        "input_bugs": len(bugs),
        "accepted_tasks": len(accepted_tasks),
        "accepted_bugs": len(accepted_bugs),
        "rejected_bugs": len(rejected),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "tasks_output": args.tasks_output,
        "bugs_output": args.bugs_output,
        "rejected_output": args.rejected_output,
    }
    report_path = Path(args.rejected_output).with_suffix(".summary.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if accepted_bugs else 1


if __name__ == "__main__":
    raise SystemExit(main())
