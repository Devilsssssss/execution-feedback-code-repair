from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from repairbench.executor import DockerExecutor
from repairbench.io import load_bugs, load_tasks
from repairbench.schemas import BugCase, ExecutionResult, RepairTask


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _suite_failures(
    result: ExecutionResult, public_tests: tuple[str, ...], hidden_tests: tuple[str, ...]
) -> tuple[list, list]:
    public = set(public_tests)
    hidden = set(hidden_tests)
    return (
        [failure for failure in result.failures if failure.test in public],
        [failure for failure in result.failures if failure.test in hidden],
    )


def _failure_summary(failure) -> dict[str, object]:
    return {
        "test": failure.test,
        "error_type": failure.error_type,
        "message": failure.message,
    }


def _run_reference(task: RepairTask, executor: DockerExecutor) -> dict[str, object]:
    tests = (*task.public_tests, *task.hidden_tests)
    result = executor.execute(task.canonical_solution, tests, task.setup_code)
    public_failures, hidden_failures = _suite_failures(
        result, task.public_tests, task.hidden_tests
    )
    return {
        "task_id": task.task_id,
        "status": result.status,
        "public_passed": not public_failures and result.status == "completed",
        "hidden_passed": not hidden_failures and result.status == "completed",
        "failures": [_failure_summary(item) for item in result.failures],
    }


def _run_bug(
    bug: BugCase, task: RepairTask, executor: DockerExecutor
) -> dict[str, object]:
    tests = (*task.public_tests, *task.hidden_tests)
    result = executor.execute(bug.buggy_code, tests, task.setup_code)
    public_failures, hidden_failures = _suite_failures(
        result, task.public_tests, task.hidden_tests
    )
    public_logic = bool(public_failures) and all(
        item.error_type == "AssertionError" for item in public_failures
    )
    hidden_logic = bool(hidden_failures) and all(
        item.error_type == "AssertionError" for item in hidden_failures
    )
    return {
        "bug_id": bug.bug_id,
        "task_id": bug.task_id,
        "status": result.status,
        "public_failed": bool(public_failures),
        "hidden_failed": bool(hidden_failures),
        "public_assertion_only": public_logic,
        "hidden_assertion_only": hidden_logic,
        "valid": (
            result.status == "completed"
            and bool(public_failures)
            and bool(hidden_failures)
            and public_logic
            and hidden_logic
        ),
        "public_failures": [_failure_summary(item) for item in public_failures],
        "hidden_failures": [_failure_summary(item) for item in hidden_failures],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-audit AST supplement references and bugs in a Docker sandbox"
    )
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--bugs", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--image", default="repairbench-audit:python3.13")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")

    tasks = load_tasks(args.tasks)
    bugs = load_bugs(args.bugs)
    task_index = {task.task_id: task for task in tasks}
    if len(task_index) != len(tasks):
        raise ValueError("duplicate task id")
    missing = sorted({bug.task_id for bug in bugs} - set(task_index))
    if missing:
        raise ValueError(f"bugs reference missing tasks: {missing[:3]}")

    def executor() -> DockerExecutor:
        return DockerExecutor(timeout_seconds=args.timeout, image=args.image)

    reference_results: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_reference, task, executor()): task.task_id for task in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            reference_results[str(result["task_id"])] = result

    bug_results: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_bug, bug, task_index[bug.task_id], executor()): bug.bug_id
            for bug in bugs
        }
        for future in as_completed(futures):
            result = future.result()
            bug_results[str(result["bug_id"])] = result

    references_ok = all(
        item["status"] == "completed"
        and item["public_passed"]
        and item["hidden_passed"]
        for item in reference_results.values()
    )
    bugs_ok = all(item["valid"] for item in bug_results.values())
    invalid_references = [
        item for item in reference_results.values() if not (
            item["status"] == "completed"
            and item["public_passed"]
            and item["hidden_passed"]
        )
    ]
    invalid_bugs = [item for item in bug_results.values() if not item["valid"]]
    summary = {
        "protocol": "ast-logic-supplement-v1-docker-audit",
        "status": "pass" if references_ok and bugs_ok else "fail",
        "executor": "docker",
        "image": args.image,
        "workers": args.workers,
        "timeout_seconds": args.timeout,
        "inputs": {
            "tasks": {"path": str(args.tasks), "rows": len(tasks), "sha256": _sha256(args.tasks)},
            "bugs": {"path": str(args.bugs), "rows": len(bugs), "sha256": _sha256(args.bugs)},
        },
        "reference_checks": len(reference_results),
        "reference_passes": sum(
            int(item["status"] == "completed" and item["public_passed"] and item["hidden_passed"])
            for item in reference_results.values()
        ),
        "bug_checks": len(bug_results),
        "bug_passes": sum(int(item["valid"]) for item in bug_results.values()),
        "invalid_references": sorted(invalid_references, key=lambda item: str(item["task_id"])),
        "invalid_bugs": sorted(invalid_bugs, key=lambda item: str(item["bug_id"])),
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
