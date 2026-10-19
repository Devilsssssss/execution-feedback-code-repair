from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from repairbench.executor import LocalSubprocessExecutor
from repairbench.feedback import FeedbackCondition
from repairbench.io import load_bugs, load_tasks, write_jsonl
from repairbench.mutations import SUPPLEMENTAL_LOGIC_MUTATOR_NAMES, generate_mutations
from repairbench.prompts import build_repair_prompt
from repairbench.schemas import BugCase, ExecutionResult, RepairTask


def _digest(code: str) -> str:
    return hashlib.sha256(code.strip().encode("utf-8")).hexdigest()


def _require_runner(result: ExecutionResult, stage: str) -> None:
    if result.status == "runner_error":
        detail = result.failures[0].message if result.failures else "unknown runner error"
        raise RuntimeError(f"executor failed during {stage}: {detail}")


def _suite_status(
    result: ExecutionResult,
    public_tests: tuple[str, ...],
    hidden_tests: tuple[str, ...],
) -> tuple[bool, bool, bool, bool]:
    public_set = set(public_tests)
    hidden_set = set(hidden_tests)
    if public_set & hidden_set:
        raise ValueError("public and hidden test suites overlap")
    public_failures = [failure for failure in result.failures if failure.test in public_set]
    hidden_failures = [failure for failure in result.failures if failure.test in hidden_set]
    if result.status != "completed":
        return False, False, False, False
    public_ok = not public_failures
    hidden_ok = not hidden_failures
    public_logic_failure = bool(public_failures) and all(
        failure.error_type == "AssertionError" for failure in public_failures
    )
    hidden_logic_failure = bool(hidden_failures) and all(
        failure.error_type == "AssertionError" for failure in hidden_failures
    )
    return public_ok, hidden_ok, public_logic_failure, hidden_logic_failure


def _suite_result(result: ExecutionResult, tests: tuple[str, ...]) -> ExecutionResult:
    test_set = set(tests)
    failures = tuple(failure for failure in result.failures if failure.test in test_set)
    return ExecutionResult(
        status=result.status,
        passed=max(0, len(tests) - len(failures)) if result.status == "completed" else 0,
        total=len(tests),
        failures=failures,
        stdout=result.stdout,
        stderr=result.stderr,
        duration_seconds=result.duration_seconds,
    )


def _candidate(
    task: RepairTask,
    mutation_name: str,
    occurrence: int,
    code: str,
    public: ExecutionResult,
    hidden: ExecutionResult,
) -> BugCase:
    bug_id = f"{task.task_id}__ast_supplement_{mutation_name}__{occurrence}"
    return BugCase(
        bug_id=bug_id,
        task_id=task.task_id,
        buggy_code=code,
        bug_type="logic_error",
        mutation=f"{mutation_name}:{occurrence}",
        metadata={
            "source": "ast_supplement_v1",
            "public_failure": public.to_dict(),
            "hidden_failure": hidden.to_dict(),
        },
    )


def collect_candidates(
    tasks: list[RepairTask],
    excluded_code_digests: set[str],
    *,
    executor: LocalSubprocessExecutor,
    reference_executor: LocalSubprocessExecutor,
) -> tuple[dict[str, list[BugCase]], dict[str, int]]:
    candidates: dict[str, list[BugCase]] = defaultdict(list)
    counters = Counter(
        tasks=len(tasks),
        reference_checks=0,
        mutation_candidates=0,
        duplicate_code=0,
        public_passed=0,
        hidden_passed=0,
        non_assertion_failure=0,
    )
    seen_code_digests = set(excluded_code_digests)
    for task in tasks:
        reference = reference_executor.execute(
            task.canonical_solution,
            (*task.public_tests, *task.hidden_tests),
            task.setup_code,
        )
        _require_runner(reference, f"{task.task_id}:reference")
        reference_public_ok, reference_hidden_ok, _, _ = _suite_status(
            reference, task.public_tests, task.hidden_tests
        )
        if not reference_public_ok or not reference_hidden_ok:
            raise ValueError(f"reference implementation failed for {task.task_id}")
        counters["reference_checks"] += 1

        for mutation in generate_mutations(
            task.canonical_solution,
            include_names=SUPPLEMENTAL_LOGIC_MUTATOR_NAMES,
        ):
            counters["mutation_candidates"] += 1
            digest = _digest(mutation.code)
            if digest in seen_code_digests:
                counters["duplicate_code"] += 1
                continue
            result = executor.execute(
                mutation.code, (*task.public_tests, *task.hidden_tests), task.setup_code
            )
            _require_runner(result, f"{task.task_id}:{mutation.mutation_id}:tests")
            public_ok, hidden_ok, public_logic, hidden_logic = _suite_status(
                result, task.public_tests, task.hidden_tests
            )
            if public_ok:
                counters["public_passed"] += 1
                continue
            if hidden_ok:
                counters["hidden_passed"] += 1
                continue
            if not public_logic or not hidden_logic:
                counters["non_assertion_failure"] += 1
                continue
            seen_code_digests.add(digest)
            public_result = _suite_result(result, task.public_tests)
            hidden_result = _suite_result(result, task.hidden_tests)
            candidates[mutation.name].append(
                _candidate(
                    task,
                    mutation.name,
                    mutation.occurrence,
                    mutation.code,
                    public_result,
                    hidden_result,
                )
            )
    return candidates, dict(counters)


def select_balanced(candidates: dict[str, list[BugCase]], count: int) -> list[BugCase]:
    if count <= 0:
        raise ValueError("target count must be positive")
    groups = {
        name: sorted(rows, key=lambda row: (row.bug_id, row.buggy_code))
        for name, rows in candidates.items()
    }
    available = sum(len(rows) for rows in groups.values())
    if available < count:
        raise ValueError(f"only {available} audited candidates available, need {count}")
    selected: list[BugCase] = []
    names = sorted(groups)
    while len(selected) < count:
        made_progress = False
        for name in names:
            if groups[name] and len(selected) < count:
                selected.append(groups[name].pop(0))
                made_progress = True
        if not made_progress:
            raise AssertionError("candidate selection stalled")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate and double-audit new AST-only logic-error mutations"
    )
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--exclude-bugs", action="append", required=True)
    parser.add_argument("--output-bugs", required=True)
    parser.add_argument("--output-sft", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--target-count", type=int, default=297)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--reference-timeout", type=float, default=15.0)
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    excluded = [bug for path in args.exclude_bugs for bug in load_bugs(path)]
    excluded_digests = {_digest(bug.buggy_code) for bug in excluded}
    executor = LocalSubprocessExecutor(timeout_seconds=args.timeout, allow_unsafe=True)
    reference_executor = LocalSubprocessExecutor(
        timeout_seconds=args.reference_timeout, allow_unsafe=True
    )
    candidates, counters = collect_candidates(
        tasks,
        excluded_digests,
        executor=executor,
        reference_executor=reference_executor,
    )
    selected = select_balanced(candidates, args.target_count)
    write_jsonl(args.output_bugs, (bug.to_dict() for bug in selected))
    task_index = {task.task_id: task for task in tasks}
    sft_rows = []
    for bug in selected:
        task = task_index[bug.task_id]
        public_failure = ExecutionResult.from_dict(bug.metadata["public_failure"])
        sft_rows.append(
            {
                "prompt": build_repair_prompt(
                    task, bug.buggy_code, public_failure, FeedbackCondition.STRUCTURED
                ),
                "completion": task.canonical_solution.strip() + "\n",
                "task_id": task.task_id,
                "bug_id": bug.bug_id,
                "bug_type": bug.bug_type,
            }
        )
    write_jsonl(args.output_sft, sft_rows)
    summary: dict[str, Any] = {
        "protocol": "ast-logic-supplement-v1",
        "audit_executor": "local_subprocess_unsafe_dev_only",
        "audit_note": (
            "Public and hidden suites were both executed locally because Docker was unavailable; "
            "rerun the frozen files with the release sandbox before remote training if policy requires."
        ),
        "tasks": len(tasks),
        "excluded_bugs": len(excluded),
        "excluded_code_digests": len(excluded_digests),
        "candidate_counters": counters,
        "candidate_counts_by_mutator": dict(sorted(
            (name, len(rows)) for name, rows in candidates.items()
        )),
        "selected_count": len(selected),
        "selected_counts_by_mutator": dict(Counter(bug.mutation.split(":", 1)[0] for bug in selected)),
        "selected_bug_ids": [bug.bug_id for bug in selected],
        "output_bugs": str(args.output_bugs),
        "output_sft": str(args.output_sft),
    }
    output_summary = Path(args.output_summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
