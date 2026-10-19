from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from repairbench.io import load_bugs, load_tasks
from repairbench.schemas import BugCase, ExecutionResult, RepairTask


def _short(value: Any, limit: int = 72) -> str:
    text = "" if value is None else str(value).replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _feedback_contract(feedback: ExecutionResult) -> dict[str, Any]:
    first = feedback.failures[0] if feedback.failures else None
    return {
        "passed": feedback.passed,
        "total": feedback.total,
        "error_type": None
        if first is None
        else _short(first.error_type, 48),
    }


def _contract_line(contract: dict[str, Any]) -> str:
    payload = json.dumps(contract, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return f"# FEEDBACK_CONTRACT {payload}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def build_rows(
    sft_rows: list[dict[str, Any]],
    bugs: list[BugCase],
    tasks: list[RepairTask],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bug_index = {bug.bug_id: bug for bug in bugs}
    task_index = {task.task_id: task for task in tasks}
    if len(bug_index) != len(bugs):
        raise ValueError("input bugs contain duplicate bug_id values")
    output = []
    for index, row in enumerate(sft_rows):
        bug_id = str(row.get("bug_id", ""))
        bug = bug_index.get(bug_id)
        task = task_index.get(str(row.get("task_id", "")))
        if bug is None or task is None:
            raise ValueError(f"row {index} references unknown bug/task: {bug_id}")
        if bug.task_id != task.task_id or str(row.get("bug_type")) != bug.bug_type:
            raise ValueError(f"row {index} has inconsistent task or bug type: {bug_id}")
        if str(row.get("completion", "")).strip() != task.canonical_solution.strip():
            raise ValueError(f"reference completion mismatch: {bug_id}")
        feedback = ExecutionResult.from_dict(bug.metadata["public_failure"])
        if feedback.all_passed or not feedback.failures:
            raise ValueError(f"bug has no public failure for contract target: {bug_id}")
        contract = _feedback_contract(feedback)
        completion = _contract_line(contract) + "\n" + str(row["completion"]).lstrip()
        output.append(
            {
                "prompt": str(row["prompt"]),
                "completion": completion,
                "task_id": task.task_id,
                "bug_id": bug_id,
                "bug_type": bug.bug_type,
                "feedback_contract": contract,
            }
        )
    if {str(row["bug_id"]) for row in output} != set(bug_index):
        raise ValueError("input SFT rows and bug records are not one-to-one")
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output)
    summary = {
        "protocol": "feedback-contract-sft-v3",
        "rows": len(output),
        "tasks": len({str(row["task_id"]) for row in output}),
        "bug_types": {},
        "completion_contract_prefix": "# FEEDBACK_CONTRACT",
        "reference_code_sha256": hashlib.sha256(
            "".join(str(row["completion"]).split("\n", 1)[1] for row in output).encode("utf-8")
        ).hexdigest(),
        "output_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "target_change": (
            "Only a deterministic feedback-derived comment is prepended to each reference code; "
            "prompts, bug IDs, task split, and reference implementation are unchanged."
        ),
    }
    counts: dict[str, int] = {}
    for row in output:
        counts[row["bug_type"]] = counts.get(row["bug_type"], 0) + 1
    summary["bug_types"] = dict(sorted(counts.items()))
    return output, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build feedback-contract SFT targets")
    parser.add_argument("--input-sft", type=Path, required=True)
    parser.add_argument("--bugs", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    rows, summary = build_rows(
        load_jsonl(args.input_sft), load_bugs(args.bugs), load_tasks(args.tasks)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary["input_sft_sha256"] = hashlib.sha256(args.input_sft.read_bytes()).hexdigest()
    summary["input_bugs_sha256"] = hashlib.sha256(args.bugs.read_bytes()).hexdigest()
    summary["input_tasks_sha256"] = hashlib.sha256(args.tasks.read_bytes()).hexdigest()
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
