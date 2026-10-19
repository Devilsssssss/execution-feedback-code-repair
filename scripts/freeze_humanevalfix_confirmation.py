from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(f"expected an object at {path}:{line_number}")
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze a deterministic HumanEvalFix confirmation subset"
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=ROOT / "data/processed/humanevalfix_atomized/audited_tasks.jsonl",
    )
    parser.add_argument(
        "--bugs",
        type=Path,
        default=ROOT / "data/processed/humanevalfix_atomized/audited_bugs.jsonl",
    )
    parser.add_argument("--count", type=int, default=36)
    parser.add_argument("--seed", default="humanevalfix-confirmation-v1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/processed/humanevalfix_atomized/confirmation_v1",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "artifacts/humanevalfix_confirmation_v1.summary.json",
    )
    args = parser.parse_args()
    if args.count < 30 or args.count > 40:
        raise ValueError("confirmation count must be between 30 and 40")

    tasks = read_jsonl(args.tasks)
    bugs = read_jsonl(args.bugs)
    task_by_id = {str(row["task_id"]): row for row in tasks}
    bug_by_task = {str(row["task_id"]): row for row in bugs}
    if len(task_by_id) != len(tasks) or len(bug_by_task) != len(bugs):
        raise ValueError("source task and bug ids must be unique")
    if set(task_by_id) != set(bug_by_task):
        raise ValueError("task and bug sources must cover the same task ids")
    if args.count > len(tasks):
        raise ValueError("confirmation count exceeds source task count")

    ranked = sorted(
        task_by_id,
        key=lambda task_id: hashlib.sha256(f"{args.seed}:{task_id}".encode()).hexdigest(),
    )
    selected_ids = ranked[: args.count]
    selected_tasks = [row for row in tasks if row["task_id"] in selected_ids]
    selected_bugs = [row for row in bugs if row["task_id"] in selected_ids]
    selected_tasks_path = args.output_dir / "tasks.jsonl"
    selected_bugs_path = args.output_dir / "bugs.jsonl"
    write_jsonl(selected_tasks_path, selected_tasks)
    write_jsonl(selected_bugs_path, selected_bugs)

    selected_digest = hashlib.sha256(
        "".join(f"{task_id}\n" for task_id in sorted(selected_ids)).encode()
    ).hexdigest()
    summary = {
        "status": "frozen-for-future-confirmation",
        "seed": args.seed,
        "count": args.count,
        "source_tasks": str(args.tasks),
        "source_bugs": str(args.bugs),
        "source_task_count": len(tasks),
        "selected_task_ids_sha256": selected_digest,
        "selected_task_ids": selected_ids,
        "source_sha256": {"tasks": sha256(args.tasks), "bugs": sha256(args.bugs)},
        "output_sha256": {
            "tasks": sha256(selected_tasks_path),
            "bugs": sha256(selected_bugs_path),
        },
        "source_used_in_prior_analysis": True,
        "blind_status": (
            "not_blind_retrospective_freeze: the 143-task source was already used in "
            "development analyses; this subset is blind only from this freeze onward"
        ),
        "protocol": (
            "Do not use this subset for mutation design, training-data selection, "
            "checkpoint selection, or prompt tuning. Evaluate it once after the AST "
            "control and mixed multi-seed runs are complete."
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
