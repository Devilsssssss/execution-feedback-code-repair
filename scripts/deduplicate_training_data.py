from __future__ import annotations

import argparse
import ast
import difflib
import json
import re
from pathlib import Path
from typing import Any

from repairbench.io import load_bugs, load_tasks, write_jsonl


def _code_signature(code: str) -> str:
    tree = ast.parse(code)
    return ast.dump(tree, annotate_fields=False, include_attributes=False)


def _prompt_tokens(prompt: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", prompt.lower()))


def _similar(a: str, b: str) -> bool:
    left = _prompt_tokens(a)
    right = _prompt_tokens(b)
    if not left or not right:
        return False
    jaccard = len(left & right) / len(left | right)
    ratio = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
    return jaccard >= 0.75 or ratio >= 0.86


def find_duplicate_task_groups(tasks: list[Any]) -> list[list[str]]:
    by_code: dict[str, list[Any]] = {}
    for task in tasks:
        by_code.setdefault(_code_signature(task.canonical_solution), []).append(task)
    groups: list[list[str]] = []
    for candidates in by_code.values():
        pending = list(candidates)
        while pending:
            first = pending.pop(0)
            group = [first]
            matching = [task for task in pending if _similar(first.prompt, task.prompt)]
            for task in matching:
                pending.remove(task)
                group.append(task)
            if len(group) > 1:
                groups.append([task.task_id for task in group])
    return groups


def _load_sft(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="remove semantic duplicate tasks from SFT data")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--bugs", required=True)
    parser.add_argument("--sft", required=True)
    parser.add_argument("--tasks-output", required=True)
    parser.add_argument("--bugs-output", required=True)
    parser.add_argument("--sft-output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    bugs = load_bugs(args.bugs)
    sft = _load_sft(args.sft)
    duplicate_groups = find_duplicate_task_groups(tasks)
    removed_task_ids = {task_id for group in duplicate_groups for task_id in group[1:]}
    removed_bug_ids = {bug.bug_id for bug in bugs if bug.task_id in removed_task_ids}
    kept_sft = [row for row in sft if str(row.get("bug_id")) not in removed_bug_ids]

    write_jsonl(args.tasks_output, (task.to_dict() for task in tasks if task.task_id not in removed_task_ids))
    write_jsonl(args.bugs_output, (bug.to_dict() for bug in bugs if bug.bug_id not in removed_bug_ids))
    write_jsonl(args.sft_output, kept_sft)
    report = {
        "input": {"tasks": len(tasks), "bugs": len(bugs), "sft": len(sft)},
        "output": {
            "tasks": len(tasks) - len(removed_task_ids),
            "bugs": len(bugs) - len(removed_bug_ids),
            "sft": len(kept_sft),
        },
        "duplicate_task_groups": duplicate_groups,
        "removed_task_ids": sorted(removed_task_ids),
        "removed_bug_ids": sorted(removed_bug_ids),
    }
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
