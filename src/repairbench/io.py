from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, TypeVar

from .schemas import BugCase, RepairTask

T = TypeVar("T")


def read_jsonl(path: str | Path, factory: Callable[[dict[str, Any]], T]) -> list[T]:
    source = Path(path)
    records: list[T] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(factory(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid record at {source}:{line_number}: {exc}") from exc
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_tasks(path: str | Path) -> list[RepairTask]:
    tasks = read_jsonl(path, RepairTask.from_dict)
    _ensure_unique((task.task_id for task in tasks), "task_id")
    return tasks


def load_bugs(path: str | Path) -> list[BugCase]:
    bugs = read_jsonl(path, BugCase.from_dict)
    _ensure_unique((bug.bug_id for bug in bugs), "bug_id")
    return bugs


def _ensure_unique(values: Iterable[str], label: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        joined = ", ".join(sorted(duplicates))
        raise ValueError(f"duplicate {label}: {joined}")


def check_task_disjointness(*task_groups: Iterable[RepairTask]) -> None:
    seen: dict[str, int] = {}
    overlaps: set[str] = set()
    for group_index, group in enumerate(task_groups):
        for task in group:
            previous = seen.get(task.task_id)
            if previous is not None and previous != group_index:
                overlaps.add(task.task_id)
            seen[task.task_id] = group_index
    if overlaps:
        joined = ", ".join(sorted(overlaps))
        raise ValueError(f"task-level data leakage detected: {joined}")


def index_tasks(tasks: Iterable[RepairTask]) -> dict[str, RepairTask]:
    result: dict[str, RepairTask] = {}
    for task in tasks:
        if task.task_id in result:
            raise ValueError(f"duplicate task_id: {task.task_id}")
        result[task.task_id] = task
    return result
