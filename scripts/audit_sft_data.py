from __future__ import annotations

import argparse
import ast
import hashlib
import json
import statistics
import warnings
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from repairbench.feedback import FeedbackCondition
from repairbench.io import load_bugs, load_tasks
from repairbench.prompts import SYSTEM_PROMPT, build_repair_prompt
from repairbench.schemas import BugCase, ExecutionResult, RepairTask


def _digest(value: str) -> str:
    return hashlib.sha256(value.strip().encode("utf-8")).hexdigest()


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    index = min(len(values) - 1, round((len(values) - 1) * percentile))
    return values[index]


def _length_stats(values: Iterable[int]) -> dict[str, int | float]:
    values = list(values)
    if not values:
        return {"count": 0, "min": 0, "mean": 0.0, "p50": 0, "p95": 0, "p99": 0, "max": 0}
    return {
        "count": len(values),
        "min": min(values),
        "mean": round(statistics.fmean(values), 2),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def _parseable(code: str) -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ast.parse(code)
        except SyntaxError:
            return False
        return True


def _parse_warnings(code: str) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            ast.parse(code)
        except SyntaxError:
            return []
    return [str(item.message) for item in caught]


def _entry_point_exists(code: str, entry_point: str) -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point
        for node in tree.body
    )


def _load_sft(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _tokenizer_lengths(rows: list[dict[str, Any]], tokenizer_id: str) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("--tokenizer requires the model optional dependencies") from exc
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    prompt_lengths = []
    completion_lengths = []
    combined_lengths = []
    chat_lengths = []
    for row in rows:
        prompt = str(row.get("prompt", ""))
        completion = str(row.get("completion", ""))
        prompt_lengths.append(len(tokenizer(prompt, add_special_tokens=True)["input_ids"]))
        completion_lengths.append(
            len(tokenizer(completion, add_special_tokens=False)["input_ids"])
        )
        combined_lengths.append(
            len(tokenizer(prompt + "\n" + completion, add_special_tokens=True)["input_ids"])
        )
        chat_encoding = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            tokenize=True,
            add_generation_prompt=False,
        )
        if hasattr(chat_encoding, "input_ids"):
            chat_lengths.append(len(chat_encoding.input_ids))
        else:
            chat_lengths.append(len(chat_encoding))
    return {
        "prompt_tokens": _length_stats(prompt_lengths),
        "completion_tokens": _length_stats(completion_lengths),
        "combined_tokens": _length_stats(combined_lengths),
        "training_chat_tokens": _length_stats(chat_lengths),
        "tokenizer": tokenizer_id,
    }


def _check_task_group(tasks: list[RepairTask]) -> dict[str, Any]:
    duplicate_tests = []
    public_hidden_overlap = []
    unparseable = []
    missing_entry_point = []
    parse_warning_tasks = []
    for task in tasks:
        public = list(task.public_tests)
        hidden = list(task.hidden_tests)
        if len(set(public)) != len(public) or len(set(hidden)) != len(hidden):
            duplicate_tests.append(task.task_id)
        if set(public) & set(hidden):
            public_hidden_overlap.append(task.task_id)
        if not _parseable(task.canonical_solution):
            unparseable.append(task.task_id)
        if _parse_warnings(task.canonical_solution):
            parse_warning_tasks.append(task.task_id)
        if not _entry_point_exists(task.canonical_solution, task.entry_point):
            missing_entry_point.append(task.task_id)
    return {
        "tasks": len(tasks),
        "duplicate_test_tasks": duplicate_tests,
        "public_hidden_overlap_tasks": public_hidden_overlap,
        "unparseable_reference_tasks": unparseable,
        "reference_parse_warning_tasks": parse_warning_tasks,
        "missing_entry_point_tasks": missing_entry_point,
        "public_tests": _length_stats(len(task.public_tests) for task in tasks),
        "hidden_tests": _length_stats(len(task.hidden_tests) for task in tasks),
        "reference_chars": _length_stats(len(task.canonical_solution) for task in tasks),
    }


def _check_bugs(
    tasks: dict[str, RepairTask], bugs: list[BugCase]
) -> tuple[dict[str, Any], dict[str, BugCase]]:
    missing_tasks = []
    unparseable = []
    missing_entry_point = []
    unchanged = []
    invalid_feedback = []
    duplicate_code: dict[str, list[str]] = {}
    task_bug_counts = Counter()
    type_counts = Counter()
    bug_index: dict[str, BugCase] = {}
    for bug in bugs:
        bug_index[bug.bug_id] = bug
        task = tasks.get(bug.task_id)
        task_bug_counts[bug.task_id] += 1
        type_counts[bug.bug_type] += 1
        duplicate_code.setdefault(_digest(bug.buggy_code), []).append(bug.bug_id)
        if task is None:
            missing_tasks.append(bug.bug_id)
            continue
        if not _parseable(bug.buggy_code):
            unparseable.append(bug.bug_id)
        if not _entry_point_exists(bug.buggy_code, task.entry_point):
            missing_entry_point.append(bug.bug_id)
        if bug.buggy_code.strip() == task.canonical_solution.strip():
            unchanged.append(bug.bug_id)
        raw_feedback = bug.metadata.get("public_failure")
        try:
            feedback = ExecutionResult.from_dict(raw_feedback)
        except (AttributeError, TypeError, ValueError):
            feedback = None
        if feedback is None or feedback.all_passed or feedback.total != len(task.public_tests):
            invalid_feedback.append(bug.bug_id)
    repeated_codes = {
        digest: bug_ids for digest, bug_ids in duplicate_code.items() if len(bug_ids) > 1
    }
    return (
        {
            "bugs": len(bugs),
            "bug_types": dict(sorted(type_counts.items())),
            "tasks_with_bugs": len(task_bug_counts),
            "bugs_per_task": _length_stats(task_bug_counts.values()),
            "missing_task_bugs": missing_tasks,
            "unparseable_bug_bugs": unparseable,
            "missing_entry_point_bugs": missing_entry_point,
            "unchanged_bugs": unchanged,
            "invalid_public_feedback_bugs": invalid_feedback,
            "duplicate_buggy_code_groups": list(repeated_codes.values()),
        },
        bug_index,
    )


def _check_sft(
    tasks: dict[str, RepairTask], bugs: dict[str, BugCase], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    missing_fields = []
    unknown_bugs = []
    completion_mismatch = []
    prompt_mismatch = []
    duplicate_rows = []
    seen_bug_ids: set[str] = set()
    prompt_lengths: list[int] = []
    completion_lengths: list[int] = []
    for index, row in enumerate(rows):
        bug_id = str(row.get("bug_id", ""))
        required = ("prompt", "completion", "task_id", "bug_id", "bug_type")
        if any(not str(row.get(field, "")).strip() for field in required):
            missing_fields.append(index)
        if bug_id in seen_bug_ids:
            duplicate_rows.append(bug_id)
        seen_bug_ids.add(bug_id)
        bug = bugs.get(bug_id)
        task = tasks.get(str(row.get("task_id", "")))
        if bug is None or task is None:
            unknown_bugs.append(bug_id or f"row_{index}")
            continue
        if str(row.get("bug_type")) != bug.bug_type:
            missing_fields.append(index)
        if str(row.get("completion", "")).strip() != task.canonical_solution.strip():
            completion_mismatch.append(bug_id)
        feedback = ExecutionResult.from_dict(bug.metadata["public_failure"])
        expected_prompt = build_repair_prompt(
            task, bug.buggy_code, feedback, FeedbackCondition.STRUCTURED
        )
        if str(row.get("prompt", "")).strip() != expected_prompt.strip():
            prompt_mismatch.append(bug_id)
        prompt_lengths.append(len(str(row.get("prompt", ""))))
        completion_lengths.append(len(str(row.get("completion", ""))))
    bug_ids = set(bugs)
    orphan_bugs = sorted(bug_ids - seen_bug_ids)
    return {
        "rows": len(rows),
        "missing_or_empty_fields": missing_fields,
        "unknown_bug_or_task_rows": unknown_bugs,
        "duplicate_bug_rows": duplicate_rows,
        "completion_mismatch_bugs": completion_mismatch,
        "prompt_mismatch_bugs": prompt_mismatch,
        "bugs_without_sft_rows": orphan_bugs,
        "prompt_chars": _length_stats(prompt_lengths),
        "completion_chars": _length_stats(completion_lengths),
        "approx_prompt_tokens": _length_stats((length + 3) // 4 for length in prompt_lengths),
        "approx_completion_tokens": _length_stats(
            (length + 3) // 4 for length in completion_lengths
        ),
    }


def _cross_split_report(groups: dict[str, list[RepairTask]]) -> dict[str, Any]:
    task_id_overlaps: dict[str, list[str]] = {}
    signature_owners: dict[str, list[str]] = {}
    for split, tasks in groups.items():
        for task in tasks:
            task_id_overlaps.setdefault(task.task_id, []).append(split)
            signature = _digest(f"{task.prompt}\n{task.canonical_solution}")
            signature_owners.setdefault(signature, []).append(f"{split}:{task.task_id}")
    return {
        "task_id_overlap_groups": [owners for owners in task_id_overlaps.values() if len(owners) > 1],
        "duplicate_prompt_solution_groups": [
            owners for owners in signature_owners.values() if len(owners) > 1
        ],
    }


def _status(check: dict[str, Any], warning_keys: set[str] | None = None) -> str:
    warning_keys = warning_keys or set()
    failures = []
    warnings = []
    for key, value in check.items():
        if not isinstance(value, list) or not value:
            continue
        if key in warning_keys:
            warnings.append(key)
        else:
            failures.append(key)
    if failures:
        return "fail"
    if warnings:
        return "warn"
    return "pass"


def build_report(
    train_tasks: list[RepairTask],
    validation_tasks: list[RepairTask],
    test_tasks: list[RepairTask],
    train_bugs: list[BugCase],
    train_sft: list[dict[str, Any]],
    tokenizer_lengths: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_groups = {
        "train": train_tasks,
        "validation": validation_tasks,
        "test": test_tasks,
    }
    train_task_index = {task.task_id: task for task in train_tasks}
    bug_report, bug_index = _check_bugs(train_task_index, train_bugs)
    sft_report = _check_sft(train_task_index, bug_index, train_sft)
    task_reports = {name: _check_task_group(tasks) for name, tasks in task_groups.items()}
    cross_split = _cross_split_report(task_groups)
    task_issues = {
        split: {
            key: value
            for key, value in details.items()
            if isinstance(value, list) and value
        }
        for split, details in task_reports.items()
    }
    task_status = "fail" if any(
        key in issues
        for issues in task_issues.values()
        for key in ("unparseable_reference_tasks", "missing_entry_point_tasks")
    ) else ("warn" if task_issues else "pass")
    checks = {
        "task_integrity": {
            "status": task_status,
            "details": {"by_split": task_reports, "issues": task_issues},
        },
        "bug_integrity": {
            "status": _status(bug_report, {"duplicate_buggy_code_groups"}),
            "details": bug_report,
        },
        "sft_integrity": {"status": _status(sft_report), "details": sft_report},
        "split_leakage": {
            "status": _status(cross_split),
            "details": cross_split,
        },
    }
    statuses = [check["status"] for check in checks.values()]
    overall = "fail" if "fail" in statuses else ("warn" if "warn" in statuses else "pass")
    recommendations = [
        "训练前保留 train_audited_tasks.jsonl 的参考答案审计结果；不要把 rejected 文件混入 SFT。",
        "若 prompt/completion mismatch 或 public feedback 无效，先修复数据生成流程再训练。",
        "正式实验固定任务级 train/validation/test 划分，并报告同题不同错误版本只能出现在一个 split。",
        "当前 train SFT 样本只有运行时错误和少量逻辑/边界错误时，应做按 bug_type 的分层统计，避免只学会 NameError 修复。",
    ]
    sft_lengths = {
        "prompt_chars": sft_report["prompt_chars"],
        "completion_chars": sft_report["completion_chars"],
        "approx_prompt_tokens": sft_report["approx_prompt_tokens"],
        "approx_completion_tokens": sft_report["approx_completion_tokens"],
    }
    if tokenizer_lengths is not None:
        sft_lengths["tokenizer"] = tokenizer_lengths
    return {
        "decision": overall,
        "inputs": {
            "train_tasks": len(train_tasks),
            "validation_tasks": len(validation_tasks),
            "test_tasks": len(test_tasks),
            "train_bugs": len(train_bugs),
            "train_sft": len(train_sft),
        },
        "checks": checks,
        "task_groups": task_reports,
        "bug_distribution": bug_report["bug_types"],
        "sft_lengths": sft_lengths,
        "recommendations": recommendations,
    }


def markdown_report(report: dict[str, Any]) -> str:
    def list_count(value: Any) -> int:
        if isinstance(value, list):
            return len(value)
        if isinstance(value, dict):
            return sum(list_count(item) for item in value.values())
        return 0

    checks = report["checks"]
    lines = [
        "# LoRA 训练前数据质量报告",
        "",
        f"结论：**{report['decision']}**",
        "",
        "## 数据规模",
        "",
        "| 项目 | 数量 |",
        "|---|---:|",
    ]
    for key, value in report["inputs"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## 检查结果", "", "| 检查 | 状态 | 重点问题数量 |", "|---|---|---:|"])
    for name, check in checks.items():
        details = check["details"]
        count = list_count(details)
        lines.append(f"| {name} | {check['status']} | {count} |")
    lines.extend(["", "## Bug 分布", "", "| 类型 | 数量 |", "|---|---:|"])
    for bug_type, count in report["bug_distribution"].items():
        lines.append(f"| {bug_type} | {count} |")
    lengths = report["sft_lengths"]
    lines.extend(
        [
            "",
            "## SFT 长度",
            "",
            "提示词近似 token 数按字符数除以 4 估算，最终训练仍应以 tokenizer 实测为准。",
            "",
            "| 字段 | p50 | p95 | p99 | 最大值 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key in ("approx_prompt_tokens", "approx_completion_tokens"):
        value = lengths[key]
        lines.append(f"| {key} | {value['p50']} | {value['p95']} | {value['p99']} | {value['max']} |")
    tokenizer = lengths.get("tokenizer")
    if tokenizer:
        lines.extend(
            [
                "",
                (
                    f"Tokenizer 实测：`{tokenizer['tokenizer']}`。训练输入合并后的最大长度为 "
                    f"{tokenizer['combined_tokens']['max']} tokens（纯文本拼接，不含对话模板）。"
                ),
                "",
                "| 字段 | p50 | p95 | p99 | 最大值 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        keys = ["prompt_tokens", "completion_tokens", "combined_tokens"]
        if "training_chat_tokens" in tokenizer:
            keys.append("training_chat_tokens")
        for key in keys:
            value = tokenizer[key]
            lines.append(
                f"| tokenizer_{key} | {value['p50']} | {value['p95']} | "
                f"{value['p99']} | {value['max']} |"
            )
    lines.extend(["", "## 建议", ""])
    lines.extend(f"- {item}" for item in report["recommendations"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="audit SFT data before LoRA training")
    parser.add_argument("--train-tasks", required=True)
    parser.add_argument("--validation-tasks", required=True)
    parser.add_argument("--test-tasks", required=True)
    parser.add_argument("--train-bugs", required=True)
    parser.add_argument("--train-sft", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_sft = _load_sft(args.train_sft)
    report = build_report(
        load_tasks(args.train_tasks),
        load_tasks(args.validation_tasks),
        load_tasks(args.test_tasks),
        load_bugs(args.train_bugs),
        train_sft,
        _tokenizer_lengths(train_sft, args.tokenizer) if args.tokenizer else None,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output.with_suffix(".md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "output": str(output)}, indent=2))
    return 0 if report["decision"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
