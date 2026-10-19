from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"expected object at {path}:{line_number}")
                rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Check controlled repair experiments before training")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "artifacts/train_sft_ast_matched_v3_manifest.json",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "artifacts/controlled_experiment_preflight.summary.json",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, details: Any) -> None:
        checks[name] = {"status": "pass" if passed else "fail", "details": details}

    train_path = ROOT / manifest["train_file"]["path"]
    bug_path = ROOT / manifest["bug_file"]["path"]
    train_rows = _read_jsonl(train_path)
    bug_rows = _read_jsonl(bug_path)
    expected_counts = manifest["normalized_counts"]
    actual_counts = dict(sorted(Counter(str(row["bug_type"]) for row in train_rows).items()))
    check(
        "ast_exact_data",
        len(train_rows) == manifest["rows"]
        and len(bug_rows) == manifest["rows"]
        and actual_counts == expected_counts
        and _sha256(train_path) == manifest["train_file"]["sha256"]
        and _sha256(bug_path) == manifest["bug_file"]["sha256"],
        {
            "rows": len(train_rows),
            "bug_rows": len(bug_rows),
            "counts": actual_counts,
            "train_sha256": _sha256(train_path),
            "bug_sha256": _sha256(bug_path),
        },
    )

    docker_report = ROOT / manifest["supplement_docker_audit_report"]
    docker_summary = json.loads(docker_report.read_text(encoding="utf-8"))
    check(
        "docker_audit",
        docker_summary.get("status") == "pass"
        and docker_summary.get("reference_passes") == docker_summary.get("reference_checks")
        and docker_summary.get("bug_passes") == docker_summary.get("bug_checks") == 297,
        {
            "status": docker_summary.get("status"),
            "reference_passes": docker_summary.get("reference_passes"),
            "reference_checks": docker_summary.get("reference_checks"),
            "bug_passes": docker_summary.get("bug_passes"),
            "bug_checks": docker_summary.get("bug_checks"),
        },
    )

    tokenizer_report = ROOT / manifest["tokenizer_quality_report"]
    tokenizer_summary = json.loads(tokenizer_report.read_text(encoding="utf-8"))
    max_chat_tokens = tokenizer_summary["sft_lengths"]["tokenizer"]["training_chat_tokens"]["max"]
    max_length = manifest["training"]["max_length"]
    check(
        "tokenizer_length",
        max_chat_tokens <= max_length,
        {"max_training_chat_tokens": max_chat_tokens, "max_length": max_length},
    )

    confirmation_summary_path = ROOT / manifest["evaluation"]["confirmation_summary"]
    confirmation_summary = json.loads(confirmation_summary_path.read_text(encoding="utf-8"))
    confirmation_tasks = ROOT / manifest["evaluation"]["confirmation_tasks"]
    confirmation_bugs = ROOT / manifest["evaluation"]["confirmation_bugs"]
    confirmation_task_rows = _read_jsonl(confirmation_tasks)
    confirmation_bug_rows = _read_jsonl(confirmation_bugs)
    task_ids = {str(row["task_id"]) for row in confirmation_task_rows}
    bug_task_ids = {str(row["task_id"]) for row in confirmation_bug_rows}
    check(
        "confirmation_freeze",
        confirmation_summary.get("status") == "frozen-for-future-confirmation"
        and confirmation_summary.get("count") == len(confirmation_task_rows) == len(confirmation_bug_rows)
        and task_ids == bug_task_ids
        and 30 <= len(task_ids) <= 40,
        {
            "status": confirmation_summary.get("status"),
            "tasks": len(confirmation_task_rows),
            "bugs": len(confirmation_bug_rows),
            "blind_status": confirmation_summary.get("blind_status"),
        },
    )

    control_checks = {}
    for label, raw_path in manifest["evaluation"]["confirmation_controls"].items():
        if label == "expected_repair_at_1":
            continue
        control_path = ROOT / raw_path
        control = json.loads(control_path.read_text(encoding="utf-8"))
        control_checks[label] = {
            "cases": control.get("cases"),
            "repair_at_1": control.get("repair_at_1"),
        }
    check(
        "confirmation_controls",
        all(
            value["cases"] == len(task_ids)
            and value["repair_at_1"] == manifest["evaluation"]["confirmation_controls"]["expected_repair_at_1"]
            for value in control_checks.values()
        ),
        control_checks,
    )

    mixed_path = ROOT / "data/processed/mbpp/train_sft_mixed_controlled.jsonl"
    mixed_quality_path = ROOT / "artifacts/train_sft_mixed_controlled_quality_tokenizer.json"
    mixed_quality = json.loads(mixed_quality_path.read_text(encoding="utf-8"))
    mixed_max = mixed_quality["sft_lengths"]["tokenizer"]["training_chat_tokens"]["max"]
    check(
        "mixed_tokenizer_length",
        mixed_path.exists() and mixed_max <= max_length,
        {"max_training_chat_tokens": mixed_max, "max_length": max_length},
    )

    failed = sorted(name for name, result in checks.items() if result["status"] != "pass")
    summary = {
        "status": "pass" if not failed else "fail",
        "manifest": str(args.manifest),
        "checks": checks,
        "failed_checks": failed,
        "training_status": "ready-for-remote-training" if not failed else "blocked",
        "model_training_performed": False,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
