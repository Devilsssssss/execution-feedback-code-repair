from pathlib import Path

from repairbench.io import load_bugs, load_tasks
from scripts.audit_sft_data import _load_sft, build_report

ROOT = Path(__file__).resolve().parents[1]


def test_mbpp_train_sft_quality_report_has_no_sft_integrity_failures() -> None:
    report = build_report(
        load_tasks(ROOT / "data/processed/mbpp/train_audited_tasks.jsonl"),
        load_tasks(ROOT / "data/processed/mbpp/validation_audited_tasks.jsonl"),
        load_tasks(ROOT / "data/processed/mbpp/test_audited_tasks.jsonl"),
        load_bugs(ROOT / "data/processed/mbpp/train_bugs.jsonl"),
        _load_sft(ROOT / "data/processed/mbpp/train_sft.jsonl"),
    )

    assert report["checks"]["sft_integrity"]["status"] == "pass"
    assert report["checks"]["split_leakage"]["status"] == "pass"
    assert report["inputs"]["train_sft"] == report["inputs"]["train_bugs"]
