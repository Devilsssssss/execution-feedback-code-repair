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


def load(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def audit(original: list[dict[str, Any]], contract: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    original_by_id = {str(row.get("bug_id")): row for row in original}
    contract_by_id = {str(row.get("bug_id")): row for row in contract}
    if len(original_by_id) != len(original):
        errors.append("original_duplicate_bug_ids")
    if len(contract_by_id) != len(contract):
        errors.append("contract_duplicate_bug_ids")
    if set(original_by_id) != set(contract_by_id):
        errors.append("bug_id_set_mismatch")
    checked = 0
    for bug_id, source in original_by_id.items():
        target = contract_by_id.get(bug_id)
        if target is None:
            continue
        if target.get("prompt") != source.get("prompt"):
            errors.append(f"prompt_changed:{bug_id}")
        if target.get("task_id") != source.get("task_id"):
            errors.append(f"task_changed:{bug_id}")
        source_completion = str(source.get("completion", "")).strip()
        target_completion = str(target.get("completion", ""))
        prefix, separator, code = target_completion.partition("\n")
        if not separator or not prefix.startswith("# FEEDBACK_CONTRACT "):
            errors.append(f"missing_contract_prefix:{bug_id}")
        elif code.strip() != source_completion:
            errors.append(f"reference_code_changed:{bug_id}")
        try:
            payload = json.loads(prefix.removeprefix("# FEEDBACK_CONTRACT "))
        except json.JSONDecodeError:
            errors.append(f"invalid_contract_json:{bug_id}")
            continue
        if payload != target.get("feedback_contract"):
            errors.append(f"contract_field_mismatch:{bug_id}")
        if payload.get("total", 0) <= payload.get("passed", 0):
            errors.append(f"contract_has_no_failure:{bug_id}")
        if not payload.get("error_type"):
            errors.append(f"contract_missing_failure_evidence:{bug_id}")
        checked += 1
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in contract
    )
    return {
        "protocol": "feedback-contract-sft-v3",
        "decision": "fail" if errors else "pass",
        "original_rows": len(original),
        "contract_rows": len(contract),
        "checked_rows": checked,
        "errors": errors,
        "contract_output_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit feedback-contract SFT targets")
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(load(args.original), load(args.contract))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
