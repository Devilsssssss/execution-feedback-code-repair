from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
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


def extract_later_successes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples = []
    for row in rows:
        if bool(row["repair_at_1"]) or not bool(row["repair_at_k"]):
            continue
        trajectory = list(row["trajectory"])
        if len(trajectory) < 2:
            raise ValueError(f"later success has fewer than two repairs: {row['bug_id']}")
        successful_step = trajectory[-1]
        if not bool(successful_step["public_after"]["all_passed"]):
            raise ValueError(f"final trajectory step did not pass public tests: {row['bug_id']}")
        prompt = str(successful_step["prompt"])
        if "Public-test execution feedback:" not in prompt:
            raise ValueError(f"later-success prompt has no execution feedback: {row['bug_id']}")
        samples.append(
            {
                "prompt": prompt,
                "completion": str(successful_step["candidate_code"]).strip() + "\n",
                "task_id": str(row["task_id"]),
                "bug_id": str(row["bug_id"]),
                "bug_type": str(row["bug_type"]),
                "source_repair_index": int(successful_step["repair_index"]),
                "source": "on_policy_later_success",
            }
        )
    return samples


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="extract later-success on-policy SFT samples")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.results)
    samples = extract_later_successes(rows)
    digest = write_jsonl(args.output, samples)
    summary = {
        "input_cases": len(rows),
        "repair_at_1_successes": sum(bool(row["repair_at_1"]) for row in rows),
        "repair_at_k_successes": sum(bool(row["repair_at_k"]) for row in rows),
        "later_success_samples": len(samples),
        "by_repair_index": dict(
            sorted(Counter(str(row["source_repair_index"]) for row in samples).items())
        ),
        "by_bug_type": dict(sorted(Counter(row["bug_type"] for row in samples).items())),
        "output": str(args.output),
        "output_sha256": digest,
        "training_decision": "insufficient_samples" if len(samples) < 100 else "review_samples",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
