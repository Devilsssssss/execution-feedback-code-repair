from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any


def load_records(spec: str) -> tuple[str, dict[str, dict[str, Any]]]:
    if "=" not in spec:
        raise ValueError(f"expected LABEL=RESULTS_JSONL, got {spec!r}")
    label, raw_path = spec.split("=", 1)
    if not label.strip():
        raise ValueError("result label must not be empty")
    rows: dict[str, dict[str, Any]] = {}
    with Path(raw_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"expected an object at {raw_path}:{line_number}")
            bug_id = str(row["bug_id"])
            if bug_id in rows:
                raise ValueError(f"duplicate bug_id in {raw_path}: {bug_id}")
            rows[bug_id] = row
    return label, rows


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    lower = min(left_only, right_only)
    tail = sum(math.comb(discordant, value) for value in range(lower + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def summarize_run(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = list(rows.values())
    outcomes: Counter[str] = Counter()
    public_failure_types: Counter[str] = Counter()
    for row in values:
        if bool(row["repair_at_k"]):
            outcomes["success"] += 1
        elif bool(row["final_public"]["all_passed"]):
            outcomes["public_pass_hidden_fail"] += 1
        else:
            outcomes["public_fail"] += 1
        public_failure_types.update(
            str(failure["error_type"])
            for failure in row["final_public"].get("failures", [])
        )
    cases = len(values)
    successes_at_1 = sum(bool(row["repair_at_1"]) for row in values)
    successes_at_k = outcomes["success"]
    return {
        "cases": cases,
        "successes_at_1": successes_at_1,
        "successes_at_k": successes_at_k,
        "repair_at_1": successes_at_1 / cases if cases else 0.0,
        "repair_at_k": successes_at_k / cases if cases else 0.0,
        "outcomes_at_k": dict(sorted(outcomes.items())),
        "mean_generated_characters": (
            sum(int(row["generated_characters"]) for row in values) / cases if cases else 0.0
        ),
        "public_failure_types": dict(public_failure_types.most_common()),
        "runner_error_records": sum(
            row["final_public"]["status"] == "runner_error"
            or row["final_hidden"]["status"] == "runner_error"
            for row in values
        ),
    }


def compare_pair(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    outcome_key: str = "repair_at_1",
) -> dict[str, Any]:
    if outcome_key not in {"repair_at_1", "repair_at_k"}:
        raise ValueError(f"unsupported outcome key: {outcome_key}")
    if left.keys() != right.keys():
        missing_left = sorted(right.keys() - left.keys())
        missing_right = sorted(left.keys() - right.keys())
        raise ValueError(
            "paired runs contain different bug IDs; "
            f"missing_left={missing_left[:5]}, missing_right={missing_right[:5]}"
        )
    both = left_only = right_only = neither = 0
    discordance_by_bug_type: dict[str, Counter[str]] = {}
    for bug_id, left_row in left.items():
        left_type = str(left_row.get("bug_type", "unknown"))
        right_type = str(right[bug_id].get("bug_type", "unknown"))
        if left_type != right_type:
            raise ValueError(f"paired runs disagree on bug type for {bug_id}")
        type_counts = discordance_by_bug_type.setdefault(left_type, Counter())
        left_success = bool(left_row[outcome_key])
        right_success = bool(right[bug_id][outcome_key])
        if left_success and right_success:
            both += 1
            type_counts["both_succeed"] += 1
        elif left_success:
            left_only += 1
            type_counts["left_only"] += 1
        elif right_success:
            right_only += 1
            type_counts["right_only"] += 1
        else:
            neither += 1
            type_counts["neither_succeeds"] += 1
    return {
        "both_succeed": both,
        "left_only": left_only,
        "right_only": right_only,
        "neither_succeeds": neither,
        "exact_mcnemar_p": exact_mcnemar_p(left_only, right_only),
        "by_bug_type": {
            bug_type: dict(counts)
            for bug_type, counts in sorted(discordance_by_bug_type.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="analyze paired repair evaluation records")
    parser.add_argument("results", nargs="+", help="LABEL=path/to/results.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runs = dict(load_records(spec) for spec in args.results)
    report = {
        "runs": {label: summarize_run(rows) for label, rows in runs.items()},
        "paired": {
            f"{left_label}__vs__{right_label}": {
                metric: compare_pair(runs[left_label], runs[right_label], metric)
                for metric in ("repair_at_1", "repair_at_k")
            }
            for left_label, right_label in combinations(runs, 2)
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"runs": len(runs), "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
