from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


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


def audit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("shuffled-feedback results are empty")
    mappings = []
    seeds = set()
    for row in rows:
        trajectory = list(row["trajectory"])
        if len(trajectory) != 1:
            raise ValueError(f"expected exactly one repair for {row['bug_id']}")
        source_id = str(trajectory[0]["feedback_source_bug_id"])
        target_id = str(row["bug_id"])
        mappings.append((target_id, source_id))
        seeds.add(row["feedback_permutation_seed"])
        if not bool(row["feedback_permuted"]):
            raise ValueError(f"record is not marked as permuted: {target_id}")
    if len(seeds) != 1:
        raise ValueError(f"expected one permutation seed, got {sorted(seeds)}")
    mapping_text = "".join(f"{target}\t{source}\n" for target, source in sorted(mappings))
    return {
        "cases": len(rows),
        "permutation_seed": seeds.pop(),
        "fixed_points": sum(target == source for target, source in mappings),
        "unique_targets": len({target for target, _ in mappings}),
        "unique_sources": len({source for _, source in mappings}),
        "mapping_sha256": hashlib.sha256(mapping_text.encode("utf-8")).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="audit a shuffled-feedback evaluation")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = audit_rows(load_jsonl(args.results))
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    correct = comparison["runs"]["correct"]
    shuffled = comparison["runs"]["shuffled"]
    paired = comparison["paired"]["correct__vs__shuffled"]["repair_at_1"]
    report.update(
        {
            "correct_feedback_successes": correct["successes_at_1"],
            "shuffled_feedback_successes": shuffled["successes_at_1"],
            "correct_only": paired["left_only"],
            "shuffled_only": paired["right_only"],
            "exact_mcnemar_p": paired["exact_mcnemar_p"],
            "results_sha256": hashlib.sha256(args.results.read_bytes()).hexdigest(),
        }
    )
    if report["fixed_points"] != 0:
        raise ValueError("feedback permutation contains fixed points")
    if report["unique_targets"] != report["cases"]:
        raise ValueError("feedback permutation contains duplicate targets")
    if report["unique_sources"] != report["cases"]:
        raise ValueError("feedback permutation does not preserve all feedback sources")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
