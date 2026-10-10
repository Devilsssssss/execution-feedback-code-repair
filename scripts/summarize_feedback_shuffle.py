from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_paired_results import compare_pair, load_records, summarize_run
from scripts.audit_feedback_shuffle import audit_rows


def parse_seed_spec(spec: str) -> tuple[int, Path]:
    if "=" not in spec:
        raise ValueError(f"expected SEED=RESULTS_JSONL, got {spec!r}")
    raw_seed, raw_path = spec.split("=", 1)
    return int(raw_seed), Path(raw_path)


def summarize_shuffles(
    correct: dict[str, dict[str, Any]],
    shuffled: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    if not shuffled:
        raise ValueError("at least one shuffled-feedback run is required")
    correct_summary = summarize_run(correct)
    runs = {}
    successes = []
    for seed, rows in sorted(shuffled.items()):
        audit = audit_rows(list(rows.values()))
        if audit["permutation_seed"] != seed:
            raise ValueError(f"seed label does not match records for seed {seed}")
        paired = compare_pair(correct, rows, "repair_at_1")
        run_summary = summarize_run(rows)
        success_count = int(run_summary["successes_at_1"])
        successes.append(success_count)
        runs[str(seed)] = {
            "successes": success_count,
            "repair_at_1": run_summary["repair_at_1"],
            "delta_from_correct_pp": 100
            * (run_summary["repair_at_1"] - correct_summary["repair_at_1"]),
            "correct_only": paired["left_only"],
            "shuffled_only": paired["right_only"],
            "exact_mcnemar_p": paired["exact_mcnemar_p"],
            "fixed_points": audit["fixed_points"],
            "unique_sources": audit["unique_sources"],
            "mapping_sha256": audit["mapping_sha256"],
        }
    cases = int(correct_summary["cases"])
    mean_successes = statistics.fmean(successes)
    return {
        "cases": cases,
        "correct_feedback": {
            "successes": int(correct_summary["successes_at_1"]),
            "repair_at_1": correct_summary["repair_at_1"],
        },
        "shuffle_seeds": sorted(shuffled),
        "shuffled_feedback": {
            "runs": runs,
            "mean_successes": mean_successes,
            "mean_repair_at_1": mean_successes / cases,
            "min_successes": min(successes),
            "max_successes": max(successes),
            "population_std_successes": statistics.pstdev(successes),
            "all_runs_below_correct": all(
                value < int(correct_summary["successes_at_1"]) for value in successes
            ),
            "uncorrected_p_below_0_05": sum(
                run["exact_mcnemar_p"] < 0.05 for run in runs.values()
            ),
        },
        "inference": (
            "directional_sensitivity_without_stable_significance; "
            "do_not_claim_feedback_improves_over_no_feedback"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="summarize repeated feedback permutations")
    parser.add_argument("--correct", type=Path, required=True)
    parser.add_argument("--shuffled", nargs="+", required=True, help="SEED=RESULTS_JSONL")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    _, correct = load_records(f"correct={args.correct}")
    shuffled = {}
    for spec in args.shuffled:
        seed, path = parse_seed_spec(spec)
        if seed in shuffled:
            raise ValueError(f"duplicate shuffle seed: {seed}")
        _, shuffled[seed] = load_records(f"shuffled={path}")
    report = summarize_shuffles(correct, shuffled)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
