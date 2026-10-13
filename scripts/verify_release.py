from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_json(relative_path: str) -> dict[str, object]:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    comparison = load_json("artifacts/humanevalfix_atomized_full_comparison.json")
    k3 = load_json("artifacts/humanevalfix_atomized_k3_feedback_ablation.json")
    shuffle_comparison = load_json(
        "artifacts/humanevalfix_feedback_shuffle_multiseed_comparison.json"
    )
    shuffle = load_json("artifacts/humanevalfix_feedback_shuffle_multiseed_summary.json")
    on_policy = load_json("artifacts/on_policy_later_success_summary.json")
    manifest = load_json("artifacts/train_sft_mixed_controlled_manifest.json")

    expected_at_1 = {
        "synthetic_none": 42,
        "synthetic_structured": 41,
        "mixed_none": 72,
        "mixed_structured": 71,
    }
    runs = comparison["runs"]
    for run_name, expected in expected_at_1.items():
        run = runs[run_name]
        require(run["cases"] == 143, f"unexpected case count for {run_name}")
        require(
            run["successes_at_1"] == expected,
            f"unexpected repair@1 successes for {run_name}",
        )

    paired = comparison["paired"]
    require(
        paired["synthetic_none__vs__mixed_none"]["repair_at_1"]["exact_mcnemar_p"]
        == 2.2721360437572002e-07,
        "unexpected no-feedback McNemar result",
    )
    require(
        paired["synthetic_structured__vs__mixed_structured"]["repair_at_1"]
        ["exact_mcnemar_p"]
        == 6.038535502739251e-07,
        "unexpected structured-feedback McNemar result",
    )
    require(
        paired["mixed_none__vs__mixed_structured"]["repair_at_1"]["exact_mcnemar_p"]
        == 1.0,
        "unexpected feedback ablation result",
    )

    require(k3["runs"]["none_k3"]["successes_at_k"] == 77, "unexpected no-feedback k=3")
    require(
        k3["runs"]["structured_k3"]["successes_at_k"] == 75,
        "unexpected structured-feedback k=3",
    )
    expected_shuffle_successes = {"42": 64, "43": 69, "44": 59, "45": 65, "46": 65}
    require(shuffle["cases"] == 143, "unexpected shuffled-feedback case count")
    require(shuffle["shuffle_seeds"] == [42, 43, 44, 45, 46], "unexpected shuffle seeds")
    require(shuffle["correct_feedback"]["successes"] == 71, "unexpected correct-feedback score")
    shuffled = shuffle["shuffled_feedback"]
    require(shuffled["mean_successes"] == 64.4, "unexpected shuffled mean")
    require(shuffled["min_successes"] == 59, "unexpected shuffled minimum")
    require(shuffled["max_successes"] == 69, "unexpected shuffled maximum")
    require(shuffled["population_std_successes"] == 3.2, "unexpected shuffled spread")
    require(shuffled["all_runs_below_correct"] is True, "not every shuffle is below correct")
    require(shuffled["uncorrected_p_below_0_05"] == 1, "unexpected significant-run count")
    for seed, expected in expected_shuffle_successes.items():
        run = shuffled["runs"][seed]
        require(run["successes"] == expected, f"unexpected shuffled score for seed {seed}")
        require(run["fixed_points"] == 0, f"seed {seed} contains fixed points")
        require(run["unique_sources"] == 143, f"seed {seed} lost source cases")
        comparison_run = shuffle_comparison["runs"][f"shuffled{seed}"]
        require(
            comparison_run["successes_at_1"] == expected,
            f"comparison mismatch for seed {seed}",
        )
        paired_run = shuffle_comparison["paired"][f"correct__vs__shuffled{seed}"][
            "repair_at_1"
        ]
        require(
            paired_run["exact_mcnemar_p"] == run["exact_mcnemar_p"],
            f"paired comparison mismatch for seed {seed}",
        )
    require(on_policy["later_success_samples"] == 4, "unexpected on-policy sample count")
    require(
        on_policy["training_decision"] == "insufficient_samples",
        "unexpected on-policy training decision",
    )
    require(manifest["rows"] == 1241, "unexpected mixed training row count")
    require(
        manifest["source_counts"]
        == {"synthetic_mutations": 772, "model_generated_errors": 469},
        "unexpected mixed training source counts",
    )

    documents = {
        "README.md": (
            "42/143",
            "72/143",
            "41/143",
            "71/143",
            "77/143",
            "75/143",
            "64.4/143",
            "59–69",
        ),
        "PROJECT_GUIDE.md": (
            "469",
            "772",
            "1241",
            "41/143",
            "71/143",
            "64.4/143",
            "59–69",
        ),
        "artifacts/experiment_report.md": (
            "42/143",
            "72/143",
            "41/143",
            "71/143",
            "64.4/143",
            "59–69",
        ),
    }
    for relative_path, required_fragments in documents.items():
        content = (ROOT / relative_path).read_text(encoding="utf-8")
        for fragment in required_fragments:
            require(fragment in content, f"{relative_path} is missing {fragment}")

    print(
        json.dumps(
            {
                "status": "pass",
                "checked_runs": list(expected_at_1),
                "checked_documents": list(documents),
                "mixed_training_rows": manifest["rows"],
                "shuffle_seeds": shuffle["shuffle_seeds"],
                "shuffled_feedback_mean_successes": shuffled["mean_successes"],
                "on_policy_later_success_samples": on_policy["later_success_samples"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
