from scripts.summarize_feedback_shuffle import summarize_shuffles


def record(bug_id: str, success: bool, source: str, seed: int) -> dict:
    return {
        "bug_id": bug_id,
        "bug_type": "logic_error",
        "repair_at_1": success,
        "repair_at_k": success,
        "generated_characters": 10,
        "feedback_permuted": True,
        "feedback_permutation_seed": seed,
        "trajectory": [{"feedback_source_bug_id": source}],
        "final_public": {
            "all_passed": success,
            "status": "completed",
            "failures": [] if success else [{"error_type": "AssertionError"}],
        },
        "final_hidden": {"status": "completed"},
    }


def test_summarize_shuffles_reports_seed_variation() -> None:
    correct = {
        "a": record("a", True, "a", 0),
        "b": record("b", True, "b", 0),
        "c": record("c", False, "c", 0),
    }
    shuffled = {
        42: {
            "a": record("a", True, "b", 42),
            "b": record("b", False, "c", 42),
            "c": record("c", False, "a", 42),
        },
        43: {
            "a": record("a", False, "c", 43),
            "b": record("b", False, "a", 43),
            "c": record("c", False, "b", 43),
        },
    }

    report = summarize_shuffles(correct, shuffled)

    assert report["correct_feedback"]["successes"] == 2
    assert report["shuffled_feedback"]["mean_successes"] == 0.5
    assert report["shuffled_feedback"]["min_successes"] == 0
    assert report["shuffled_feedback"]["max_successes"] == 1
    assert report["shuffled_feedback"]["all_runs_below_correct"] is True
