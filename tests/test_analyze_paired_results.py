from scripts.analyze_paired_results import compare_pair, exact_mcnemar_p, summarize_run


def record(bug_id: str, success: bool, *, public_passed: bool = False) -> dict:
    return {
        "bug_id": bug_id,
        "bug_type": "logic_error",
        "repair_at_1": success,
        "repair_at_k": success,
        "generated_characters": 10,
        "final_public": {
            "all_passed": public_passed or success,
            "status": "completed",
            "failures": [] if public_passed or success else [{"error_type": "AssertionError"}],
        },
        "final_hidden": {"status": "completed"},
    }


def test_summarize_run_separates_hidden_and_public_failures() -> None:
    summary = summarize_run(
        {
            "a": record("a", True),
            "b": record("b", False, public_passed=True),
            "c": record("c", False),
        }
    )

    assert summary["successes_at_1"] == 1
    assert summary["successes_at_k"] == 1
    assert summary["outcomes_at_k"] == {
        "public_fail": 1,
        "public_pass_hidden_fail": 1,
        "success": 1,
    }
    assert summary["public_failure_types"] == {"AssertionError": 1}


def test_compare_pair_counts_discordant_cases() -> None:
    left = {"a": record("a", True), "b": record("b", True), "c": record("c", False)}
    right = {"a": record("a", True), "b": record("b", False), "c": record("c", True)}

    comparison = compare_pair(left, right)

    assert comparison["both_succeed"] == 1
    assert comparison["left_only"] == 1
    assert comparison["right_only"] == 1
    assert comparison["neither_succeeds"] == 0
    assert comparison["by_bug_type"]["logic_error"] == {
        "both_succeed": 1,
        "left_only": 1,
        "right_only": 1,
    }
    assert exact_mcnemar_p(1, 1) == 1.0


def test_compare_pair_can_use_repair_at_k() -> None:
    left = {"a": record("a", False), "b": record("b", False)}
    right = {"a": record("a", False), "b": record("b", False)}
    left["a"]["repair_at_k"] = True
    right["b"]["repair_at_k"] = True

    comparison = compare_pair(left, right, "repair_at_k")

    assert comparison["left_only"] == 1
    assert comparison["right_only"] == 1
