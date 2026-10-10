from scripts.audit_feedback_shuffle import audit_rows


def row(target: str, source: str) -> dict:
    return {
        "bug_id": target,
        "feedback_permuted": True,
        "feedback_permutation_seed": 42,
        "trajectory": [{"feedback_source_bug_id": source}],
    }


def test_audit_rows_counts_a_complete_derangement() -> None:
    report = audit_rows([row("a", "b"), row("b", "c"), row("c", "a")])

    assert report["cases"] == 3
    assert report["permutation_seed"] == 42
    assert report["fixed_points"] == 0
    assert report["unique_targets"] == 3
    assert report["unique_sources"] == 3
