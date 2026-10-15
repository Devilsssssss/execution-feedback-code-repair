from scripts.audit_ast_control import (
    count_types,
    maximal_proportional_quotas,
    normalize_bug_type,
    select_rows,
)


def test_model_error_types_collapse_to_ast_families() -> None:
    assert normalize_bug_type("model_logic_error") == "logic_error"
    assert normalize_bug_type("model_runtime_error") == "runtime_error"
    assert count_types(
        [
            {"bug_type": "model_logic_error"},
            {"bug_type": "logic_error"},
            {"bug_type": "model_timeout_error"},
        ]
    ) == {"logic_error": 2, "runtime_error": 1}


def test_maximal_quotas_preserve_target_ratio() -> None:
    quotas, scale = maximal_proportional_quotas(
        {"boundary_condition": 2, "logic_error": 4, "runtime_error": 4},
        {"boundary_condition": 2, "logic_error": 2, "runtime_error": 8},
    )
    assert scale == 0.5
    assert quotas == {"boundary_condition": 1, "logic_error": 2, "runtime_error": 2}


def test_selection_is_seeded_and_respects_normalized_quotas() -> None:
    rows = [
        {"bug_id": "b1", "bug_type": "logic_error"},
        {"bug_id": "b2", "bug_type": "model_logic_error"},
        {"bug_id": "b3", "bug_type": "runtime_error"},
    ]
    selected = select_rows(rows, {"logic_error": 1, "runtime_error": 1}, seed=7)
    assert selected == select_rows(rows, {"logic_error": 1, "runtime_error": 1}, seed=7)
    assert count_types(selected) == {"logic_error": 1, "runtime_error": 1}

