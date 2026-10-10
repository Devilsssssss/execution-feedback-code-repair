from __future__ import annotations

import pytest

from scripts.build_controlled_mixed_sft import build_mixture, parse_quotas


def row(bug_id: str, bug_type: str, prompt: str) -> dict[str, str]:
    return {
        "bug_id": bug_id,
        "bug_type": bug_type,
        "prompt": prompt,
        "completion": "def f(): pass",
        "task_id": bug_id.split("__", 1)[0],
    }


def test_build_mixture_has_fixed_composition_and_is_deterministic() -> None:
    synthetic = [
        row("a__0", "boundary", "a0"),
        row("a__1", "boundary", "a1"),
        row("b__0", "runtime", "b0"),
        row("b__1", "runtime", "b1"),
    ]
    model = [row("c__model_0", "model_logic", "c0")]

    first, summary = build_mixture(synthetic, model, {"boundary": 1, "runtime": 1}, 7)
    second, _ = build_mixture(synthetic, model, {"boundary": 1, "runtime": 1}, 7)

    assert first == second
    assert len(first) == 3
    assert summary["source_counts"] == {
        "synthetic_mutations": 2,
        "model_generated_errors": 1,
    }


def test_build_mixture_rejects_duplicate_selected_prompts() -> None:
    synthetic = [row("a__0", "runtime", "same")]
    model = [row("b__model_0", "model_logic", "same")]

    with pytest.raises(ValueError, match="quota runtime=1 exceeds 0"):
        build_mixture(synthetic, model, {"runtime": 1}, 42)


def test_parse_quotas_rejects_invalid_value() -> None:
    assert parse_quotas(["logic=3"]) == {"logic": 3}
    with pytest.raises(ValueError, match="invalid quota"):
        parse_quotas(["logic:3"])
