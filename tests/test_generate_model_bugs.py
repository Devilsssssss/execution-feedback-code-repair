import ast

from repairbench.schemas import ExecutionResult
from repairbench.schemas import TestFailure as FailureRecord
from scripts.generate_model_bugs import (
    build_synthesis_prompt,
    classify_failure,
    has_entry_point,
    semantic_digest,
)


def test_synthesis_prompt_names_required_entry_point() -> None:
    prompt = build_synthesis_prompt("Add one to x.", "add_one")

    assert "Add one to x." in prompt
    assert "Required entry point: add_one" in prompt


def test_entry_point_and_semantic_digest() -> None:
    code = "def add_one(x):\n    return x + 1\n"

    assert has_entry_point(code, "add_one")
    assert not has_entry_point(code, "other")
    assert semantic_digest(code) == semantic_digest(ast.unparse(ast.parse(code)))


def test_failure_classification_distinguishes_logic_runtime_and_timeout() -> None:
    logic = ExecutionResult(
        status="completed",
        passed=0,
        total=1,
        failures=(FailureRecord(test="", error_type="AssertionError", message="failed"),),
    )
    runtime = ExecutionResult(
        status="completed",
        passed=0,
        total=1,
        failures=(FailureRecord(test="", error_type="NameError", message="failed"),),
    )
    timeout = ExecutionResult(status="timeout", passed=0, total=1)

    assert classify_failure(logic) == "logic_error"
    assert classify_failure(runtime) == "runtime_error"
    assert classify_failure(timeout) == "timeout_error"
