from repairbench.feedback import FeedbackCondition, format_feedback
from repairbench.schemas import ExecutionResult
from repairbench.schemas import TestFailure as FailureRecord


def test_structured_feedback_contains_comparison_details() -> None:
    result = ExecutionResult(
        status="completed",
        passed=0,
        total=1,
        failures=(
            FailureRecord(
                test="assert f([]) == 0",
                error_type="AssertionError",
                message="comparison assertion failed",
                expected="0",
                actual="1",
            ),
        ),
    )

    feedback = format_feedback(result, FeedbackCondition.STRUCTURED)

    assert "Expected: 0" in feedback
    assert "Actual: 1" in feedback
