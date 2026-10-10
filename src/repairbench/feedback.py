from __future__ import annotations

from enum import Enum

from .schemas import ExecutionResult


class FeedbackCondition(str, Enum):
    NONE = "none"
    RAW = "raw"
    STRUCTURED = "structured"


def format_feedback(result: ExecutionResult, condition: FeedbackCondition) -> str:
    if condition is FeedbackCondition.NONE:
        return ""
    if condition is FeedbackCondition.RAW:
        lines = [
            f"status={result.status}",
            f"passed={result.passed}/{result.total}",
        ]
        for failure in result.failures:
            lines.append(f"test: {failure.test}")
            lines.append(f"{failure.error_type}: {failure.message}")
            if failure.traceback:
                lines.append(failure.traceback)
        if result.stdout:
            lines.append(f"stdout:\n{result.stdout}")
        if result.stderr:
            lines.append(f"stderr:\n{result.stderr}")
        return "\n".join(lines)

    lines = [
        "Execution summary:",
        f"- Status: {result.status}",
        f"- Public tests passed: {result.passed}/{result.total}",
    ]
    for index, failure in enumerate(result.failures, start=1):
        lines.extend(
            [
                f"- Failure {index}",
                f"  - Test: {failure.test}",
                f"  - Error type: {failure.error_type}",
                f"  - Message: {failure.message or '(none)'}",
            ]
        )
        if failure.expected is not None:
            lines.append(f"  - Expected: {failure.expected}")
        if failure.actual is not None:
            lines.append(f"  - Actual: {failure.actual}")
    return "\n".join(lines)
