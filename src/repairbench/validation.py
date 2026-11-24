from __future__ import annotations

from collections.abc import Sequence

from .executor import CodeExecutor
from .schemas import ExecutionResult


def execute_checked(
    executor: CodeExecutor,
    code: str,
    tests: Sequence[str],
    setup_code: str = "",
    *,
    stage: str,
) -> ExecutionResult:
    """Run code and reject infrastructure or malformed executor results."""
    result = executor.execute(code, tests, setup_code=setup_code)
    if result.status not in {"completed", "timeout"}:
        detail = result.failures[0].message if result.failures else "unknown executor error"
        if result.stderr.strip():
            detail = f"{detail}; stderr={result.stderr.strip()[-500:]}"
        raise RuntimeError(
            f"execution infrastructure failed during {stage} "
            f"(status={result.status}): {detail}"
        )
    if result.total != len(tests) or result.passed < 0 or result.passed > result.total:
        raise RuntimeError(
            f"executor returned an invalid test count during {stage}: "
            f"passed={result.passed}, total={result.total}, expected={len(tests)}"
        )
    return result


def require_reference_passes(result: ExecutionResult, stage: str) -> None:
    if result.all_passed:
        return
    raise ValueError(
        f"reference solution failed during {stage}: status={result.status}, "
        f"passed={result.passed}/{result.total}"
    )


def require_candidate_fails(result: ExecutionResult, stage: str) -> None:
    # Timeouts are intentional candidate failures in this benchmark (for
    # example, model_timeout_error); runner_error and unknown statuses have
    # already been rejected by execute_checked.
    if result.status == "timeout" or not result.all_passed:
        return
    raise ValueError(
        f"bug does not fail during {stage}: status={result.status}, "
        f"passed={result.passed}/{result.total}"
    )
