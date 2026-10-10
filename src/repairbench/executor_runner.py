"""Standalone child process that executes candidate code and tests.

This file intentionally has no package-relative imports so it can run inside a
minimal Docker image from a read-only bind mount.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import operator
import sys
import time
import traceback
from typing import Any

OPERATORS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda left, right: left in right,
    ast.NotIn: lambda left, right: left not in right,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}


def safe_repr(value: Any, limit: int = 500) -> str:
    try:
        rendered = repr(value)
    except Exception:  # noqa: BLE001 - candidate objects can raise arbitrary repr errors
        rendered = f"<{type(value).__name__}: repr failed>"
    if len(rendered) > limit:
        return rendered[:limit] + "..."
    return rendered


def evaluate_test(test_source: str, namespace: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    tree = ast.parse(test_source, mode="exec")
    assertions = [node for node in tree.body if isinstance(node, ast.Assert)]
    if len(assertions) == 1:
        assertion = assertions[0]
        prefix = [node for node in tree.body if node is not assertion]
        if prefix:
            exec(  # noqa: S102 - sandbox worker
                compile(ast.Module(body=prefix, type_ignores=[]), "<test-setup>", "exec"),
                namespace,
            )
        expression = assertion.test
        if isinstance(expression, ast.Compare) and len(expression.ops) == 1:
            left = eval(compile(ast.Expression(expression.left), "<test-left>", "eval"), namespace)
            right = eval(
                compile(ast.Expression(expression.comparators[0]), "<test-right>", "eval"),
                namespace,
            )
            compare = OPERATORS.get(type(expression.ops[0]))
            if compare is not None:
                passed = bool(compare(left, right))
                return passed, {
                    "actual": safe_repr(left),
                    "expected": safe_repr(right),
                    "message": "comparison assertion failed" if not passed else "",
                }
    exec(compile(tree, "<test>", "exec"), namespace)  # noqa: S102 - sandbox worker
    return True, {}


def run(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    tests = [str(test) for test in payload.get("tests", [])]
    failures: list[dict[str, Any]] = []
    passed = 0
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()

    for test in tests:
        namespace: dict[str, Any] = {"__name__": "__candidate__"}
        try:
            with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(
                captured_stderr
            ):
                exec(  # noqa: S102 - sandbox worker
                    compile(str(payload.get("setup_code", "")), "<setup>", "exec"), namespace
                )
                exec(  # noqa: S102 - sandbox worker
                    compile(str(payload["code"]), "<candidate>", "exec"), namespace
                )
                ok, details = evaluate_test(test, namespace)
            if ok:
                passed += 1
            else:
                failures.append(
                    {
                        "test": test,
                        "error_type": "AssertionError",
                        "message": details.get("message", "assertion failed"),
                        "expected": details.get("expected"),
                        "actual": details.get("actual"),
                    }
                )
        except BaseException as exc:  # noqa: BLE001 - isolate SystemExit from candidate code
            failures.append(
                {
                    "test": test,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": "".join(traceback.format_exception(exc))[-2000:],
                }
            )

    return {
        "status": "completed",
        "passed": passed,
        "total": len(tests),
        "failures": failures,
        "stdout": captured_stdout.getvalue()[-2000:],
        "stderr": captured_stderr.getvalue()[-2000:],
        "duration_seconds": time.perf_counter() - started,
    }


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        result = run(payload)
    except BaseException as exc:  # noqa: BLE001 - always serialize runner failures
        result = {
            "status": "runner_error",
            "passed": 0,
            "total": 0,
            "failures": [
                {
                    "test": "",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": "".join(traceback.format_exception(exc))[-2000:],
                }
            ],
        }
    sys.stdout.write(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
