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
import os
import sys
import time
import traceback
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

MAX_CAPTURED_OUTPUT_CHARS = 2000


class BoundedTextCapture(io.TextIOBase):
    """Keep only a bounded tail of candidate output while preserving print()."""

    def __init__(self, max_chars: int = MAX_CAPTURED_OUTPUT_CHARS) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self.max_chars = max_chars
        self.truncated = False
        self._chunks: deque[str] = deque()
        self._length = 0

    @property
    def encoding(self) -> str:
        return "utf-8"

    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        if not isinstance(value, str):
            raise TypeError("write() argument must be str")
        if not value:
            return 0

        self._chunks.append(value)
        self._length += len(value)
        overflow = self._length - self.max_chars
        if overflow > 0:
            self.truncated = True
        while overflow > 0:
            first = self._chunks[0]
            if len(first) <= overflow:
                self._chunks.popleft()
                self._length -= len(first)
                overflow -= len(first)
            else:
                self._chunks[0] = first[overflow:]
                self._length -= overflow
                overflow = 0
        return len(value)

    def getvalue(self) -> str:
        value = "".join(self._chunks)
        if not self.truncated:
            return value
        marker = "[output truncated; showing final characters]\n"
        marker = marker[: self.max_chars]
        tail_length = self.max_chars - len(marker)
        tail = value[-tail_length:] if tail_length else ""
        return marker + tail


@contextmanager
def suppress_native_output() -> Iterator[None]:
    """Keep os.write(1/2, ...) from bypassing bounded Python stream capture."""
    sys.stdout.flush()
    sys.stderr.flush()
    null_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fds: dict[int, int] = {}
    try:
        for fd in (1, 2):
            saved_fds[fd] = os.dup(fd)
            os.dup2(null_fd, fd)
        yield
    finally:
        for fd, saved_fd in saved_fds.items():
            os.dup2(saved_fd, fd)
            os.close(saved_fd)
        os.close(null_fd)


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
    captured_stdout = BoundedTextCapture()
    captured_stderr = BoundedTextCapture()

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
        "stdout": captured_stdout.getvalue(),
        "stderr": captured_stderr.getvalue(),
        "duration_seconds": time.perf_counter() - started,
    }


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        with suppress_native_output():
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
