from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _non_empty(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


@dataclass(frozen=True)
class RepairTask:
    task_id: str
    prompt: str
    entry_point: str
    canonical_solution: str
    public_tests: tuple[str, ...]
    hidden_tests: tuple[str, ...]
    setup_code: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _non_empty(self.task_id, "task_id"))
        object.__setattr__(self, "prompt", _non_empty(self.prompt, "prompt"))
        object.__setattr__(self, "entry_point", _non_empty(self.entry_point, "entry_point"))
        object.__setattr__(
            self, "canonical_solution", _non_empty(self.canonical_solution, "canonical_solution")
        )
        if not self.public_tests:
            raise ValueError(f"task {self.task_id} has no public tests")
        if not self.hidden_tests:
            raise ValueError(f"task {self.task_id} has no hidden tests")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RepairTask:
        return cls(
            task_id=str(value["task_id"]),
            prompt=str(value["prompt"]),
            entry_point=str(value["entry_point"]),
            canonical_solution=str(value["canonical_solution"]),
            public_tests=tuple(str(test) for test in value["public_tests"]),
            hidden_tests=tuple(str(test) for test in value["hidden_tests"]),
            setup_code=str(value.get("setup_code", "")),
            metadata=dict(value.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["public_tests"] = list(self.public_tests)
        value["hidden_tests"] = list(self.hidden_tests)
        return value


@dataclass(frozen=True)
class BugCase:
    bug_id: str
    task_id: str
    buggy_code: str
    bug_type: str
    mutation: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("bug_id", "task_id", "buggy_code", "bug_type", "mutation"):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BugCase:
        return cls(
            bug_id=str(value["bug_id"]),
            task_id=str(value["task_id"]),
            buggy_code=str(value["buggy_code"]),
            bug_type=str(value["bug_type"]),
            mutation=str(value["mutation"]),
            metadata=dict(value.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TestFailure:
    test: str
    error_type: str
    message: str
    expected: str | None = None
    actual: str | None = None
    traceback: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TestFailure:
        return cls(
            test=str(value.get("test", "")),
            error_type=str(value.get("error_type", "UnknownError")),
            message=str(value.get("message", "")),
            expected=value.get("expected"),
            actual=value.get("actual"),
            traceback=value.get("traceback"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    passed: int
    total: int
    failures: tuple[TestFailure, ...] = ()
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0

    @property
    def all_passed(self) -> bool:
        return self.status == "completed" and self.total > 0 and self.passed == self.total

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExecutionResult:
        return cls(
            status=str(value.get("status", "runner_error")),
            passed=int(value.get("passed", 0)),
            total=int(value.get("total", 0)),
            failures=tuple(TestFailure.from_dict(item) for item in value.get("failures", [])),
            stdout=str(value.get("stdout", "")),
            stderr=str(value.get("stderr", "")),
            duration_seconds=float(value.get("duration_seconds", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "total": self.total,
            "all_passed": self.all_passed,
            "failures": [failure.to_dict() for failure in self.failures],
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
        }
