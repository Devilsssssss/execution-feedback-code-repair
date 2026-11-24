import hashlib
import json
from argparse import Namespace
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

from repairbench import cli
from repairbench.io import write_jsonl
from repairbench.models import NoOpRepairModel
from repairbench.provenance import _safe_identifier, sha256_file
from repairbench.schemas import BugCase, ExecutionResult, RepairTask


class ResultExecutor:
    def __init__(self, results: Iterable[ExecutionResult]) -> None:
        self._results = iter(results)

    def execute(self, code: str, tests, setup_code: str = "") -> ExecutionResult:
        return next(self._results)


def _result(passed: int) -> ExecutionResult:
    return ExecutionResult(status="completed", passed=passed, total=1)


def test_sha256_file_hashes_file_contents(tmp_path: Path) -> None:
    path = tmp_path / "input.jsonl"
    path.write_bytes(b"sample\n")

    assert sha256_file(path) == hashlib.sha256(b"sample\n").hexdigest()


def test_safe_identifier_removes_embedded_url_credentials() -> None:
    model = SimpleNamespace(
        model_id="https://user:secret@example.com/private-model?access_token=query-secret"
    )

    record = json.dumps({"model": _safe_identifier(model.model_id)})

    assert "secret" not in record
    assert "https://example.com/private-model" in record


def test_evaluate_writes_manifest_with_input_output_and_code_hashes(
    tmp_path: Path, monkeypatch
) -> None:
    task = RepairTask(
        task_id="task-1",
        prompt="Return x + 1.",
        entry_point="add_one",
        canonical_solution="def add_one(x):\n    return x + 1\n",
        public_tests=("assert add_one(1) == 2",),
        hidden_tests=("assert add_one(2) == 3",),
    )
    bug = BugCase(
        bug_id="bug-1",
        task_id=task.task_id,
        buggy_code="def add_one(x):\n    return x\n",
        bug_type="logic_error",
        mutation="manual",
    )
    tasks_path = tmp_path / "tasks.jsonl"
    bugs_path = tmp_path / "bugs.jsonl"
    output_path = tmp_path / "run.jsonl"
    write_jsonl(tasks_path, [task.to_dict()])
    write_jsonl(bugs_path, [bug.to_dict()])
    executor = ResultExecutor([_result(0), _result(0), _result(0), _result(0)])
    monkeypatch.setattr(cli, "_executor", lambda args: executor)
    args = Namespace(
        command="evaluate-noop",
        tasks=str(tasks_path),
        bugs=str(bugs_path),
        output=str(output_path),
        limit=None,
        executor="local",
        timeout=1.0,
        allow_unsafe_local=True,
        condition="none",
        max_repairs=1,
        shuffle_feedback_seed=None,
    )

    assert cli._run_evaluation(args, NoOpRepairModel(), run_name="noop_control") == 0
    manifest_path = tmp_path / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["configuration"]["command"] == "evaluate-noop"
    assert manifest["inputs"]["tasks"]["sha256"] == sha256_file(tasks_path)
    assert manifest["inputs"]["bugs"]["sha256"] == sha256_file(bugs_path)
    assert manifest["outputs"]["records"]["sha256"] == sha256_file(output_path)
    assert manifest["outputs"]["summary"]["sha256"] == sha256_file(
        tmp_path / "run.summary.json"
    )
    assert len(manifest["environment"]["source"]["sha256"]) == 64
    assert manifest["environment"]["git"]["available"] is True
    assert manifest["model"]["kind"] == "NoOpRepairModel"
    assert manifest["executor"]["requested"] == "local"
