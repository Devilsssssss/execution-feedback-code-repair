import json
import sys
from pathlib import Path

from repairbench.schemas import BugCase, RepairTask

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_feedback_contract_sft import build_rows, load_jsonl  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_build_rows_preserves_prompt_and_references_feedback(tmp_path) -> None:
    task = RepairTask(
        task_id="t1",
        prompt="def f(x):\n    return x\n",
        entry_point="f",
        canonical_solution="def f(x):\n    return x + 1\n",
        public_tests=("assert f(1) == 2",),
        hidden_tests=("assert f(2) == 3",),
    )
    bug = BugCase(
        bug_id="b1",
        task_id="t1",
        buggy_code="def f(x):\n    return x\n",
        bug_type="logic_error",
        mutation="off-by-one",
        metadata={
            "public_failure": {
                "status": "completed",
                "passed": 0,
                "total": 1,
                "failures": [
                    {
                        "test": "assert f(1) == 2",
                        "error_type": "AssertionError",
                        "message": "assert 1 == 2",
                        "expected": "2",
                        "actual": "1",
                    }
                ],
            }
        },
    )
    sft_path = tmp_path / "sft.jsonl"
    _write_jsonl(
        sft_path,
        [
            {
                "prompt": task.prompt,
                "completion": task.canonical_solution,
                "task_id": "t1",
                "bug_id": "b1",
                "bug_type": "logic_error",
            }
        ],
    )
    bugs_path = tmp_path / "bugs.jsonl"
    _write_jsonl(bugs_path, [bug.to_dict()])
    tasks_path = tmp_path / "tasks.jsonl"
    _write_jsonl(tasks_path, [task.to_dict()])

    rows, summary = build_rows(load_jsonl(sft_path), [bug], [task])

    assert len(rows) == 1
    row = rows[0]
    assert row["prompt"] == task.prompt
    first_line, code = row["completion"].split("\n", 1)
    assert first_line.startswith("# FEEDBACK_CONTRACT ")
    payload = json.loads(first_line.removeprefix("# FEEDBACK_CONTRACT "))
    assert payload == {"passed": 0, "total": 1, "error_type": "AssertionError"}
    assert code.strip() == task.canonical_solution.strip()
    assert summary["rows"] == 1
    assert summary["bug_types"] == {"logic_error": 1}


def test_build_rows_rejects_passed_bug() -> None:
    task = RepairTask(
        task_id="t1",
        prompt="def f(x):\n    return x\n",
        entry_point="f",
        canonical_solution="def f(x):\n    return x\n",
        public_tests=("assert f(1) == 1",),
        hidden_tests=("assert f(2) == 2",),
    )
    bug = BugCase(
        bug_id="b1",
        task_id="t1",
        buggy_code="def f(x):\n    return x\n",
        bug_type="logic_error",
        mutation="no-op",
        metadata={
            "public_failure": {"status": "completed", "passed": 1, "total": 1, "failures": []}
        },
    )
    sft_row = {
        "prompt": task.prompt,
        "completion": task.canonical_solution,
        "task_id": "t1",
        "bug_id": "b1",
        "bug_type": "logic_error",
    }

    try:
        build_rows([sft_row], [bug], [task])
    except ValueError as error:
        assert "no public failure" in str(error)
    else:
        raise AssertionError("expected ValueError for passed bug")
