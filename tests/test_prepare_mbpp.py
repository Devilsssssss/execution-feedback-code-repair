import json

from scripts.prepare_mbpp import deterministic_splits, load_local_rows, normalize_setup


def test_local_json_and_jsonl_loaders(tmp_path) -> None:
    rows = [{"task_id": 2}, {"task_id": 1}]
    json_path = tmp_path / "tasks.json"
    jsonl_path = tmp_path / "tasks.jsonl"
    json_path.write_text(json.dumps(rows), encoding="utf-8")
    jsonl_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    assert load_local_rows(json_path) == rows
    assert load_local_rows(jsonl_path) == rows


def test_deterministic_splits_are_task_disjoint_and_ordered() -> None:
    rows = [{"task_id": number} for number in range(1, 11)]
    splits = deterministic_splits(rows)

    assert [row["task_id"] for row in splits["train"]] == list(range(1, 9))
    assert [row["task_id"] for row in splits["validation"]] == [9]
    assert [row["task_id"] for row in splits["test"]] == [10]


def test_normalize_setup_supports_both_mbpp_formats() -> None:
    assert normalize_setup({"test_setup_code": "import math"}) == "import math"
    assert normalize_setup({"test_imports": ["import math", "from fractions import Fraction"]}) == (
        "import math\nfrom fractions import Fraction"
    )
