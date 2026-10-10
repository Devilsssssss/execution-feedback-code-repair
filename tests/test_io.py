import pytest

from repairbench.io import check_task_disjointness
from repairbench.schemas import RepairTask


def task(task_id: str) -> RepairTask:
    return RepairTask(
        task_id=task_id,
        prompt="prompt",
        entry_point="f",
        canonical_solution="def f():\n    return 1\n",
        public_tests=("assert f() == 1",),
        hidden_tests=("assert f() == 1",),
    )


def test_split_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="data leakage"):
        check_task_disjointness([task("same")], [task("same")])


def test_disjoint_splits_are_accepted() -> None:
    check_task_disjointness([task("train")], [task("test")])
