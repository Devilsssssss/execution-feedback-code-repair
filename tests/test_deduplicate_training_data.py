from types import SimpleNamespace

from scripts.deduplicate_training_data import find_duplicate_task_groups


def test_find_duplicate_task_groups_matches_near_duplicate_prompts() -> None:
    tasks = [
        SimpleNamespace(
            task_id="a",
            prompt="Write a function to interchange the first and last elements in a list.",
            canonical_solution="def swap(values):\n    return values\n",
        ),
        SimpleNamespace(
            task_id="b",
            prompt="Write a function to interchange the first and last element in a given list.",
            canonical_solution="def swap(values):\n    return values\n",
        ),
        SimpleNamespace(
            task_id="c",
            prompt="Write a function that adds two numbers.",
            canonical_solution="def add(a, b):\n    return a + b\n",
        ),
    ]

    assert find_duplicate_task_groups(tasks) == [["a", "b"]]
