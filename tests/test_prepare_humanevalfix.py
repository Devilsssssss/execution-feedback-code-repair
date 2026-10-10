from scripts.prepare_humanevalfix import (
    combine_declaration_and_body,
    convert_row,
    extract_example_tests,
)


def sample_row() -> dict:
    return {
        "task_id": "Python/0",
        "declaration": "from typing import List\n\ndef add_one(values: List[int]) -> List[int]:\n",
        "canonical_solution": "    return [value + 1 for value in values]\n",
        "buggy_solution": "    return [value - 1 for value in values]\n",
        "bug_type": "operator misuse",
        "failure_symptoms": "incorrect output",
        "entry_point": "add_one",
        "test": "def check(candidate):\n    assert candidate([1]) == [2]\ncheck(add_one)",
        "example_test": "def check(candidate):\n    assert candidate([]) == []\ncheck(add_one)",
        "instruction": "Increment every integer in the list.",
        "test_setup": "",
    }


def test_combine_declaration_and_body_preserves_indentation() -> None:
    code = combine_declaration_and_body("def add_one(value):\n", "    return value + 1\n")

    assert code == "def add_one(value):\n    return value + 1\n"
    compile(code, "<test>", "exec")


def test_convert_row_keeps_example_tests_separate_from_hidden_tests() -> None:
    task, bug = convert_row(sample_row())

    assert task["task_id"] == "humanevalfix_python_0"
    assert task["public_tests"] == ["assert add_one([]) == []\n"]
    assert task["hidden_tests"] == [sample_row()["test"]]
    assert bug["bug_type"] == "operator_misuse"
    assert bug["mutation"] == "human_written"
    assert bug["task_id"] == task["task_id"]


def test_extract_example_tests_splits_assertions_and_keeps_preamble() -> None:
    tests = extract_example_tests(
        """def check(candidate):
    import math
    assert candidate(1) == 2
    assert candidate(2) == 3
check(add_one)
""",
        "add_one",
    )

    assert tests == [
        "import math\nassert add_one(1) == 2\n",
        "import math\nassert add_one(2) == 3\n",
    ]
