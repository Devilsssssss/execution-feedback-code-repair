from repairbench.models import HeuristicRepairModel, NoOpRepairModel, extract_python_code


def test_extract_python_code_accepts_truncated_fence() -> None:
    text = """Here is the fix:

```python
def f(x):
    return x + 1
"""

    assert extract_python_code(text) == "def f(x):\n    return x + 1\n"


def test_extract_python_code_skips_unfenced_explanation() -> None:
    text = "The corrected implementation is below.\n\ndef f(x):\n    return x + 1\n"

    assert extract_python_code(text) == "def f(x):\n    return x + 1\n"


def test_extract_python_code_removes_prompt_tail() -> None:
    text = """```python
def f(x):
    return x + 1
```
# Public test execution feedback:
Execution summary:
"""

    assert extract_python_code(text) == "def f(x):\n    return x + 1\n"


def test_noop_model_extracts_current_code_without_feedback() -> None:
    prompt = """Problem:\nFix it.\n\nRequired entry point: f\n\nCurrent incorrect code:\ndef f(x):\n    return x\n\nPublic-test execution feedback:\nExecution summary:\n- Status: completed\n\nReturn a corrected implementation that generalizes beyond the visible tests."""

    assert NoOpRepairModel().repair(prompt) == "def f(x):\n    return x\n"


def test_heuristic_model_repairs_name_error_from_public_feedback() -> None:
    prompt = """Problem:\nFix it.\n\nRequired entry point: f\n\nCurrent incorrect code:\ndef f(value):\n    return value_missing + 1\n\nPublic-test execution feedback:\nExecution summary:\n- Status: completed\n- Error type: NameError\n- Message: name 'value_missing' is not defined\n\nReturn a corrected implementation that generalizes beyond the visible tests."""

    assert HeuristicRepairModel().repair(prompt) == "def f(value):\n    return value + 1\n"


def test_heuristic_model_leaves_non_name_errors_unchanged() -> None:
    prompt = """Problem:\nFix it.\n\nRequired entry point: f\n\nCurrent incorrect code:\ndef f(value):\n    return value - 1\n\nPublic-test execution feedback:\nExecution summary:\n- Error type: AssertionError\n\nReturn a corrected implementation that generalizes beyond the visible tests."""

    assert HeuristicRepairModel().repair(prompt) == "def f(value):\n    return value - 1\n"
