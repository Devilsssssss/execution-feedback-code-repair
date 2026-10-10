from repairbench.executor import LocalSubprocessExecutor
from repairbench.mutations import generate_mutations


def test_boundary_mutation_is_valid_and_fails_test() -> None:
    code = """def is_between(value, lower, upper):
    return value > lower and value < upper
"""
    mutation = next(item for item in generate_mutations(code) if item.bug_type == "boundary_condition")
    result = LocalSubprocessExecutor(allow_unsafe=True).execute(
        mutation.code, ["assert is_between(0, 0, 10) is False", "assert is_between(5, 0, 10) is True"]
    )

    compile(mutation.code, "<mutation>", "exec")
    assert not result.all_passed


def test_runtime_mutation_produces_name_error() -> None:
    code = """def double(value):
    return value + value
"""
    mutation = next(item for item in generate_mutations(code) if item.bug_type == "runtime_error")
    result = LocalSubprocessExecutor(allow_unsafe=True).execute(
        mutation.code, ["assert double(2) == 4"]
    )

    assert result.failures[0].error_type == "NameError"


def test_constant_boundary_mutation_shifts_numeric_threshold() -> None:
    code = """def clamp(value):
    if value < 10:
        return 0
    return 1
"""
    mutation = next(
        item for item in generate_mutations(code) if item.name == "constant_boundary_shift"
    )

    assert mutation.bug_type == "boundary_condition"
    assert "value < 11" in mutation.code
