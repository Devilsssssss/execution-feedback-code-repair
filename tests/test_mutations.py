import ast

from repairbench.executor import LocalSubprocessExecutor
from repairbench.mutations import SUPPLEMENTAL_LOGIC_MUTATOR_NAMES, generate_mutations


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


def test_comparison_logic_swap_inverts_equality() -> None:
    code = """def is_same(left, right):
    return left == right
"""
    mutation = next(
        item for item in generate_mutations(code) if item.name == "comparison_logic_swap"
    )

    assert mutation.bug_type == "logic_error"
    assert "left != right" in mutation.code


def test_condition_negate_wraps_if_test() -> None:
    code = """def sign(value):
    if value >= 0:
        return 1
    return -1
"""
    mutation = next(
        item for item in generate_mutations(code) if item.name == "condition_negate"
    )

    assert mutation.bug_type == "logic_error"
    assert "not value >= 0" in mutation.code


def test_if_branch_swap_exchanges_branches() -> None:
    code = """def classify(value):
    if value > 0:
        return "positive"
    else:
        return "non-positive"
"""
    mutation = next(
        item for item in generate_mutations(code) if item.name == "if_branch_swap"
    )

    assert mutation.bug_type == "logic_error"
    function = next(
        node for node in ast.walk(ast.parse(mutation.code)) if isinstance(node, ast.FunctionDef)
    )
    branch = next(node for node in ast.walk(function) if isinstance(node, ast.If))
    assert ast.literal_eval(branch.body[0].value) == "non-positive"
    assert ast.literal_eval(branch.orelse[0].value) == "positive"


def test_ternary_branch_swap_exchanges_branches() -> None:
    code = 'value = 1\nlabel = "even" if value % 2 == 0 else "odd"\n'
    mutation = next(
        item for item in generate_mutations(code) if item.name == "ternary_branch_swap"
    )

    assignment = next(
        node
        for node in ast.parse(mutation.code).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "label" for target in node.targets)
    )
    assert isinstance(assignment.value, ast.IfExp)
    assert ast.literal_eval(assignment.value.body) == "odd"
    assert ast.literal_eval(assignment.value.orelse) == "even"


def test_boolean_operand_drop_removes_last_operand() -> None:
    code = """def check(value):
    return value > 0 and value < 10 and value % 2 == 0
"""
    mutation = next(
        item for item in generate_mutations(code) if item.name == "boolean_operand_drop"
    )

    assert "% 2" not in mutation.code


def test_loop_else_swap_exchanges_clauses() -> None:
    code = """def first_even(values):
    for value in values:
        if value % 2 == 0:
            return value
    else:
        return -1
"""
    mutation = next(
        item for item in generate_mutations(code) if item.name == "loop_else_swap"
    )

    assert mutation.bug_type == "logic_error"
    assert mutation.code.index("return -1") < mutation.code.index("return value")


def test_supplemental_logic_mutators_can_be_selected_as_a_pool() -> None:
    code = """def choose(values, index):
    total = 0
    for value in range(index):
        total += value
    return values[index] if total else False
"""
    mutations = list(generate_mutations(code, include_names=SUPPLEMENTAL_LOGIC_MUTATOR_NAMES))

    assert mutations
    assert {mutation.bug_type for mutation in mutations} == {"logic_error"}
    assert {mutation.name for mutation in mutations} <= SUPPLEMENTAL_LOGIC_MUTATOR_NAMES
