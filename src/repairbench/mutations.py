from __future__ import annotations

import ast
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Mutation:
    name: str
    bug_type: str
    occurrence: int
    code: str

    @property
    def mutation_id(self) -> str:
        return f"{self.name}:{self.occurrence}"


class _OccurrenceTransformer(ast.NodeTransformer):
    def __init__(self, target: int) -> None:
        self.target = target
        self.seen = 0
        self.changed = False

    def select(self) -> bool:
        current = self.seen
        self.seen += 1
        if current == self.target:
            self.changed = True
            return True
        return False


class _ComparisonBoundary(_OccurrenceTransformer):
    mapping: ClassVar[dict[type[ast.cmpop], Callable[[], ast.cmpop]]] = {
        ast.Lt: ast.LtE,
        ast.LtE: ast.Lt,
        ast.Gt: ast.GtE,
        ast.GtE: ast.Gt,
    }

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        for index, item in enumerate(node.ops):
            replacement = self.mapping.get(type(item))
            if replacement is not None and self.select():
                node.ops[index] = replacement()
                return node
        return node


class _ConstantBoundaryShift(_OccurrenceTransformer):
    """Shift an integer comparison threshold by one.

    This creates a second, distinct off-by-one family for tasks that compare
    against a literal.  It is intentionally limited to numeric constants so
    the mutation remains easy to explain and audit.
    """

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        for index, comparator in enumerate(node.comparators):
            if (
                isinstance(comparator, ast.Constant)
                and isinstance(comparator.value, int)
                and not isinstance(comparator.value, bool)
                and self.select()
            ):
                node.comparators[index] = ast.Constant(value=comparator.value + 1)
                return node
        return node


class _ArithmeticSwap(_OccurrenceTransformer):
    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        replacement: ast.operator | None = None
        if isinstance(node.op, ast.Add):
            replacement = ast.Sub()
        elif isinstance(node.op, ast.Sub):
            replacement = ast.Add()
        if replacement is not None and self.select():
            node.op = replacement
        return node


class _BooleanSwap(_OccurrenceTransformer):
    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        replacement: ast.boolop | None = None
        if isinstance(node.op, ast.And):
            replacement = ast.Or()
        elif isinstance(node.op, ast.Or):
            replacement = ast.And()
        if replacement is not None and self.select():
            node.op = replacement
        return node


class _AggregationSwap(_OccurrenceTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in {"min", "max"} and self.select():
            node.func.id = "max" if node.func.id == "min" else "min"
        return node


class _ComparisonLogicSwap(_OccurrenceTransformer):
    """Invert equality and membership predicates without changing boundaries."""

    mapping: ClassVar[dict[type[ast.cmpop], Callable[[], ast.cmpop]]] = {
        ast.Eq: ast.NotEq,
        ast.NotEq: ast.Eq,
        ast.In: ast.NotIn,
        ast.NotIn: ast.In,
        ast.Is: ast.IsNot,
        ast.IsNot: ast.Is,
    }

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        for index, item in enumerate(node.ops):
            replacement = self.mapping.get(type(item))
            if replacement is not None and self.select():
                node.ops[index] = replacement()
                return node
        return node


class _ConditionNegate(_OccurrenceTransformer):
    """Negate an if/while condition, preserving the original expression tree."""

    def visit_If(self, node: ast.If) -> ast.AST:
        self.generic_visit(node)
        if self.select():
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
        return node

    def visit_While(self, node: ast.While) -> ast.AST:
        self.generic_visit(node)
        if self.select():
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
        return node


class _IfBranchSwap(_OccurrenceTransformer):
    """Swap the two explicit branches of an if statement."""

    def visit_If(self, node: ast.If) -> ast.AST:
        self.generic_visit(node)
        if node.orelse and self.select():
            node.body, node.orelse = node.orelse, node.body
        return node


class _TernaryBranchSwap(_OccurrenceTransformer):
    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        self.generic_visit(node)
        if self.select():
            node.body, node.orelse = node.orelse, node.body
        return node


class _BooleanOperandDrop(_OccurrenceTransformer):
    """Remove one operand from a three-or-more operand boolean expression."""

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        if len(node.values) >= 3 and self.select():
            node.values.pop()
        return node


class _LoopElseSwap(_OccurrenceTransformer):
    """Swap loop body and else clauses when both are present."""

    def visit_For(self, node: ast.For) -> ast.AST:
        self.generic_visit(node)
        if node.orelse and self.select():
            node.body, node.orelse = node.orelse, node.body
        return node

    def visit_While(self, node: ast.While) -> ast.AST:
        self.generic_visit(node)
        if node.orelse and self.select():
            node.body, node.orelse = node.orelse, node.body
        return node


class _IfExpConditionNegate(_OccurrenceTransformer):
    """Negate the condition of a conditional expression."""

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        self.generic_visit(node)
        if self.select():
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
        return node


class _LoopRangeShift(_OccurrenceTransformer):
    """Shift one literal bound in a range() call by one."""

    delta: ClassVar[int] = 1

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "range":
            for index, argument in enumerate(node.args):
                if (
                    isinstance(argument, ast.Constant)
                    and isinstance(argument.value, int)
                    and not isinstance(argument.value, bool)
                    and self.select()
                ):
                    node.args[index] = ast.Constant(value=argument.value + self.delta)
                    return node
        return node


class _LoopRangeIncrement(_LoopRangeShift):
    delta = 1


class _LoopRangeDecrement(_LoopRangeShift):
    delta = -1


class _ReturnBooleanFlip(_OccurrenceTransformer):
    """Flip a literal boolean returned by a function."""

    def visit_Return(self, node: ast.Return) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, bool)
            and self.select()
        ):
            node.value = ast.Constant(value=not node.value.value)
        return node


class _ReturnNumericShift(_OccurrenceTransformer):
    """Shift one literal integer returned by a function."""

    def visit_Return(self, node: ast.Return) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, int)
            and not isinstance(node.value.value, bool)
            and self.select()
        ):
            node.value = ast.Constant(value=node.value.value + 1)
        return node


class _AugAssignSwap(_OccurrenceTransformer):
    mapping: ClassVar[dict[type[ast.operator], Callable[[], ast.operator]]] = {
        ast.Add: ast.Sub,
        ast.Sub: ast.Add,
        ast.Mult: ast.FloorDiv,
        ast.FloorDiv: ast.Mult,
    }

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        self.generic_visit(node)
        replacement = self.mapping.get(type(node.op))
        if replacement is not None and self.select():
            node.op = replacement()
        return node


class _ComparisonOperandSwap(_OccurrenceTransformer):
    """Swap comparison operands without adjusting the operator."""

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if len(node.ops) == 1 and len(node.comparators) == 1 and self.select():
            left, right = node.left, node.comparators[0]
            node.left, node.comparators[0] = right, left
        return node


class _SubscriptIndexShift(_OccurrenceTransformer):
    """Shift an integer list/string index by one."""

    delta: ClassVar[int] = 1

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, int)
            and not isinstance(node.slice.value, bool)
            and self.select()
        ):
            node.slice = ast.Constant(value=node.slice.value + self.delta)
        return node


class _SubscriptIndexIncrement(_SubscriptIndexShift):
    delta = 1


class _SubscriptIndexDecrement(_SubscriptIndexShift):
    delta = -1


class _SliceBoundShift(_OccurrenceTransformer):
    """Shift one literal slice endpoint by one."""

    delta: ClassVar[int] = 1

    def visit_Slice(self, node: ast.Slice) -> ast.AST:
        self.generic_visit(node)
        for field in ("lower", "upper", "step"):
            value = getattr(node, field)
            if (
                isinstance(value, ast.Constant)
                and isinstance(value.value, int)
                and not isinstance(value.value, bool)
                and self.select()
            ):
                setattr(node, field, ast.Constant(value=value.value + self.delta))
                return node
        return node


class _SliceBoundIncrement(_SliceBoundShift):
    delta = 1


class _SliceBoundDecrement(_SliceBoundShift):
    delta = -1


class _UnaryNumericSignFlip(_OccurrenceTransformer):
    """Flip the sign of a literal numeric unary expression."""

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))
            and not isinstance(node.operand.value, bool)
            and isinstance(node.op, (ast.UAdd, ast.USub))
            and self.select()
        ):
            node.op = ast.USub() if isinstance(node.op, ast.UAdd) else ast.UAdd()
        return node


class _BooleanConstantFlip(_OccurrenceTransformer):
    """Flip a literal boolean used by the implementation."""

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, bool) and self.select():
            return ast.Constant(value=not node.value)
        return node


class _BreakContinueSwap(_OccurrenceTransformer):
    """Swap one loop-control statement."""

    def visit_Break(self, node: ast.Break) -> ast.AST:
        if self.select():
            return ast.Continue()
        return node

    def visit_Continue(self, node: ast.Continue) -> ast.AST:
        if self.select():
            return ast.Break()
        return node


class _ListLiteralReverse(_OccurrenceTransformer):
    """Reverse one literal list or tuple used by the implementation."""

    def visit_List(self, node: ast.List) -> ast.AST:
        self.generic_visit(node)
        if len(node.elts) >= 2 and self.select():
            node.elts.reverse()
        return node

    def visit_Tuple(self, node: ast.Tuple) -> ast.AST:
        self.generic_visit(node)
        if len(node.elts) >= 2 and self.select():
            node.elts.reverse()
        return node


class _StringCaseSwap(_OccurrenceTransformer):
    """Change one literal string's case without changing program structure."""

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, str) and node.value and self.select():
            swapped = node.value.swapcase()
            if swapped != node.value:
                return ast.Constant(value=swapped)
        return node


class _RuntimeNameError(_OccurrenceTransformer):
    protected: ClassVar[set[str]] = {
        "range",
        "len",
        "enumerate",
        "zip",
        "sum",
        "min",
        "max",
        "sorted",
        "set",
        "list",
        "dict",
        "tuple",
        "abs",
        "all",
        "any",
    }

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if (
            isinstance(node.ctx, ast.Load)
            and node.id not in self.protected
            and not node.id.startswith("__")
            and self.select()
        ):
            node.id = f"{node.id}_missing"
        return node


MUTATORS: tuple[tuple[str, str, type[_OccurrenceTransformer]], ...] = (
    ("comparison_boundary", "boundary_condition", _ComparisonBoundary),
    ("constant_boundary_shift", "boundary_condition", _ConstantBoundaryShift),
    ("arithmetic_swap", "logic_error", _ArithmeticSwap),
    ("boolean_swap", "logic_error", _BooleanSwap),
    ("aggregation_swap", "logic_error", _AggregationSwap),
    ("comparison_logic_swap", "logic_error", _ComparisonLogicSwap),
    ("condition_negate", "logic_error", _ConditionNegate),
    ("if_branch_swap", "logic_error", _IfBranchSwap),
    ("ternary_branch_swap", "logic_error", _TernaryBranchSwap),
    ("boolean_operand_drop", "logic_error", _BooleanOperandDrop),
    ("loop_else_swap", "logic_error", _LoopElseSwap),
    ("ifexp_condition_negate", "logic_error", _IfExpConditionNegate),
    ("loop_range_increment", "logic_error", _LoopRangeIncrement),
    ("loop_range_decrement", "logic_error", _LoopRangeDecrement),
    ("return_boolean_flip", "logic_error", _ReturnBooleanFlip),
    ("return_numeric_shift", "logic_error", _ReturnNumericShift),
    ("augassign_swap", "logic_error", _AugAssignSwap),
    ("comparison_operand_swap", "logic_error", _ComparisonOperandSwap),
    ("subscript_index_increment", "logic_error", _SubscriptIndexIncrement),
    ("subscript_index_decrement", "logic_error", _SubscriptIndexDecrement),
    ("slice_bound_increment", "logic_error", _SliceBoundIncrement),
    ("slice_bound_decrement", "logic_error", _SliceBoundDecrement),
    ("unary_numeric_sign_flip", "logic_error", _UnaryNumericSignFlip),
    ("boolean_constant_flip", "logic_error", _BooleanConstantFlip),
    ("break_continue_swap", "logic_error", _BreakContinueSwap),
    ("list_literal_reverse", "logic_error", _ListLiteralReverse),
    ("string_case_swap", "logic_error", _StringCaseSwap),
    ("runtime_name_error", "runtime_error", _RuntimeNameError),
)


SUPPLEMENTAL_LOGIC_MUTATOR_NAMES: frozenset[str] = frozenset(
    {
        "ifexp_condition_negate",
        "loop_range_increment",
        "loop_range_decrement",
        "return_boolean_flip",
        "return_numeric_shift",
        "augassign_swap",
        "comparison_operand_swap",
        "subscript_index_increment",
        "subscript_index_decrement",
        "slice_bound_increment",
        "slice_bound_decrement",
        "unary_numeric_sign_flip",
        "boolean_constant_flip",
        "break_continue_swap",
        "list_literal_reverse",
        "string_case_swap",
    }
)


def generate_mutations(
    code: str,
    max_per_mutator: int = 20,
    include_names: set[str] | frozenset[str] | None = None,
) -> Iterable[Mutation]:
    ast.parse(code)
    for name, bug_type, transformer_type in MUTATORS:
        if include_names is not None and name not in include_names:
            continue
        for occurrence in range(max_per_mutator):
            tree = ast.parse(code)
            transformer = transformer_type(occurrence)
            mutated = transformer.visit(tree)
            if not transformer.changed:
                break
            ast.fix_missing_locations(mutated)
            mutated_code = ast.unparse(mutated).strip() + "\n"
            compile(mutated_code, "<mutated>", "exec")
            if mutated_code != code.strip() + "\n":
                yield Mutation(name, bug_type, occurrence, mutated_code)
