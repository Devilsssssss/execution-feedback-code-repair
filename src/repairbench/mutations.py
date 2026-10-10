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
    ("runtime_name_error", "runtime_error", _RuntimeNameError),
)


def generate_mutations(code: str, max_per_mutator: int = 20) -> Iterable[Mutation]:
    ast.parse(code)
    for name, bug_type, transformer_type in MUTATORS:
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
