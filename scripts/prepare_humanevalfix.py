from __future__ import annotations

import argparse
import ast
import json
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

from repairbench.io import write_jsonl


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"expected an object at {source}:{line_number}")
            rows.append(value)
    return rows


def combine_declaration_and_body(declaration: str, body: str) -> str:
    declaration = declaration.rstrip()
    body = body.rstrip()
    if not declaration or not body:
        raise ValueError("function declaration and body must not be empty")
    return f"{declaration}\n{body}\n"


class _RenameCheckArgument(ast.NodeTransformer):
    def __init__(self, argument: str, entry_point: str) -> None:
        self.argument = argument
        self.entry_point = entry_point

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id != self.argument:
            return node
        return ast.copy_location(ast.Name(id=self.entry_point, ctx=node.ctx), node)


def extract_example_tests(source: str, entry_point: str) -> list[str]:
    """Turn HumanEvalPack's public check function into independent assertions."""
    tree = ast.parse(textwrap.dedent(source))
    checks = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "check"
    ]
    if len(checks) != 1:
        raise ValueError("example_test must define exactly one check function")
    check = checks[0]
    arguments = check.args.posonlyargs + check.args.args
    if len(arguments) != 1 or check.args.vararg or check.args.kwarg or check.args.kwonlyargs:
        raise ValueError("check function must take exactly one positional argument")

    assertions = [node for node in check.body if isinstance(node, ast.Assert)]
    if not assertions:
        raise ValueError("check function contains no assertions")
    first_assert = next(index for index, node in enumerate(check.body) if isinstance(node, ast.Assert))
    preamble = check.body[:first_assert]
    if any(not isinstance(node, (ast.Import, ast.ImportFrom, ast.Expr)) for node in preamble):
        raise ValueError("unsupported public-test setup before the first assertion")
    if any(not isinstance(node, ast.Assert) for node in check.body[first_assert:]):
        raise ValueError("unsupported public-test statements between assertions")

    transformer = _RenameCheckArgument(arguments[0].arg, entry_point)
    tests: list[str] = []
    for assertion in assertions:
        nodes = [
            transformer.visit(ast.parse(ast.unparse(node)).body[0]) for node in preamble
        ]
        nodes.append(transformer.visit(ast.parse(ast.unparse(assertion)).body[0]))
        test_tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        tests.append(ast.unparse(test_tree).strip() + "\n")
    return tests


def convert_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    required = (
        "task_id",
        "declaration",
        "canonical_solution",
        "buggy_solution",
        "bug_type",
        "entry_point",
        "test",
        "example_test",
    )
    missing = [key for key in required if not str(row.get(key, "")).strip()]
    if missing:
        raise ValueError(f"HumanEvalFix row is missing fields: {', '.join(missing)}")

    source_task_id = str(row["task_id"])
    task_suffix = source_task_id.replace("/", "_").lower()
    task_id = f"humanevalfix_{task_suffix}"
    bug_type = str(row["bug_type"]).strip().replace(" ", "_")
    prompt = str(row.get("instruction") or row.get("docstring") or row.get("prompt"))
    task = {
        "task_id": task_id,
        "prompt": prompt,
        "entry_point": str(row["entry_point"]),
        "canonical_solution": combine_declaration_and_body(
            str(row["declaration"]), str(row["canonical_solution"])
        ),
        # HumanEvalPack documents example_test as separate from the official test.
        # Atomizing assertions lets the executor report failing inputs and values.
        "public_tests": extract_example_tests(
            str(row["example_test"]), str(row["entry_point"])
        ),
        "hidden_tests": [str(row["test"])],
        "setup_code": str(row.get("test_setup", "")),
        "metadata": {
            "source": "bigcode/humanevalpack",
            "source_task_id": source_task_id,
            "benchmark": "HumanEvalFix",
            "public_test_source": "example_test",
            "public_test_format": "atomized_check_assertions",
            "hidden_test_source": "test",
        },
    }
    bug = {
        "bug_id": f"{task_id}__{bug_type}__0",
        "task_id": task_id,
        "buggy_code": combine_declaration_and_body(
            str(row["declaration"]), str(row["buggy_solution"])
        ),
        "bug_type": bug_type,
        "mutation": "human_written",
        "metadata": {
            "source": "bigcode/humanevalpack",
            "source_task_id": source_task_id,
            "failure_symptoms": str(row.get("failure_symptoms", "")),
        },
    }
    return task, bug


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert the Python HumanEvalFix split into repairbench tasks and bugs."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-file", help="local HumanEvalPack Python JSONL file")
    parser.add_argument("--dataset-name", default="bigcode/humanevalpack")
    parser.add_argument("--config", default="python")
    args = parser.parse_args()

    if args.source_file:
        rows = load_jsonl(args.source_file)
        source = str(Path(args.source_file))
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise SystemExit('install data dependencies with: pip install ".[data]"') from exc
        rows = [dict(row) for row in load_dataset(args.dataset_name, args.config)["test"]]
        source = f"{args.dataset_name}/{args.config}"

    tasks: list[dict[str, Any]] = []
    bugs: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for row in rows:
        if not str(row.get("example_test", "")).strip():
            skipped.append(
                {
                    "task_id": str(row.get("task_id", "unknown")),
                    "reason": "missing_example_test",
                }
            )
            continue
        try:
            task, bug = convert_row(row)
        except (SyntaxError, ValueError) as exc:
            skipped.append(
                {
                    "task_id": str(row.get("task_id", "unknown")),
                    "reason": "invalid_example_test",
                    "detail": str(exc),
                }
            )
            continue
        tasks.append(task)
        bugs.append(bug)

    output_dir = Path(args.output_dir)
    tasks_path = output_dir / "tasks.jsonl"
    bugs_path = output_dir / "bugs.jsonl"
    write_jsonl(tasks_path, tasks)
    write_jsonl(bugs_path, bugs)
    print(
        json.dumps(
            {
                "source": source,
                "tasks": len(tasks),
                "bugs": len(bugs),
                "skipped": len(skipped),
                "skipped_rows": skipped,
                "bug_types": dict(sorted(Counter(bug["bug_type"] for bug in bugs).items())),
                "tasks_output": str(tasks_path),
                "bugs_output": str(bugs_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
