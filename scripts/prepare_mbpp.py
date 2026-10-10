from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def normalize_tests(row: dict[str, Any]) -> tuple[list[str], list[str], str]:
    standard = [str(test) for test in row.get("test_list", []) if str(test).strip()]
    challenge = [
        str(test) for test in row.get("challenge_test_list", []) if str(test).strip()
    ]
    if challenge:
        return standard, challenge, "challenge_test_list"
    if len(standard) < 2:
        raise ValueError(f"task {row.get('task_id')} has fewer than two usable tests")
    hidden_count = max(1, len(standard) // 3)
    return standard[:-hidden_count], standard[-hidden_count:], "held_out_test_list"


def load_local_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {source}:{line_number}: {exc}") from exc
                if not isinstance(value, dict):
                    raise TypeError(f"expected an object at {source}:{line_number}")
                rows.append(value)
        return rows

    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected a JSON list of objects in {source}")
    return list(value)


def deterministic_splits(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: int(row["task_id"]))
    total = len(ordered)
    train_end = max(1, round(total * 0.8))
    validation_end = min(total - 1, train_end + max(1, round(total * 0.1)))
    return {
        "train": ordered[:train_end],
        "validation": ordered[train_end:validation_end],
        "test": ordered[validation_end:],
    }


def normalize_setup(row: dict[str, Any]) -> str:
    setup = row.get("test_setup_code", "")
    if setup:
        return str(setup)
    imports = row.get("test_imports", [])
    if isinstance(imports, list):
        return "\n".join(str(item) for item in imports if str(item).strip())
    return str(imports) if imports else ""


def convert_rows(
    rows: list[dict[str, Any]],
    *,
    split_name: str,
    source_name: str,
    split_source: str,
) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        public_tests, hidden_tests, hidden_source = normalize_tests(row)
        prompt = row.get("text", row.get("prompt"))
        code = row.get("code")
        if not prompt or not code:
            raise ValueError(f"task {row.get('task_id')} is missing prompt or code")
        records.append(
            {
                "task_id": f"mbpp_{row['task_id']}",
                "prompt": str(prompt),
                "entry_point": _entry_point(str(code)),
                "canonical_solution": str(code),
                "public_tests": public_tests,
                "hidden_tests": hidden_tests,
                "setup_code": normalize_setup(row),
                "metadata": {
                    "source": source_name,
                    "source_split": split_name,
                    "split_source": split_source,
                    "hidden_test_source": hidden_source,
                },
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="google-research-datasets/mbpp")
    parser.add_argument("--config", default="sanitized")
    parser.add_argument(
        "--source-file",
        help="local MBPP JSON/JSONL mirror; bypasses the Hugging Face Hub",
    )
    args = parser.parse_args()

    from repairbench.io import write_jsonl

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.source_file:
        rows = load_local_rows(args.source_file)
        dataset = deterministic_splits(rows)
        source_name = str(Path(args.source_file))
        split_source = "deterministic_task_id_80_10_10"
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise SystemExit('install data dependencies with: pip install ".[data]"') from exc
        dataset = load_dataset(args.dataset_name, args.config)
        source_name = args.dataset_name
        split_source = "official_hub_split"

    for split_name, split in dataset.items():
        records = convert_rows(
            [dict(row) for row in split],
            split_name=split_name,
            source_name=source_name,
            split_source=split_source,
        )
        write_jsonl(output_dir / f"{split_name}_tasks.jsonl", records)
        print(f"wrote {len(records)} tasks to {output_dir / f'{split_name}_tasks.jsonl'}")
    return 0


def _entry_point(code: str) -> str:
    import ast

    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    raise ValueError("reference code has no top-level function")


if __name__ == "__main__":
    raise SystemExit(main())
