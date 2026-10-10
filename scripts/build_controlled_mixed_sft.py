from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_quotas(values: list[str]) -> dict[str, int]:
    quotas: dict[str, int] = {}
    for value in values:
        name, separator, count = value.partition("=")
        if not separator or not name or not count.isdigit():
            raise ValueError(f"invalid quota {value!r}; expected BUG_TYPE=COUNT")
        if name in quotas:
            raise ValueError(f"duplicate quota for {name}")
        quotas[name] = int(count)
    return quotas


def _index_unique(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        bug_id = str(row.get("bug_id", ""))
        if not bug_id:
            raise ValueError(f"{label} contains an empty bug_id")
        if bug_id in index:
            raise ValueError(f"{label} contains duplicate bug_id {bug_id}")
        index[bug_id] = row
    return index


def build_mixture(
    synthetic_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    quotas: dict[str, int],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _index_unique(synthetic_rows, "synthetic SFT")
    _index_unique(model_rows, "model SFT")
    shared_ids = {str(row["bug_id"]) for row in synthetic_rows} & {
        str(row["bug_id"]) for row in model_rows
    }
    if shared_ids:
        raise ValueError(f"bug IDs overlap across sources: {sorted(shared_ids)[:3]}")

    model_prompts = {str(row.get("prompt", "")).strip() for row in model_rows}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded_prompt_duplicates = 0
    for row in synthetic_rows:
        if str(row.get("prompt", "")).strip() in model_prompts:
            excluded_prompt_duplicates += 1
            continue
        groups[str(row.get("bug_type", ""))].append(row)

    rng = random.Random(seed)
    selected_synthetic: list[dict[str, Any]] = []
    for bug_type, count in quotas.items():
        candidates = list(groups.get(bug_type, []))
        if len(candidates) < count:
            raise ValueError(
                f"quota {bug_type}={count} exceeds {len(candidates)} available rows"
            )
        rng.shuffle(candidates)
        selected_synthetic.extend(candidates[:count])

    selected = selected_synthetic + list(model_rows)
    prompt_counts = Counter(str(row.get("prompt", "")).strip() for row in selected)
    duplicate_prompts = [prompt for prompt, count in prompt_counts.items() if count > 1]
    if duplicate_prompts:
        raise ValueError(f"selected rows contain {len(duplicate_prompts)} duplicate prompts")
    rng.shuffle(selected)
    summary = {
        "seed": seed,
        "rows": len(selected),
        "source_counts": {
            "synthetic_mutations": len(selected_synthetic),
            "model_generated_errors": len(model_rows),
        },
        "synthetic_quotas": quotas,
        "bug_type_counts": dict(sorted(Counter(str(row["bug_type"]) for row in selected).items())),
        "tasks": len({str(row["task_id"]) for row in selected}),
        "excluded_cross_source_prompt_duplicates": excluded_prompt_duplicates,
    }
    return selected, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="build a size-controlled mixed SFT dataset")
    parser.add_argument("--synthetic-sft", required=True)
    parser.add_argument("--model-sft", required=True)
    parser.add_argument("--synthetic-bugs", required=True)
    parser.add_argument("--model-bugs", required=True)
    parser.add_argument("--synthetic-quota", action="append", default=[], metavar="TYPE=COUNT")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-sft", required=True)
    parser.add_argument("--output-bugs", required=True)
    parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()

    synthetic_rows = load_jsonl(args.synthetic_sft)
    model_rows = load_jsonl(args.model_sft)
    selected, summary = build_mixture(
        synthetic_rows,
        model_rows,
        parse_quotas(args.synthetic_quota),
        args.seed,
    )

    all_bugs = _index_unique(
        load_jsonl(args.synthetic_bugs) + load_jsonl(args.model_bugs), "combined bugs"
    )
    selected_ids = {str(row["bug_id"]) for row in selected}
    missing_bugs = sorted(selected_ids - set(all_bugs))
    if missing_bugs:
        raise ValueError(f"missing bug records for {missing_bugs[:3]}")
    selected_bugs = [all_bugs[str(row["bug_id"])] for row in selected]

    write_jsonl(args.output_sft, selected)
    write_jsonl(args.output_bugs, selected_bugs)
    summary["outputs"] = {
        "sft": {"path": args.output_sft, "sha256": file_sha256(args.output_sft)},
        "bugs": {"path": args.output_bugs, "sha256": file_sha256(args.output_bugs)},
    }
    output_summary = Path(args.output_summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
