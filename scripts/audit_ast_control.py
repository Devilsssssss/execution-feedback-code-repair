from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TYPE_ALIASES = {
    "boundary_condition": "boundary_condition",
    "logic_error": "logic_error",
    "model_logic_error": "logic_error",
    "runtime_error": "runtime_error",
    "model_runtime_error": "runtime_error",
    "model_timeout_error": "runtime_error",
}


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


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_bug_type(value: str) -> str:
    try:
        return TYPE_ALIASES[value]
    except KeyError as exc:
        supported = ", ".join(sorted(TYPE_ALIASES))
        raise ValueError(f"unsupported bug type {value!r}; expected one of {supported}") from exc


def count_types(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(normalize_bug_type(str(row.get("bug_type", ""))) for row in rows)
    return dict(sorted(counts.items()))


def _index_by_bug_id(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        bug_id = str(row.get("bug_id", ""))
        if not bug_id:
            raise ValueError(f"{label} contains an empty bug_id")
        if bug_id in index:
            raise ValueError(f"{label} contains duplicate bug_id {bug_id}")
        index[bug_id] = row
    return index


def target_quotas(mixed_rows: list[dict[str, Any]]) -> dict[str, int]:
    """Return the Mixed composition after collapsing model_* labels to AST labels."""

    return count_types(mixed_rows)


def maximal_proportional_quotas(
    target: dict[str, int], available: dict[str, int]
) -> tuple[dict[str, int], float]:
    if not target:
        raise ValueError("target distribution is empty")
    ratios = [available.get(kind, 0) / count for kind, count in target.items() if count]
    scale = min(ratios, default=0.0)
    quotas = {kind: int(count * scale) for kind, count in target.items()}
    return quotas, scale


def select_rows(
    rows: list[dict[str, Any]], quotas: dict[str, int], seed: int
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[normalize_bug_type(str(row.get("bug_type", "")))].append(row)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for kind, count in quotas.items():
        candidates = list(groups.get(kind, []))
        if len(candidates) < count:
            raise ValueError(
                f"quota {kind}={count} exceeds {len(candidates)} available AST rows"
            )
        rng.shuffle(candidates)
        selected.extend(candidates[:count])
    rng.shuffle(selected)
    return selected


def build_report(
    mixed_rows: list[dict[str, Any]],
    ast_rows: list[dict[str, Any]],
    *,
    mixed_path: str | Path,
    ast_path: str | Path,
    ast_bugs_path: str | Path,
    seed: int,
) -> dict[str, Any]:
    _index_by_bug_id(mixed_rows, "Mixed SFT")
    ast_index = _index_by_bug_id(ast_rows, "AST SFT")
    ast_bugs = _index_by_bug_id(load_jsonl(ast_bugs_path), "AST bugs")
    missing_bug_ids = sorted(set(ast_index) - set(ast_bugs))
    if missing_bug_ids:
        raise ValueError(f"AST SFT rows missing bug records: {missing_bug_ids[:3]}")

    target = target_quotas(mixed_rows)
    available = count_types(ast_rows)
    deficits = {
        kind: max(0, count - available.get(kind, 0)) for kind, count in target.items()
    }
    exact_feasible = not any(deficits.values())
    maximal, scale = maximal_proportional_quotas(target, available)
    report = {
        "protocol": "mixed-composition-ast-only-v1",
        "seed": seed,
        "inputs": {
            "mixed_sft": {"path": str(mixed_path), "rows": len(mixed_rows), "sha256": file_sha256(mixed_path)},
            "ast_sft": {"path": str(ast_path), "rows": len(ast_rows), "sha256": file_sha256(ast_path)},
            "ast_bugs": {"path": str(ast_bugs_path), "rows": len(ast_bugs), "sha256": file_sha256(ast_bugs_path)},
        },
        "target_normalized_counts": target,
        "ast_available_normalized_counts": available,
        "deficits": deficits,
        "exact_feasible": exact_feasible,
        "maximal_proportional_scale": scale,
        "maximal_proportional_quotas": maximal,
        "maximal_proportional_rows": sum(maximal.values()),
    }
    return report


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="audit and optionally build an AST-only control matching Mixed bug types"
    )
    parser.add_argument("--mixed-sft", required=True)
    parser.add_argument("--ast-sft", required=True)
    parser.add_argument("--ast-bugs", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--mode", choices=("report", "exact", "maximal"), default="report")
    parser.add_argument("--output-sft")
    parser.add_argument("--output-bugs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mixed_rows = load_jsonl(args.mixed_sft)
    ast_rows = load_jsonl(args.ast_sft)
    report = build_report(
        mixed_rows,
        ast_rows,
        mixed_path=args.mixed_sft,
        ast_path=args.ast_sft,
        ast_bugs_path=args.ast_bugs,
        seed=args.seed,
    )
    quotas = report["target_normalized_counts"]
    if args.mode == "maximal":
        quotas = report["maximal_proportional_quotas"]
    if args.mode in {"exact", "maximal"}:
        if args.mode == "exact" and not report["exact_feasible"]:
            report["selection"] = {"status": "blocked", "reason": "insufficient AST candidates"}
        else:
            if not args.output_sft or not args.output_bugs:
                raise SystemExit("--output-sft and --output-bugs are required for dataset-building modes")
            selected = select_rows(ast_rows, quotas, args.seed)
            bug_index = _index_by_bug_id(load_jsonl(args.ast_bugs), "AST bugs")
            selected_bugs = [bug_index[str(row["bug_id"])] for row in selected]
            _write_jsonl(args.output_sft, selected)
            _write_jsonl(args.output_bugs, selected_bugs)
            report["selection"] = {
                "status": "built",
                "mode": args.mode,
                "rows": len(selected),
                "normalized_counts": count_types(selected),
                "outputs": {
                    "sft": {"path": args.output_sft, "sha256": file_sha256(args.output_sft)},
                    "bugs": {"path": args.output_bugs, "sha256": file_sha256(args.output_bugs)},
                },
            }
    output_summary = Path(args.output_summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.mode == "exact" and not report["exact_feasible"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
