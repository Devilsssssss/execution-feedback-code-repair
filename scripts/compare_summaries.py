from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_summary(spec: str) -> tuple[str, dict[str, Any]]:
    if "=" not in spec:
        raise ValueError(f"expected LABEL=SUMMARY_JSON, got {spec!r}")
    label, path = spec.split("=", 1)
    if not label.strip():
        raise ValueError("summary label must not be empty")
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"summary must be a JSON object: {path}")
    return label, value


def main() -> int:
    parser = argparse.ArgumentParser(description="compare repair evaluation summaries")
    parser.add_argument("summaries", nargs="+", help="LABEL=path/to/summary.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    for spec in args.summaries:
        label, summary = load_summary(spec)
        rows.append(
            {
                "label": label,
                "run_name": summary.get("run_name", label),
                "condition": summary.get("condition"),
                "cases": summary.get("cases", 0),
                "repair_at_1": summary.get("repair_at_1", 0.0),
                "repair_at_k": summary.get("repair_at_k", 0.0),
                "by_bug_type": summary.get("by_bug_type", {}),
            }
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"runs": len(rows), "output": args.output}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
