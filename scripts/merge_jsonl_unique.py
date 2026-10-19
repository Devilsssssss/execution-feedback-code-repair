from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: str) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"{path}:{line_number} is not an object")
                rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge JSONL objects with unique bug IDs")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in args.input:
        for row in load(path):
            key = str(row.get("bug_id", ""))
            if not key:
                raise ValueError(f"missing bug_id in {path}")
            if key in seen:
                raise ValueError(f"duplicate bug_id: {key}")
            seen.add(key)
            rows.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
