from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def repairbench_main(arguments: list[str]) -> int:
    source_dir = ROOT / "src"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    from repairbench.cli import main

    return main(arguments)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the reviewed, CPU-only repairbench demo end to end."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "demo_run",
        help="directory for generated demo files",
    )
    return parser


def run_step(name: str, arguments: list[str]) -> None:
    print(f"\n[{name}]")
    status = repairbench_main(arguments)
    if status != 0:
        raise SystemExit(f"demo step failed: {name} (exit code {status})")


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = ROOT / "data" / "demo" / "tasks.jsonl"
    bugs = output_dir / "bugs.jsonl"
    sft = output_dir / "sft.jsonl"
    heuristic = output_dir / "heuristic.jsonl"

    executor_args = ["--executor", "local", "--allow-unsafe-local"]
    run_step(
        "1/4 generate controlled bugs",
        [
            "generate-bugs",
            "--tasks",
            str(tasks),
            "--output",
            str(bugs),
            "--per-task",
            "2",
            *executor_args,
        ],
    )
    run_step(
        "2/4 validate references and bugs",
        ["validate", "--tasks", str(tasks), "--bugs", str(bugs), *executor_args],
    )
    run_step(
        "3/4 build SFT records",
        [
            "build-sft",
            "--tasks",
            str(tasks),
            "--bugs",
            str(bugs),
            "--output",
            str(sft),
            *executor_args,
        ],
    )
    run_step(
        "4/4 evaluate interpretable baseline",
        [
            "evaluate-heuristic",
            "--tasks",
            str(tasks),
            "--bugs",
            str(bugs),
            "--output",
            str(heuristic),
            "--condition",
            "structured",
            "--max-repairs",
            "1",
            *executor_args,
        ],
    )

    summary_path = heuristic.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report = {
        "status": "pass",
        "tasks": sum(1 for line in tasks.read_text(encoding="utf-8").splitlines() if line),
        "bugs": sum(1 for line in bugs.read_text(encoding="utf-8").splitlines() if line),
        "sft_samples": sum(1 for line in sft.read_text(encoding="utf-8").splitlines() if line),
        "baseline_repair_at_1": summary["repair_at_1"],
        "output_dir": str(output_dir),
    }
    report_path = output_dir / "demo_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\n[complete]")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
