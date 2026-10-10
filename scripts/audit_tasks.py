from __future__ import annotations

import argparse
import json
from typing import Any

from repairbench.executor import DockerExecutor, LocalSubprocessExecutor
from repairbench.io import load_tasks, write_jsonl


def build_executor(args: argparse.Namespace):
    if args.executor == "local":
        return LocalSubprocessExecutor(
            timeout_seconds=args.timeout,
            allow_unsafe=args.allow_unsafe_local,
        )
    return DockerExecutor(timeout_seconds=args.timeout)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Keep only tasks whose reference solution passes public and hidden tests."
    )
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rejected-output", required=True)
    parser.add_argument("--executor", choices=("docker", "local"), default="docker")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--allow-unsafe-local", action="store_true")
    args = parser.parse_args()

    executor = build_executor(args)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    tasks = load_tasks(args.tasks)
    for task in tasks:
        public = executor.execute(task.canonical_solution, task.public_tests, task.setup_code)
        hidden = executor.execute(task.canonical_solution, task.hidden_tests, task.setup_code)
        if public.all_passed and hidden.all_passed:
            accepted.append(task.to_dict())
            continue
        rejected.append(
            {
                "task": task.to_dict(),
                "public_result": public.to_dict(),
                "hidden_result": hidden.to_dict(),
            }
        )

    write_jsonl(args.output, accepted)
    write_jsonl(args.rejected_output, rejected)
    print(
        json.dumps(
            {
                "input": args.tasks,
                "accepted": len(accepted),
                "rejected": len(rejected),
                "output": args.output,
                "rejected_output": args.rejected_output,
            },
            indent=2,
        )
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
