from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluation import EvaluationRunner, summarize
from .executor import BubblewrapExecutor, DockerExecutor, LocalSubprocessExecutor
from .feedback import FeedbackCondition
from .io import check_task_disjointness, index_tasks, load_bugs, load_tasks, write_jsonl
from .models import HeuristicRepairModel, NoOpRepairModel, RepairModel, TransformersRepairModel
from .mutations import generate_mutations
from .prompts import build_repair_prompt
from .schemas import BugCase


def _add_executor_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--executor", choices=("docker", "bwrap", "local"), default="docker")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument(
        "--allow-unsafe-local",
        action="store_true",
        help="required for local mode; local mode is not a security sandbox",
    )


def _executor(args: argparse.Namespace):
    if args.executor == "local":
        return LocalSubprocessExecutor(
            timeout_seconds=args.timeout, allow_unsafe=args.allow_unsafe_local
        )
    if args.executor == "bwrap":
        return BubblewrapExecutor(timeout_seconds=args.timeout)
    return DockerExecutor(timeout_seconds=args.timeout)


def command_generate_bugs(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks)
    executor = _executor(args)
    bugs: list[BugCase] = []
    for task in tasks:
        accepted = 0
        for mutation in generate_mutations(task.canonical_solution):
            result = executor.execute(
                mutation.code, task.public_tests, setup_code=task.setup_code
            )
            if result.all_passed:
                continue
            bugs.append(
                BugCase(
                    bug_id=f"{task.task_id}__{mutation.bug_type}__{accepted}",
                    task_id=task.task_id,
                    buggy_code=mutation.code,
                    bug_type=mutation.bug_type,
                    mutation=mutation.mutation_id,
                    metadata={"public_failure": result.to_dict()},
                )
            )
            accepted += 1
            if accepted >= args.per_task:
                break
    write_jsonl(args.output, (bug.to_dict() for bug in bugs))
    print(json.dumps({"tasks": len(tasks), "bugs": len(bugs), "output": args.output}, indent=2))
    return 0


def command_validate(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks)
    bugs = load_bugs(args.bugs)
    task_index = index_tasks(tasks)
    executor = _executor(args)
    failures: list[str] = []

    for task in tasks:
        reference_public = executor.execute(
            task.canonical_solution, task.public_tests, setup_code=task.setup_code
        )
        reference_hidden = executor.execute(
            task.canonical_solution, task.hidden_tests, setup_code=task.setup_code
        )
        if not reference_public.all_passed or not reference_hidden.all_passed:
            failures.append(f"reference solution failed for {task.task_id}")

    for bug in bugs:
        task = task_index.get(bug.task_id)
        if task is None:
            failures.append(f"bug {bug.bug_id} references missing task {bug.task_id}")
            continue
        public_result = executor.execute(bug.buggy_code, task.public_tests, task.setup_code)
        if public_result.all_passed:
            failures.append(f"bug {bug.bug_id} does not fail any public test")

    report = {"tasks": len(tasks), "bugs": len(bugs), "valid": not failures, "failures": failures}
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


def command_check_splits(args: argparse.Namespace) -> int:
    groups = [load_tasks(path) for path in args.task_files]
    check_task_disjointness(*groups)
    print(json.dumps({"valid": True, "groups": [len(group) for group in groups]}, indent=2))
    return 0


def command_build_sft(args: argparse.Namespace) -> int:
    tasks = index_tasks(load_tasks(args.tasks))
    bugs = load_bugs(args.bugs)
    executor = _executor(args)
    records: list[dict[str, Any]] = []
    for bug in bugs:
        task = tasks.get(bug.task_id)
        if task is None:
            raise ValueError(f"bug {bug.bug_id} references unknown task {bug.task_id}")
        public_result = executor.execute(bug.buggy_code, task.public_tests, task.setup_code)
        if public_result.all_passed:
            continue
        prompt = build_repair_prompt(
            task, bug.buggy_code, public_result, FeedbackCondition.STRUCTURED
        )
        records.append(
            {
                "prompt": prompt,
                "completion": task.canonical_solution.strip() + "\n",
                "task_id": task.task_id,
                "bug_id": bug.bug_id,
                "bug_type": bug.bug_type,
            }
        )
    write_jsonl(args.output, records)
    print(json.dumps({"samples": len(records), "output": args.output}, indent=2))
    return 0


def _run_evaluation(
    args: argparse.Namespace,
    model: RepairModel,
    *,
    run_name: str,
) -> int:
    tasks = index_tasks(load_tasks(args.tasks))
    bugs = load_bugs(args.bugs)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        bugs = bugs[: args.limit]
    executor = _executor(args)
    runner = EvaluationRunner(
        model=model,
        executor=executor,
        condition=FeedbackCondition(args.condition),
        max_repairs=args.max_repairs,
    )
    records = []
    for bug in bugs:
        task = tasks.get(bug.task_id)
        if task is None:
            raise ValueError(f"bug {bug.bug_id} references unknown task {bug.task_id}")
        record = runner.evaluate_case(task, bug)
        record["run_name"] = run_name
        records.append(record)
        print(
            json.dumps(
                {
                    "bug_id": bug.bug_id,
                    "repair_at_k": record["repair_at_k"],
                    "repairs_used": record["repairs_used"],
                }
            )
        )
    write_jsonl(args.output, records)
    summary = summarize(records)
    summary["run_name"] = run_name
    summary["condition"] = args.condition
    summary_path = Path(args.output).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    model = TransformersRepairModel(
        model_id=args.model_id,
        adapter_id=args.adapter_id,
        max_new_tokens=args.max_new_tokens,
        device_map=args.device_map,
    )
    run_name = args.adapter_id or args.model_id
    return _run_evaluation(args, model, run_name=run_name)


def command_evaluate_noop(args: argparse.Namespace) -> int:
    return _run_evaluation(args, NoOpRepairModel(), run_name="noop_control")


def command_evaluate_heuristic(args: argparse.Namespace) -> int:
    return _run_evaluation(args, HeuristicRepairModel(), run_name="name_error_heuristic")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="repairbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-bugs", help="create controlled failing programs")
    generate.add_argument("--tasks", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--per-task", type=int, default=3)
    _add_executor_arguments(generate)
    generate.set_defaults(func=command_generate_bugs)

    validate = subparsers.add_parser("validate", help="validate tasks, references, and bugs")
    validate.add_argument("--tasks", required=True)
    validate.add_argument("--bugs", required=True)
    _add_executor_arguments(validate)
    validate.set_defaults(func=command_validate)

    splits = subparsers.add_parser("check-splits", help="reject task overlap across splits")
    splits.add_argument("task_files", nargs="+", help="two or more task JSONL files")
    splits.set_defaults(func=command_check_splits)

    sft = subparsers.add_parser("build-sft", help="build prompt-completion SFT records")
    sft.add_argument("--tasks", required=True)
    sft.add_argument("--bugs", required=True)
    sft.add_argument("--output", required=True)
    _add_executor_arguments(sft)
    sft.set_defaults(func=command_build_sft)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a local Hugging Face model")
    evaluate.add_argument("--tasks", required=True)
    evaluate.add_argument("--bugs", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--model-id", required=True)
    evaluate.add_argument("--adapter-id")
    evaluate.add_argument("--condition", choices=[item.value for item in FeedbackCondition], required=True)
    evaluate.add_argument("--max-repairs", type=int, default=3)
    evaluate.add_argument("--max-new-tokens", type=int, default=512)
    evaluate.add_argument(
        "--limit",
        type=int,
        help="evaluate only the first N bug cases (useful for a CPU smoke test)",
    )
    evaluate.add_argument("--device-map", default="auto")
    _add_executor_arguments(evaluate)
    evaluate.set_defaults(func=command_evaluate)

    noop = subparsers.add_parser(
        "evaluate-noop",
        help="run a no-change control to validate the evaluation pipeline",
    )
    noop.add_argument("--tasks", required=True)
    noop.add_argument("--bugs", required=True)
    noop.add_argument("--output", required=True)
    noop.add_argument("--condition", choices=[item.value for item in FeedbackCondition], required=True)
    noop.add_argument("--max-repairs", type=int, default=1)
    noop.add_argument("--limit", type=int)
    _add_executor_arguments(noop)
    noop.set_defaults(func=command_evaluate_noop)

    heuristic = subparsers.add_parser(
        "evaluate-heuristic",
        help="run an interpretable NameError-only repair baseline",
    )
    heuristic.add_argument("--tasks", required=True)
    heuristic.add_argument("--bugs", required=True)
    heuristic.add_argument("--output", required=True)
    heuristic.add_argument("--condition", choices=[item.value for item in FeedbackCondition], required=True)
    heuristic.add_argument("--max-repairs", type=int, default=1)
    heuristic.add_argument("--limit", type=int)
    _add_executor_arguments(heuristic)
    heuristic.set_defaults(func=command_evaluate_heuristic)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
