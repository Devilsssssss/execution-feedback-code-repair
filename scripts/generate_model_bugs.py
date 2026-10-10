from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os

from repairbench.executor import BubblewrapExecutor, DockerExecutor, LocalSubprocessExecutor
from repairbench.io import load_tasks, write_jsonl
from repairbench.models import extract_python_code
from repairbench.schemas import BugCase, ExecutionResult

# Keep Transformers from importing an unrelated system TensorFlow/Keras stack.
os.environ.setdefault("USE_TF", "0")

SYNTHESIS_SYSTEM_PROMPT = """You solve Python programming problems. Return only the complete Python code.
Do not use Markdown fences. Define the requested entry point. Do not add tests or explanations."""


def build_synthesis_prompt(problem: str, entry_point: str) -> str:
    return (
        f"Problem:\n{problem.strip()}\n\n"
        f"Required entry point: {entry_point}\n\n"
        "Write a complete implementation."
    )


def has_entry_point(code: str, entry_point: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point
        for node in tree.body
    )


def semantic_digest(code: str) -> str:
    tree = ast.parse(code)
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def classify_failure(result: ExecutionResult) -> str:
    if result.status == "timeout":
        return "timeout_error"
    error_types = {failure.error_type for failure in result.failures}
    if error_types and error_types <= {"AssertionError"}:
        return "logic_error"
    return "runtime_error"


def build_executor(args: argparse.Namespace):
    if args.executor == "local":
        return LocalSubprocessExecutor(
            timeout_seconds=args.timeout,
            allow_unsafe=args.allow_unsafe_local,
        )
    if args.executor == "bwrap":
        return BubblewrapExecutor(timeout_seconds=args.timeout)
    return DockerExecutor(timeout_seconds=args.timeout)


def require_runner(result: ExecutionResult, stage: str) -> None:
    if result.status != "runner_error":
        return
    detail = result.failures[0].message if result.failures else "unknown runner error"
    raise RuntimeError(f"generation infrastructure failed during {stage}: {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample model-written solutions and retain diverse, test-confirmed bugs."
    )
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-task", type=int, default=4)
    parser.add_argument("--max-per-task", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--limit", type=int, help="sample only the first N tasks for a smoke test")
    parser.add_argument("--executor", choices=("docker", "bwrap", "local"), default="docker")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--allow-unsafe-local", action="store_true")
    args = parser.parse_args()
    if args.samples_per_task <= 0 or args.max_per_task <= 0 or args.max_new_tokens <= 0:
        raise ValueError("sampling counts and max-new-tokens must be positive")
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        raise ValueError("temperature must be positive and top-p must be in (0, 1]")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit('install model dependencies with: pip install ".[model]"') from exc

    tasks = load_tasks(args.tasks)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        tasks = tasks[: args.limit]
    executor = build_executor(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map=args.device_map)
    model.eval()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    bugs: list[BugCase] = []
    counters = {
        "sampled": 0,
        "unparseable_or_missing_entry_point": 0,
        "duplicate": 0,
        "public_passed": 0,
        "hidden_passed": 0,
        "accepted": 0,
    }
    for task_index, task in enumerate(tasks, start=1):
        messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_synthesis_prompt(task.prompt, task.entry_point),
            },
        ]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.samples_per_task,
                pad_token_id=tokenizer.eos_token_id,
            )

        seen: set[str] = set()
        accepted_for_task = 0
        for sample_index, output in enumerate(outputs):
            counters["sampled"] += 1
            generated = output[inputs["input_ids"].shape[1] :]
            code = extract_python_code(tokenizer.decode(generated, skip_special_tokens=True))
            if not has_entry_point(code, task.entry_point):
                counters["unparseable_or_missing_entry_point"] += 1
                continue
            digest = semantic_digest(code)
            if digest in seen:
                counters["duplicate"] += 1
                continue
            seen.add(digest)

            public_result = executor.execute(code, task.public_tests, task.setup_code)
            require_runner(public_result, f"{task.task_id}:sample_{sample_index}:public")
            if public_result.all_passed:
                counters["public_passed"] += 1
                continue
            hidden_result = executor.execute(code, task.hidden_tests, task.setup_code)
            require_runner(hidden_result, f"{task.task_id}:sample_{sample_index}:hidden")
            if hidden_result.all_passed:
                counters["hidden_passed"] += 1
                continue

            bug_type = classify_failure(public_result)
            bugs.append(
                BugCase(
                    bug_id=f"{task.task_id}__model_{bug_type}__{accepted_for_task}",
                    task_id=task.task_id,
                    buggy_code=code,
                    bug_type=f"model_{bug_type}",
                    mutation=f"model_sample:{sample_index}",
                    metadata={
                        "generator_model": args.model_id,
                        "seed": args.seed,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "public_failure": public_result.to_dict(),
                        "hidden_failure": hidden_result.to_dict(),
                    },
                )
            )
            accepted_for_task += 1
            counters["accepted"] += 1
            if accepted_for_task >= args.max_per_task:
                break
        print(
            json.dumps(
                {
                    "task": task.task_id,
                    "progress": f"{task_index}/{len(tasks)}",
                    "accepted": accepted_for_task,
                }
            ),
            flush=True,
        )

    write_jsonl(args.output, (bug.to_dict() for bug in bugs))
    print(json.dumps({**counters, "tasks": len(tasks), "output": args.output}, indent=2))
    return 0 if bugs else 1


if __name__ == "__main__":
    raise SystemExit(main())
