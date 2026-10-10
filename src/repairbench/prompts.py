from __future__ import annotations

from .feedback import FeedbackCondition, format_feedback
from .schemas import ExecutionResult, RepairTask

SYSTEM_PROMPT = """You repair a single Python function. Return only the complete corrected Python code.
Do not use Markdown fences. Do not change the function name or signature. Do not add tests or explanations."""


def build_repair_prompt(
    task: RepairTask,
    current_code: str,
    execution: ExecutionResult,
    condition: FeedbackCondition,
) -> str:
    sections = [
        f"Problem:\n{task.prompt.strip()}",
        f"Required entry point: {task.entry_point}",
        f"Current incorrect code:\n{current_code.strip()}",
    ]
    feedback = format_feedback(execution, condition)
    if feedback:
        sections.append(f"Public-test execution feedback:\n{feedback}")
    sections.append("Return a corrected implementation that generalizes beyond the visible tests.")
    return "\n\n".join(sections)
