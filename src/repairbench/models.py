from __future__ import annotations

import ast
import difflib
import re
from dataclasses import dataclass
from typing import Protocol

from .prompts import SYSTEM_PROMPT


class RepairModel(Protocol):
    def repair(self, prompt: str) -> str:
        ...


@dataclass(frozen=True)
class NoOpRepairModel:
    """Control model that returns the current candidate unchanged.

    This is a pipeline control, not a useful repair strategy. It establishes
    the initial failure rate before comparing actual code models.
    """

    marker: str = "Current incorrect code:\n"

    def repair(self, prompt: str) -> str:
        if self.marker not in prompt:
            raise ValueError("repair prompt is missing the current-code section")
        code = prompt.split(self.marker, 1)[1]
        code = code.split("\n\nPublic-test execution feedback:", 1)[0]
        code = code.split("\n\nReturn a corrected implementation", 1)[0]
        return code.strip() + "\n"


@dataclass(frozen=True)
class HeuristicRepairModel:
    """Small, interpretable control that repairs only obvious NameErrors."""

    def repair(self, prompt: str) -> str:
        code = NoOpRepairModel().repair(prompt)
        match = re.search(r"name ['\"]([^'\"]+)['\"] is not defined", prompt)
        if match is None:
            return code

        missing = match.group(1)
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code
        candidates = _defined_names(tree)
        if not candidates:
            return code

        base = re.sub(r"_(?:missing|undefined|typo)$", "", missing)
        prefixed = [name for name in candidates if name.startswith(base)] if base else []
        if prefixed:
            replacement = min(prefixed, key=lambda name: (len(name), name))
        else:
            replacement = difflib.get_close_matches(missing, candidates, n=1, cutoff=0.72)
            if not replacement:
                return code
            replacement = replacement[0]
        if replacement == missing:
            return code
        return re.sub(rf"\b{re.escape(missing)}\b", replacement, code)


def _defined_names(tree: ast.AST) -> list[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.update(argument.arg for argument in node.args.args)
                names.update(argument.arg for argument in node.args.kwonlyargs)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name.split(".")[0])
    return sorted(names)


def extract_python_code(text: str) -> str:
    # Generation can be cut off before the closing fence.  Treat the rest of
    # an opened Python fence as code instead of executing the explanatory
    # prose that preceded it.
    fenced = re.search(
        r"```(?:python|py)?\s*(.*?)(?:```|$)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    code = fenced.group(1) if fenced else text
    if fenced is None:
        # Some instruction-tuned models omit fences altogether but still put
        # a short explanation before the function.  Start at the first
        # top-level Python declaration when one is present.
        declaration = re.search(
            r"(?m)^(?=(?:async\s+def|def|class|from|import)\b)", code
        )
        if declaration:
            code = code[declaration.start() :]
    # A decoder may continue by copying the tail of the prompt after it has
    # finished the implementation.  Never execute that feedback as code.
    for marker in (
        "\n# Public test execution feedback:",
        "\nPublic test execution feedback:",
        "\nReturn a corrected implementation",
    ):
        if marker in code:
            code = code.split(marker, 1)[0]
    code = code.strip()
    if code.lower().startswith("python\n"):
        code = code.split("\n", 1)[1]
    return code.strip() + "\n"


@dataclass
class TransformersRepairModel:
    model_id: str
    adapter_id: str | None = None
    max_new_tokens: int = 512
    device_map: str = "auto"
    dtype: str = "auto"

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError('install model dependencies with: pip install ".[model]"') from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model_kwargs = {"device_map": self.device_map}
        if self.dtype != "auto":
            model_kwargs["torch_dtype"] = getattr(torch, self.dtype)
        if self.adapter_id:
            try:
                from peft import AutoPeftModelForCausalLM
            except ImportError as exc:
                raise RuntimeError("adapter loading requires the train optional dependencies") from exc
            self._model = AutoPeftModelForCausalLM.from_pretrained(
                self.adapter_id, **model_kwargs
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        self._model.eval()

    def repair(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        rendered = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(rendered, return_tensors="pt").to(self._model.device)
        with self._torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return extract_python_code(self._tokenizer.decode(generated, skip_special_tokens=True))
