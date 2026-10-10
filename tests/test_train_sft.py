import json

from repairbench.prompts import SYSTEM_PROMPT
from scripts.train_sft import build_parser, load_training_rows


def test_train_parser_accepts_qlora_memory_flags() -> None:
    args = build_parser().parse_args(
        [
            "--train-file",
            "train.jsonl",
            "--output-dir",
            "out",
            "--load-in-4bit",
            "--gradient-checkpointing",
            "--max-length",
            "768",
            "--compute-dtype",
            "float16",
        ]
    )

    assert args.load_in_4bit is True
    assert args.gradient_checkpointing is True
    assert args.max_length == 768
    assert args.compute_dtype == "float16"


def test_training_rows_use_same_chat_roles_as_inference(tmp_path) -> None:
    source = tmp_path / "train.jsonl"
    source.write_text(
        json.dumps({"prompt": "Fix this function.", "completion": "def f():\n    return 1\n"})
        + "\n",
        encoding="utf-8",
    )

    row = load_training_rows(source)[0]

    assert row["prompt"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Fix this function."},
    ]
    assert row["completion"] == [
        {"role": "assistant", "content": "def f():\n    return 1\n"}
    ]
