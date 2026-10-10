from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from repairbench.prompts import SYSTEM_PROMPT

# This project trains with PyTorch only.  Explicitly disable optional
# TensorFlow imports so an unrelated system Keras installation cannot break
# Transformers/TRL initialization.
os.environ.setdefault("USE_TF", "0")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="train a LoRA/QLoRA repair adapter")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="load the base model with NF4 4-bit quantization (QLoRA)",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="reduce activation memory at the cost of extra compute",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=("auto", "float16", "bfloat16"),
        default="auto",
        help="compute dtype for QLoRA; auto selects bf16 when supported",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _compute_dtype(torch, requested: str):
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_training_rows(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(
                {
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": str(item["prompt"])},
                    ],
                    "completion": [
                        {"role": "assistant", "content": str(item["completion"])},
                    ],
                }
            )
    return rows


def main() -> int:
    args = build_parser().parse_args()

    try:
        from datasets import Dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit('install training dependencies with: pip install ".[train]"') from exc

    import torch

    if args.max_length <= 0 or args.batch_size <= 0 or args.gradient_accumulation <= 0:
        raise SystemExit("max length, batch size, and gradient accumulation must be positive")
    if args.load_in_4bit and not torch.cuda.is_available():
        raise SystemExit("QLoRA requires a CUDA GPU; use a CUDA PyTorch build and NVIDIA GPU")

    rows = load_training_rows(args.train_file)
    if not rows:
        raise SystemExit("training file contains no samples")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    dataset = Dataset.from_list(rows)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = args.model_id
    if args.load_in_4bit:
        dtype = _compute_dtype(torch, args.compute_dtype)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map=args.device_map,
            torch_dtype=dtype,
            quantization_config=quantization_config,
        )
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False

    gradient_checkpointing_kwargs = None
    if args.gradient_checkpointing:
        gradient_checkpointing_kwargs = {"use_reentrant": False}
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_length=args.max_length,
        completion_only_loss=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        seed=args.seed,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
