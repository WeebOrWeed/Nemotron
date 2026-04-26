"""Local QLoRA SFT training for the DAG planner.

Reads scored planner candidates from `kaggle_output/planner_scores.jsonl`
(produced by the Kaggle notebook or `train_planner.py`), filters the
high-reward winners, and fine-tunes a LoRA adapter on top of a base
instruct model using TRL's SFTTrainer with 4-bit quantization.

Defaults are tuned for a single 10-16 GB consumer GPU (RTX 3080 / 4080)
running Qwen2.5-3B-Instruct.  Switch to Qwen-7B with --model if you have
24 GB+.

Usage
-----
    python train_lora.py
    python train_lora.py --model Qwen/Qwen2.5-7B-Instruct --epochs 3
    python train_lora.py --scores kaggle_output/planner_scores.jsonl --min-reward 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _load_winners(path: str, min_reward: float, types: list[str] | None = None) -> list[dict]:
    """Read scored JSONL, keep best (valid, high-reward) candidate per puzzle.

    If ``types`` is given, only records whose ``puzzle_type`` is in the list
    are kept.
    """
    if not os.path.exists(path):
        sys.exit(f"ERROR: scores file not found: {path}\n"
                 f"Run train_planner.py or pull from Kaggle first.")

    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    winners = [r for r in records if r.get("dag_valid") and r.get("reward", 0) >= min_reward]
    if types:
        type_set = set(types)
        winners = [r for r in winners if r.get("puzzle_type") in type_set]

    best_per_puzzle: dict[str, dict] = {}
    for r in winners:
        pid = r["puzzle_id"]
        if pid not in best_per_puzzle or r["reward"] > best_per_puzzle[pid]["reward"]:
            best_per_puzzle[pid] = r

    print(f"Total records:          {len(records)}")
    if types:
        print(f"Filtering to types:     {types}")
    print(f"Winners (reward>={min_reward}): {len(winners)}")
    print(f"Unique puzzles:         {len(best_per_puzzle)}")
    print(f"By type:                {dict(Counter(r['puzzle_type'] for r in best_per_puzzle.values()))}")
    return list(best_per_puzzle.values())


def _build_dataset(records: list[dict], planner_system: str):
    from datasets import Dataset

    def build_chat(rec):
        user_msg = f"PUZZLE_TYPE: {rec['puzzle_type']}\n\nPROMPT:\n{rec['prompt']}"
        return {
            "messages": [
                {"role": "system", "content": planner_system},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": rec["planner_output"]},
            ]
        }

    return Dataset.from_list([build_chat(r) for r in records])


def _check_gpu(min_capability: int = 7) -> bool:
    """Return True if the local GPU supports 4-bit QLoRA (sm_70+)."""
    import torch

    if not torch.cuda.is_available():
        print("WARNING: No CUDA GPU detected. Training will be infeasibly slow on CPU.")
        return False
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {name}  |  sm_{cap[0]}{cap[1]}  |  {vram:.1f} GB VRAM")
    if cap[0] < min_capability:
        print(f"WARNING: GPU sm_{cap[0]}{cap[1]} below required sm_{min_capability}0+ for 4-bit QLoRA.")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Local QLoRA SFT for the DAG planner")
    parser.add_argument("--scores", default="kaggle_output/planner_scores.jsonl",
                        help="Scored planner JSONL (default: kaggle_output/planner_scores.jsonl)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct",
                        help="Base HF model (default: Qwen/Qwen2.5-3B-Instruct)")
    parser.add_argument("--output", default="models/planner-lora",
                        help="Output dir for LoRA adapter (default: models/planner-lora)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size (default: 1)")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length (default: 2048)")
    parser.add_argument("--min-reward", type=float, default=0.5,
                        help="Min reward to count as a winner (default: 0.5)")
    parser.add_argument("--types", nargs="*", default=None,
                        help="Restrict training to specific puzzle types (default: all)")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization (needs much more VRAM)")
    args = parser.parse_args()

    print("=" * 60)
    print("Local QLoRA SFT for DAG Planner")
    print("=" * 60)

    use_4bit = not args.no_4bit
    if use_4bit and not _check_gpu():
        sys.exit("Aborting. Either upgrade GPU, use --no-4bit (needs 24GB+), or train on Kaggle/Colab.")

    print(f"\nLoading winners from {args.scores}...")
    winners = _load_winners(args.scores, args.min_reward, args.types)
    if len(winners) < 5:
        sys.exit(f"ERROR: only {len(winners)} winning examples - collect more data first.")

    print(f"\nImporting transformers / peft / trl...")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    from src.planner import PLANNER_SYSTEM

    dataset = _build_dataset(winners, PLANNER_SYSTEM)
    print(f"Training dataset: {len(dataset)} examples")

    print(f"\nLoading base model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_seq_len

    model_kwargs = {"device_map": "auto", "trust_remote_code": True}
    if use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    if torch.cuda.is_available():
        print(f"GPU memory used: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    print(f"\nApplying LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    os.makedirs(args.output, exist_ok=True)
    training_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"\nStarting training: {args.epochs} epochs on {len(dataset)} examples")
    trainer.train()
    print("Training complete!")

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    total_size = sum(
        os.path.getsize(os.path.join(args.output, f))
        for f in os.listdir(args.output)
        if os.path.isfile(os.path.join(args.output, f))
    )
    print(f"\nLoRA adapter saved to: {args.output}")
    print(f"Adapter size: {total_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
