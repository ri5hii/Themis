"""LoRA fine-tuning of the plain-language SLM.

PEFT LoRA on an instruct base model (default Llama-3.2-1B-Instruct; Qwen2.5
and other chat-templated models work via --model). Trains on aligned
finding→explanation pairs with engine-authoritative fields.

Usage:
    python scripts/finetune_slm.py [--model Qwen/Qwen2.5-1.5B-Instruct] \
        [--epochs 3] [--lr 2e-4] [--r 8] [--alpha 16]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_DATA = "data/finetune/train.jsonl"
DEFAULT_EVAL = "data/finetune/eval.jsonl"
DEFAULT_OUT = "models/finetuned/lora"


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_chat(example: dict, tokenizer) -> str:
    """Format a chat example via the tokenizer's chat template (model-agnostic)."""
    return tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model")
    parser.add_argument("--train-data", default=DEFAULT_DATA)
    parser.add_argument("--eval-data", default=DEFAULT_EVAL)
    parser.add_argument("--output", "-o", default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--load-8bit", action="store_true", help="QLoRA: 8-bit base (needs bitsandbytes, CUDA)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--start-epoch", type=int, default=0, help="Resume from epoch")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, eval only")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"[data] loading {args.train_data}...")
    train_data = load_jsonl(args.train_data)
    eval_data = load_jsonl(args.eval_data) if Path(args.eval_data).exists() else []
    print(f"  train: {len(train_data)} pairs, eval: {len(eval_data)} pairs")

    if not train_data:
        print("[error] no training data")
        return 1

    # Check dependencies
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"[error] missing dependency: {e}")
        print("  pip install torch transformers peft accelerate")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[model] loading {args.model} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.load_8bit and device == "cuda":
        from peft import prepare_model_for_kbit_training

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cpu":
            model = model.to(device)

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if device == "cuda":
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    if args.eval_only:
        print("[eval-only] skipping training")
        return 0

    # Format training data
    print("[train] formatting training examples...")
    train_texts = [format_chat(ex, tokenizer) for ex in train_data]

    # Simple training loop (no HF Trainer dependency)
    print(f"[train] starting training: {args.epochs} epochs, lr={args.lr}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    t0 = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_texts), args.batch_size):
            batch = train_texts[i : i + args.batch_size]
            encodings = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            ).to(device)

            outputs = model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss / args.grad_accum
            loss.backward()
            epoch_loss += loss.item() * args.grad_accum

            if (i // args.batch_size + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                n_batches += 1
                step = i + len(batch)
                print(f"  epoch {epoch+1} step {step}/{len(train_texts)} loss {loss.item() * args.grad_accum:.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        print(f"[epoch {epoch+1}] loss={avg_loss:.4f} elapsed={elapsed:.0f}s")

        # Save checkpoint
        ckpt_dir = out_dir / f"epoch{epoch+1}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))
        print(f"  saved checkpoint: {ckpt_dir}")

    # Save final
    model.save_pretrained(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))

    elapsed = time.time() - t0
    print(f"\n[done] training complete in {elapsed:.0f}s")
    print(f"  checkpoints: {out_dir}/epoch*/")
    print(f"  final adapter: {out_dir}/final/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
