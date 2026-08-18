"""LoRA fine-tuning of the plain-language SLM (moved from scripts/finetune_slm.py).

PEFT LoRA on an instruct base model (default Qwen2.5-1.5B-Instruct). Trains
on aligned finding->explanation pairs with engine-authoritative fields.
Writes per-epoch checkpoints and a final adapter.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from legalrag.train.artifacts import artifact_stamp, git_stamp
from legalrag.train.data import load_finetune_pairs

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_DATA = "data/finetune/train.jsonl"
DEFAULT_EVAL = "data/finetune/eval.jsonl"
DEFAULT_OUT = "models/finetuned/lora"


def format_chat(example: dict, tokenizer) -> str:
    """Format a chat example via the tokenizer's chat template (model-agnostic)."""
    return tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )


def finetune(
    model_name: str = DEFAULT_MODEL,
    train_data_path: str | Path = DEFAULT_DATA,
    eval_data_path: str | Path = DEFAULT_EVAL,
    out_dir: Path | None = None,
    epochs: int = 3,
    lr: float = 2e-4,
    r: int = 8,
    alpha: int = 16,
    max_length: int = 1024,
    load_8bit: bool = False,
    batch_size: int = 1,
    grad_accum: int = 8,
    eval_only: bool = False,
    verbose: bool = True,
) -> int:
    """Run the LoRA training loop; returns 0 on success."""
    out_dir = out_dir or Path(DEFAULT_OUT)
    if out_dir.exists() and any(out_dir.iterdir()) and not eval_only:
        rev = git_stamp().get("git_commit") or "nogit"
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest = out_dir.parent / "backups" / "slm-lora" / f"{ts}-{rev}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        out_dir.replace(dest)
        if verbose:
            print(f"[backup] previous LoRA artifacts -> {dest}")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data = load_finetune_pairs(train_data_path)
    eval_data = load_finetune_pairs(eval_data_path) if Path(eval_data_path).exists() else []
    if verbose:
        print(f"[data] train: {len(train_data)} pairs, eval: {len(eval_data)} pairs")
    if not train_data:
        print("[error] no training data")
        return 1

    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"[error] missing dependency: {e}")
        print("  pip install torch transformers peft accelerate")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"[model] loading {model_name} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_8bit and device == "cuda":
        from peft import prepare_model_for_kbit_training

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cpu":
            model = model.to(device)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if device == "cuda":
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    if eval_only:
        print("[eval-only] skipping training")
        return 0

    if verbose:
        print("[train] formatting training examples...")
    train_texts = [format_chat(ex, tokenizer) for ex in train_data]

    if verbose:
        print(f"[train] starting training: {epochs} epochs, lr={lr}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_texts), batch_size):
            batch = train_texts[i : i + batch_size]
            encodings = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            outputs = model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss / grad_accum
            loss.backward()
            epoch_loss += loss.item() * grad_accum

            if (i // batch_size + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                n_batches += 1
                step = i + len(batch)
                if verbose:
                    print(f"  epoch {epoch+1} step {step}/{len(train_texts)} loss {loss.item() * grad_accum:.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        if verbose:
            print(f"[epoch {epoch+1}] loss={avg_loss:.4f} elapsed={elapsed:.0f}s")

        ckpt_dir = out_dir / f"epoch{epoch+1}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))
        if verbose:
            print(f"  saved checkpoint: {ckpt_dir}")

    model.save_pretrained(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))

    meta = {
        "model": model_name,
        "epochs": epochs,
        "lr": lr,
        "r": r,
        "alpha": alpha,
        "max_length": max_length,
        "load_8bit": load_8bit,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "n_train": len(train_data),
        "n_eval": len(eval_data),
    }
    meta.update(artifact_stamp(train_data))
    (out_dir / "finetune_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    if verbose:
        elapsed = time.time() - t0
        print(f"\n[done] training complete in {elapsed:.0f}s")
        print(f"  checkpoints: {out_dir}/epoch*/")
        print(f"  final adapter: {out_dir}/final/")

    return 0