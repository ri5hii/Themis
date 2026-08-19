"""Merge the trained LoRA adapters into a standalone Hugging Face model.

Loads the base model (fp16 on CPU to stay within the small GPU budget),
applies the adapters from models/finetuned/lora/final/, and saves the
merged model to models/qwen2.5-1.5b-merged/ ready for GGUF conversion.

Usage:
    python scripts/merge_lora.py [--base models/qwen2.5-1.5b]
                                 [--adapter models/finetuned/lora/final]
                                 [--out models/qwen2.5-1.5b-merged]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legalrag.train.artifacts import data_sha256, git_stamp, package_version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="models/qwen2.5-1.5b", help="Base model dir")
    parser.add_argument("--adapter", default="models/finetuned/lora/final", help="LoRA adapter dir")
    parser.add_argument("--out", default="models/qwen2.5-1.5b-merged", help="Merged model output dir")
    args = parser.parse_args()

    base_dir = Path(args.base)
    adapter_dir = Path(args.adapter)
    out_dir = Path(args.out)

    if not (base_dir / "config.json").exists():
        print(f"[error] base model not found: {base_dir}")
        return 1
    if not (adapter_dir / "adapter_config.json").exists():
        print(f"[error] adapter not found: {adapter_dir}")
        return 1

    if out_dir.exists() and any(out_dir.iterdir()):
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rev = git_stamp().get("git_commit") or "nogit"
        dest = out_dir.parent / "backups" / "slm-merged" / f"{ts}-{rev}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        out_dir.replace(dest)
        print(f"[backup] previous merged model -> {dest}")

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"[error] missing dependency: {e}")
        return 1

    print(f"[load] base model {base_dir} (fp16, CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.float16, device_map="cpu"
    )
    print(f"[merge] applying adapters from {adapter_dir}...")
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    merged = model.merge_and_unload()
    del model

    print(f"[save] merged model -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(base_dir)
    tokenizer.save_pretrained(str(out_dir))

    meta = {
        "merged_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "base_model": str(base_dir),
        "adapter": str(adapter_dir),
        "train_data_sha256": data_sha256([
            json.loads(line) for line in Path("data/finetune/train.jsonl").read_text(encoding="utf-8").splitlines()
        ]),
        **git_stamp(),
        "package_version": package_version(),
    }
    (out_dir / "merge_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[meta] stamped {out_dir}/merge_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())