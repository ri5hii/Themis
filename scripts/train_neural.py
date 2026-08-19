"""DistilBERT/LegalBERT comparison: fine-tune + evaluate on a supervised task.

Runs the neural comparison for the v0.2.x eval. CPU-only (n_gpu unavailable:
RTX 3050 is 4GiB and unusable). Bounded epochs + small batches so it completes
on CPU in reasonable time; results compared against the deterministic baseline.

Usage:
    PYTHONUNBUFFERED=1 python scripts/train_neural.py --task redflag_paragraph \
        --model nlpaueb/legal-bert-base-uncased --epochs 2

Note: only the multiclass 'redflag_paragraph' task is supported initially;
the 'deontic_multilabel' multi-label variant needs label smoothing / BCE head.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legalrag import tasks


def _load_split(dir_: Path, name: str, split: str) -> list[dict]:
    return tasks.loadJsonl(str(dir_ / f"{name}.{split}.jsonl"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("redflag_paragraph", "deontic_multilabel"), default="redflag_paragraph")
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--max-train", type=int, default=0, help="cap training rows (0 = no cap)")
    parser.add_argument("--splits", default=str(Path("data/splits")))
    parser.add_argument("--out", default=str(Path("eval/artifacts")))
    args = parser.parse_args()

    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    split_dir = Path(args.splits)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_rows = _load_split(split_dir, args.task, "train")
    test_rows = _load_split(split_dir, args.task, "test")

    if args.max_train and len(train_rows) > args.max_train:
        rng = __import__("random").Random(0)
        train_rows = rng.sample(train_rows, args.max_train)

    multilabel = args.task == "deontic_multilabel"
    if not multilabel:
        # Downsample 'none' to bound CPU time (3x positives), mirroring the
        # deterministic baseline's rebalancing.
        pos = [r for r in train_rows if r["type"] != "none"]
        neg = [r for r in train_rows if r["type"] == "none"]
        rng = __import__("random").Random(0)
        train_rows = pos + rng.sample(neg, min(len(neg), len(pos) * 3))

    if multilabel:
        labels = ["obl", "ent", "pro", "per", "oth", "nen", "none"]
        label2id = {lab: i for i, lab in enumerate(labels)}
    else:
        labels = sorted({r["type"] for r in train_rows} | {r["type"] for r in test_rows})
        label2id = {lab: i for i, lab in enumerate(labels)}

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(labels),
        id2label={i: l for l, i in label2id.items()},
        problem_type="multi_label_classification" if multilabel else "single_label_classification",
    )

    def encode(rows: list[dict]):
        return tok(
            [r["text"] for r in rows],
            padding=True,
            truncation=True,
            max_length=args.max_len,
            return_tensors="pt",
        )

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, rows: list[dict]):
            self.enc = encode(rows)
            if multilabel:
                self.labels = torch.tensor([r["label"] for r in rows], dtype=torch.float)
            else:
                self.labels = torch.tensor([label2id[r["type"]] for r in rows])

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            return {k: v[i] for k, v in self.enc.items()} | {"labels": self.labels[i]}

    train_ds, test_ds = Dataset(train_rows), Dataset(test_rows)

    train_args = TrainingArguments(
        output_dir=str(out / f"{args.task}.{Path(args.model).name}.run"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_strategy="no",
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        seed=42,
    )
    trainer = Trainer(model=model, args=train_args, train_dataset=train_ds, eval_dataset=test_ds)
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    import numpy as np

    from legalrag.eval import metrics

    preds = trainer.predict(test_ds).predictions
    if multilabel:
        logits = np.asarray(preds)
        result = {
            "gold_density": round(float(np.mean([sum(r["label"]) for r in test_rows])), 4),
            "logit_stats": {
                "frac_pos": round(float(np.mean(logits > 0)), 4),
                "mean": round(float(logits.mean()), 4),
                "std": round(float(logits.std()), 4),
                "p25": round(float(np.percentile(logits, 25)), 4),
                "p50": round(float(np.percentile(logits, 50)), 4),
                "p75": round(float(np.percentile(logits, 75)), 4),
            },
        }
        y_pred = (logits > 0).astype(int).tolist()
        y_true = [r["label"] for r in test_rows]
        result.update(metrics.multilabelStats(y_true, y_pred, labels))
    else:
        y_pred_labels = [labels[i] for i in np.argmax(preds, axis=1)]
        y_true_labels = [r["type"] for r in test_rows]
        result = metrics.multiclassStats(y_true_labels, y_pred_labels)
    result["task"] = args.task
    result["model"] = args.model
    result["train_n"] = len(train_rows)
    result["test_n"] = len(test_rows)
    result["elapsed_s"] = round(elapsed, 1)

    dst = out / f"{args.task}.neural.json"
    dst.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "classes"}, indent=2))
    print(f"[train_neural] wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())