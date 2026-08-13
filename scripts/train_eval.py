"""Train + evaluate baselines for each supervised task.

Usage:
    python scripts/train_eval.py --task redflag_paragraph --model deterministic
    python scripts/train_eval.py --task deontic_multilabel --model tfidf

Models:
  tfidf        TF-IDF + logistic regression (v0.1.1 baseline)
  deterministic TF-IDF + deontic trigger features + balanced class weights
                (+ optional 'none' downsampling for the multiclass task)

Metrics via legalrag.eval.metrics; JSON result written to eval/artifacts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legalrag import tasks
from legalrag.eval import features, metrics

TASKS = ("redflag_paragraph", "deontic_multilabel")
MODELS = ("tfidf", "deterministic")
MULTILABEL_LABELS = ["obl", "ent", "pro", "per", "oth", "nen", "none"]


def _load_split(dir_: Path, name: str, split: str) -> list[dict]:
    return tasks.loadJsonl(str(dir_ / f"{name}.{split}.jsonl"))


def _make_x(vec, texts: list[str], use_triggers: bool):
    import numpy as np
    from scipy import sparse

    X = vec.transform(texts)
    if not use_triggers:
        return X
    trig = np.asarray(features.triggerFeatures(texts), dtype=float)
    return sparse.hstack([X, sparse.csr_matrix(trig)]).tocsr()


def _fit(rows: list[dict], model: str, downsample_none: bool):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier

    use_triggers = model == "deterministic"
    texts = [r["text"] for r in rows]
    if downsample_none and "type" in rows[0]:
        import random

        rng = random.Random(0)
        pos = [r for r in rows if r["type"] != "none"]
        neg = [r for r in rows if r["type"] == "none"]
        n_keep = min(len(neg), len(pos) * 3)
        neg = rng.sample(neg, n_keep)
        rows = pos + neg
        texts = [r["text"] for r in rows]
    vec = TfidfVectorizer(max_features=20000, sublinear_tf=True)
    vec.fit(texts)
    kwargs = {"class_weight": "balanced"} if model == "deterministic" else {}
    if "type" in rows[0]:
        y = [r["type"] for r in rows]
        X = _make_x(vec, texts, use_triggers)
        clf = LogisticRegression(max_iter=1000, **kwargs)
        clf.fit(X, y)
    else:
        Y = [r["label"] for r in rows]
        X = _make_x(vec, texts, use_triggers)
        clf = MultiOutputClassifier(LogisticRegression(max_iter=1000, **kwargs))
        clf.fit(X, Y)
    return vec, clf


def _predict(vec, clf, rows: list[dict], model: str):
    texts = [r["text"] for r in rows]
    X = _make_x(vec, texts, model == "deterministic")
    pred = clf.predict(X)
    return texts, list(pred)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, default="redflag_paragraph")
    parser.add_argument("--model", choices=MODELS, default="tfidf")
    parser.add_argument("--downsample-none", action="store_true", help="cap 'none' rows at 3x positives (multiclass)")
    parser.add_argument("--seed", type=int, default=tasks.DEFAULT_SEED)
    parser.add_argument("--splits", default=str(Path("data/splits")))
    parser.add_argument("--out", default=str(Path("eval/artifacts")))
    args = parser.parse_args()

    split_dir = Path(args.splits)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_rows = _load_split(split_dir, args.task, "train")
    test_rows = _load_split(split_dir, args.task, "test")
    t0 = time.time()
    vec, clf = _fit(train_rows, args.model, args.downsample_none)
    _, pred = _predict(vec, clf, test_rows, args.model)

    if args.task == "redflag_paragraph":
        y_true = [r["type"] for r in test_rows]
        result = metrics.multiclassStats(y_true, pred)
    else:
        y_true = [r["label"] for r in test_rows]
        result = metrics.multilabelStats(y_true, pred, MULTILABEL_LABELS)
    result["task"] = args.task
    result["model"] = args.model
    result["downsample_none"] = args.downsample_none
    result["train_n"] = len(train_rows)
    result["test_n"] = len(test_rows)
    result["elapsed_s"] = round(time.time() - t0, 1)

    dst = out / f"{args.task}.{args.model}{'.ds' if args.downsample_none else ''}.json"
    dst.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[train_eval] {args.task}/{args.model}: wrote {dst}")
    print(json.dumps({k: v for k, v in result.items() if k not in ("classes", "labels")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())