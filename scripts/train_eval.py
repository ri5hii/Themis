"""Train + evaluate baselines for each supervised task.

Usage:
    python scripts/train_eval.py --task redflag_paragraph --model deterministic
    python scripts/train_eval.py --task deontic_multilabel --model tfidf

Split selection:
    --splits         base split dir (default data/splits); train/test/val dirs
    --train-splits   override where train rows come from (default --splits)
    --test-splits    override where val/test rows come from (default --splits)
    --mix-dir        extra train dir; its <task>.train.jsonl is appended to
                     the train rows (plain concat, no resampling)

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


def _load_train_rows(train_dir: Path, mix_dir: Path | None, task: str) -> list[dict]:
    """Train rows from train_dir, plus (plain concat) rows from mix_dir."""
    rows = _load_split(train_dir, task, "train")
    if mix_dir is not None:
        rows = rows + _load_split(mix_dir, task, "train")
    return rows


def _text(rows: list[dict]) -> list[str]:
    """Feature text per row: prefer raw_text (full paragraph), fall back to text."""
    return [str(r.get("raw_text") or r.get("text", "")) for r in rows]


def _make_x(vec, rows: list[dict], use_triggers: bool, use_party: bool):
    import numpy as np
    from scipy import sparse

    texts = _text(rows)
    X = vec.transform(texts)
    cols = []
    if use_triggers:
        cols.append(np.asarray(features.triggerFeatures(texts), dtype=float))
    if use_party:
        cols.append(np.asarray([features.partyVector(r.get("party", "")) for r in rows], dtype=float))
    if not cols:
        return X
    return sparse.hstack([X, sparse.csr_matrix(np.hstack(cols))]).tocsr()


def _fit(
    rows: list[dict],
    model: str,
    downsample_none: bool,
    ngram: tuple[int, int] = (1, 1),
    max_features: int = 20000,
    use_party: bool = False,
):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier

    use_triggers = model == "deterministic"
    if downsample_none and "type" in rows[0]:
        import random

        rng = random.Random(0)
        pos = [r for r in rows if r["type"] != "none"]
        neg = [r for r in rows if r["type"] == "none"]
        n_keep = min(len(neg), len(pos) * 3)
        neg = rng.sample(neg, n_keep)
        rows = pos + neg
    vec = TfidfVectorizer(max_features=max_features, sublinear_tf=True, ngram_range=ngram)
    vec.fit(_text(rows))
    kwargs = {"class_weight": "balanced"} if model == "deterministic" else {}
    X = _make_x(vec, rows, use_triggers, use_party)
    if "type" in rows[0]:
        y = [r["type"] for r in rows]
        clf = LogisticRegression(max_iter=1000, **kwargs)
        clf.fit(X, y)
    else:
        Y = [r["label"] for r in rows]
        clf = MultiOutputClassifier(LogisticRegression(max_iter=1000, **kwargs))
        clf.fit(X, Y)
    return vec, clf


def _predict(
    vec,
    clf,
    rows: list[dict],
    model: str,
    thresholds: dict[str, float] | None = None,
    use_party: bool = False,
):
    X = _make_x(vec, rows, model == "deterministic", use_party)
    texts = _text(rows)
    if thresholds is not None:
        proba = clf.predict_proba(X)
        y_pred = [
            [1 if proba[j][i][1] >= thresholds[lab] else 0 for j, lab in enumerate(MULTILABEL_LABELS)]
            for i in range(len(texts))
        ]
        return texts, y_pred
    pred = clf.predict(X)
    return texts, list(pred)


def _calibrate_thresholds(vec, clf, rows: list[dict], model: str, use_party: bool = False) -> dict[str, float]:
    """Per-label decision thresholds maximizing val micro-F1 (multilabel only)."""
    if not rows:
        return {}
    import numpy as np

    X = _make_x(vec, rows, model == "deterministic", use_party)
    proba = clf.predict_proba(X)
    y_true = np.asarray([r["label"] for r in rows], dtype=int)
    best = {}
    grid = np.linspace(0.05, 0.95, 19)
    for i, lab in enumerate(MULTILABEL_LABELS):
        col_true = y_true[:, i]
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            yp = (proba[i][:, 1] >= t).astype(int)
            tp = int(np.sum((col_true == 1) & (yp == 1)))
            fp = int(np.sum((col_true == 0) & (yp == 1)))
            fn = int(np.sum((col_true == 1) & (yp == 0)))
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best[lab] = float(best_t)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, default="redflag_paragraph")
    parser.add_argument("--model", choices=MODELS, default="tfidf")
    parser.add_argument("--downsample-none", action="store_true", help="cap 'none' rows at 3x positives (multiclass)")
    parser.add_argument("--calibrate", action="store_true", help="per-label threshold calibration on val split (multilabel)")
    parser.add_argument("--no-party", action="store_true", help="disable tenant/landlord party feature (deontic_multilabel)")
    parser.add_argument("--max-f", type=int, default=20000, help="TF-IDF max_features (default: 20000)")
    parser.add_argument("--ngram", type=str, default=None, help="TF-IDF n-gram range, e.g. '1,2' (default: task default)")
    parser.add_argument("--seed", type=int, default=tasks.DEFAULT_SEED)
    parser.add_argument("--splits", default=str(Path("data/splits")))
    parser.add_argument("--train-splits", default=None, help="train split dir (default: --splits)")
    parser.add_argument("--test-splits", default=None, help="val/test split dir (default: --splits)")
    parser.add_argument("--mix-dir", default=None, help="extra train dir, appended to train rows (plain concat)")
    parser.add_argument("--out", default=str(Path("eval/artifacts")))
    args = parser.parse_args()

    split_dir = Path(args.splits)
    train_dir = Path(args.train_splits) if args.train_splits else split_dir
    test_dir = Path(args.test_splits) if args.test_splits else split_dir
    mix_dir = Path(args.mix_dir) if args.mix_dir else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_rows = _load_train_rows(train_dir, mix_dir, args.task)
    test_rows = _load_split(test_dir, args.task, "test")
    ngram = tuple(int(x) for x in (args.ngram or ("1,2" if args.task == "deontic_multilabel" else "1,1")).split(","))
    use_party = args.task == "deontic_multilabel" and not args.no_party
    t0 = time.time()
    vec, clf = _fit(train_rows, args.model, args.downsample_none, ngram, args.max_f, use_party)
    thresholds = None
    if args.calibrate and args.task == "deontic_multilabel":
        val_rows = _load_split(test_dir, args.task, "val")
        thresholds = _calibrate_thresholds(vec, clf, val_rows, args.model, use_party)
    _, pred = _predict(vec, clf, test_rows, args.model, thresholds, use_party)

    if args.task == "redflag_paragraph":
        y_true = [r["type"] for r in test_rows]
        result = metrics.multiclassStats(y_true, pred)
    else:
        y_true = [r["label"] for r in test_rows]
        result = metrics.multilabelStats(y_true, pred, MULTILABEL_LABELS)
    result["task"] = args.task
    result["model"] = args.model
    result["ngram"] = args.ngram or ("1,2" if args.task == "deontic_multilabel" else "1,1")
    result["downsample_none"] = args.downsample_none
    result["calibrate"] = args.calibrate
    result["party"] = use_party
    result["max_features"] = args.max_f
    if thresholds:
        result["thresholds"] = thresholds
    result["train_n"] = len(train_rows)
    result["test_n"] = len(test_rows)
    result["train_splits"] = str(train_dir)
    result["test_splits"] = str(test_dir)
    result["mix_dir"] = args.mix_dir
    result["elapsed_s"] = round(time.time() - t0, 1)

    dst = out / f"{args.task}.{args.model}{'.ds' if args.downsample_none else ''}{'.cal' if args.calibrate else ''}{'.party' if use_party else ''}.json"
    dst.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[train_eval] {args.task}/{args.model}: wrote {dst}")
    print(json.dumps({k: v for k, v in result.items() if k not in ("classes", "labels")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())