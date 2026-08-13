"""Train + evaluate a lightweight baseline for each supervised task.

Usage:
    python scripts/train_eval.py [--task redflag_paragraph] [--seed 42]
                                 [--splits data/splits] [--out eval/artifacts]

Runs a TF-IDF + logistic-regression baseline on the seeded splits, computes
metrics via legalrag.eval.metrics, and writes a JSON result. The multiclass
'redflag_paragraph' and multi-label 'deontic_multilabel' tasks are supported.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legalrag import tasks
from legalrag.eval import metrics

TASKS = ("redflag_paragraph", "deontic_multilabel")


def _load_split(dir_: Path, name: str, split: str) -> list[dict]:
    return tasks.loadJsonl(str(dir_ / f"{name}.{split}.jsonl"))


def _fit_multiclass(rows: list[dict]) -> tuple:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    texts = [r["text"] for r in rows]
    y = [r["type"] for r in rows]
    vec = TfidfVectorizer(max_features=20000, sublinear_tf=True)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    return vec, clf


def _fit_multilabel(rows: list[dict]) -> tuple:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier

    texts = [r["text"] for r in rows]
    Y = [r["label"] for r in rows]
    vec = TfidfVectorizer(max_features=20000, sublinear_tf=True)
    X = vec.fit_transform(texts)
    clf = MultiOutputClassifier(LogisticRegression(max_iter=1000))
    clf.fit(X, Y)
    return vec, clf


def _predict_multiclass(vec, clf, rows: list[dict]) -> tuple[list[str], list[str]]:
    X = vec.transform([r["text"] for r in rows])
    pred = clf.predict(X)
    return [r["type"] for r in rows], list(pred)


def _predict_multilabel(vec, clf, rows: list[dict], labels: list[str]) -> tuple[list[list[int]], list[list[int]]]:
    X = vec.transform([r["text"] for r in rows])
    pred = clf.predict(X)
    return [r["label"] for r in rows], [list(map(int, p)) for p in pred]
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, default="redflag_paragraph")
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
    if args.task == "redflag_paragraph":
        vec, clf = _fit_multiclass(train_rows)
        y_true, y_pred = _predict_multiclass(vec, clf, test_rows)
        result = metrics.multiclassStats(y_true, y_pred)
    else:
        labels = ["obl", "ent", "pro", "per", "oth", "nen", "none"]
        vec, clf = _fit_multilabel(train_rows)
        y_true, y_pred = _predict_multilabel(vec, clf, test_rows, labels)
        result = metrics.multilabelStats(y_true, y_pred, labels)
    result["task"] = args.task
    result["train_n"] = len(train_rows)
    result["test_n"] = len(test_rows)
    result["elapsed_s"] = round(time.time() - t0, 1)

    dst = out / f"{args.task}.baseline.json"
    dst.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[train_eval] {args.task}: wrote {dst}")
    print(json.dumps({k: v for k, v in result.items() if k not in ("classes", "labels")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())