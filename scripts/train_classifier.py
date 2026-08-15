#!/usr/bin/env python3
"""Train the clause classifier fallback on auto-labeled real sections.

Training data: data/annotated/leivaditi_leases.jsonl — sections auto-labeled
by the fast-lane (`type_fast_lane`); only non-`unknown` rows are used
(candidate labels, not gold). A multinomial logistic regression (linear softmax
head) is fit on frozen LegalBERT mean-pooled embeddings.

Outputs:
    models/classifier.npz     -- weights + classes + threshold (load-only dep)
    models/classifier.joblib  -- full sklearn estimator checkpoint

Usage:
    python scripts/train_classifier.py [--model nlpaueb/legal-bert-base-uncased]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from legalrag.extract.classifier import encodeTexts

ROOT = Path(__file__).resolve().parent.parent
LEASES = ROOT / "data" / "annotated" / "leivaditi_leases.jsonl"
DEFAULT_MODEL = "nlpaueb/legal-bert-base-uncased"


def load_auto_labeled(path: Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        label = row.get("type_fast_lane")
        text = row.get("text", "")
        if label and label != "unknown" and text.strip():
            texts.append(text)
            labels.append(label)
    # Drop classes with <2 members: they cannot be stratified and are not
    # trainable with a linear head (e.g. holdover n=1).
    from collections import Counter

    keep = {c for c, n in Counter(labels).items() if n >= 2}
    kept_texts = [t for t, c in zip(texts, labels) if c in keep]
    kept_labels = [c for c in labels if c in keep]
    return kept_texts, kept_labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="LegalBERT-family model for embeddings")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    texts, labels = load_auto_labeled(LEASES)
    print(f"[train_classifier] {len(texts)} auto-labeled sections, classes={len(set(labels))}")

    X = encodeTexts(texts, args.model)
    y = np.asarray(labels)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, stratify=y)

    clf = LogisticRegression(
        max_iter=1000,
        C=10.0,
        class_weight="balanced",
        random_state=args.seed,
        solver="lbfgs",
    )
    clf.fit(X_tr, y_tr)

    acc = clf.score(X_te, y_te)
    print(f"[train_classifier] test accuracy: {acc:.4f}")
    print(classification_report(y_te, clf.predict(X_te), zero_division=0))

    classes = [str(c) for c in clf.classes_]
    npz_path = ROOT / "models" / "classifier.npz"
    joblib_path = ROOT / "models" / "classifier.joblib"
    from legalrag.extract.classifier import TrainedClassifier

    tc = TrainedClassifier.from_sklearn(args.model, clf, classes)
    tc.save(npz_path)
    joblib.dump(clf, joblib_path)
    (ROOT / "models" / "classifier_meta.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "classes": classes,
                "n_train": len(texts),
                "test_accuracy": round(float(acc), 4),
                "threshold": tc.threshold,
                "npz": str(npz_path),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[train_classifier] saved {npz_path} and {joblib_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
