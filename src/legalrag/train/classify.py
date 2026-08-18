"""Train the clause classifier fallback on auto- or gold-labeled sections.

A multinomial logistic regression (linear softmax head) is fit on frozen
LegalBERT mean-pooled embeddings. Outputs mirror the legacy script:
models/classifier.npz + .joblib + classifier_meta.json.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from legalrag.extract.classifier import TrainedClassifier, encodeTexts
from legalrag.train.artifacts import artifact_stamp, backup_existing

DEFAULT_MODEL = "nlpaueb/legal-bert-base-uncased"


def train_classifier(
    texts: list[str],
    labels: list[str],
    model_name: str = DEFAULT_MODEL,
    test_size: float = 0.2,
    seed: int = 42,
    out_dir: Path | None = None,
    backup_root: Path | None = None,
    encode_fn: Callable[[list[str], str], np.ndarray] = encodeTexts,
    verbose: bool = True,
) -> dict:
    """Fit the linear head on frozen embeddings; persist npz/joblib/meta.

    Returns the metadata dict written to classifier_meta.json.
    """
    from collections import Counter

    keep = {c for c, n in Counter(labels).items() if n >= 2}
    texts = [t for t, c in zip(texts, labels) if c in keep]
    labels = [c for c in labels if c in keep]
    if len(texts) < 4:
        raise ValueError(f"need >=4 labeled rows after class filtering, got {len(texts)}")

    X = encode_fn(texts, model_name)
    y = np.asarray(labels)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    clf = LogisticRegression(
        max_iter=1000,
        C=10.0,
        class_weight="balanced",
        random_state=seed,
        solver="lbfgs",
    )
    clf.fit(X_tr, y_tr)

    acc = clf.score(X_te, y_te)
    if verbose:
        print(f"[classify] {len(texts)} labeled sections, classes={len(set(labels))}")
        print(f"[classify] test accuracy: {acc:.4f}")
        print(classification_report(y_te, clf.predict(X_te), zero_division=0))

    classes = [str(c) for c in clf.classes_]
    out_dir = out_dir or Path.cwd() / "models"
    backed = backup_existing(
        out_dir,
        ["classifier.npz", "classifier.joblib", "classifier_meta.json"],
        backup_root or (out_dir / "backups"),
        "classifier",
    )
    npz_path = out_dir / "classifier.npz"
    joblib_path = out_dir / "classifier.joblib"
    tc = TrainedClassifier.from_sklearn(model_name, clf, classes)
    tc.save(npz_path)
    joblib.dump(clf, joblib_path)
    meta = {
        "model": model_name,
        "classes": classes,
        "n_train": len(texts),
        "test_accuracy": round(float(acc), 4),
        "threshold": tc.threshold,
        "npz": str(npz_path),
    }
    meta.update(artifact_stamp(list(zip(texts, labels))))
    (out_dir / "classifier_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    if verbose:
        print(f"[classify] saved {npz_path} and {joblib_path}")
        if backed:
            print(f"[classify] previous artifact backed up -> {backed}")
    return meta