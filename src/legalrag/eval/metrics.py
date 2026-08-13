# Evaluation metrics for the v0.2.x pipeline.
#
# Pure numpy, no sklearn dependency in the hot path so results are simple and
# deterministic. Multi-class (redflag type) uses macro/micro precision-recall-F1;
# multi-label (deontic) computes per-label scores and micro-aggregates.
from __future__ import annotations

from typing import Any

import numpy as np


def _f1(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return round(2 * p * r / (p + r), 4)


def _scores(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": _f1(p, r)}


def multiclassStats(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    """Per-class and macro/micro aggregate stats for a multiclass task."""
    labels = sorted(set(y_true) | set(y_pred))
    per_class: dict[str, dict[str, Any]] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        per_class[label] = _scores(tp, fp, fn)
    exact = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = exact / len(y_true) if y_true else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "n": len(y_true),
        "classes": per_class,
        "macro": {
            "precision": round(float(np.mean([c["precision"] for c in per_class.values()])), 4) if per_class else 0.0,
            "recall": round(float(np.mean([c["recall"] for c in per_class.values()])), 4) if per_class else 0.0,
            "f1": round(
                float(np.mean([c["f1"] for c in per_class.values()])),
                4,
            )
            if per_class
            else 0.0,
        },
        # For multiclass, micro precision = micro recall = micro F1 = accuracy.
        "micro": {
            "precision": round(accuracy, 4),
            "recall": round(accuracy, 4),
            "f1": round(accuracy, 4),
        },
    }


def multilabelStats(
    y_true: list[list[int]],
    y_pred: list[list[int]],
    labels: list[str],
) -> dict[str, Any]:
    """Per-label and micro-aggregate stats for a multi-label task."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    per_label: dict[str, dict[str, Any]] = {}
    for i, name in enumerate(labels):
        tp = int(np.sum((yt[:, i] == 1) & (yp[:, i] == 1)))
        fp = int(np.sum((yt[:, i] == 0) & (yp[:, i] == 1)))
        fn = int(np.sum((yt[:, i] == 1) & (yp[:, i] == 0)))
        per_label[name] = _scores(tp, fp, fn)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    exact = float(np.mean(np.all(yt == yp, axis=1))) if len(yt) else 0.0
    return {
        "n": len(yt),
        "exact_match_ratio": round(exact, 4),
        "labels": per_label,
        "micro": _scores(tp, fp, fn),
    }