# Task definitions for the v0.2.x training pipeline.
#
# Each task maps a cleaned corpus to a supervised learning setup with a
# deterministic train/val/test split (fixed seed), so training and eval are
# reproducible.
#
# Tasks:
#   redflag_paragraph  multi-class redflag type over Leivaditi full paragraphs
#   deontic_span       sentence-level redflag span detection (19 types)
#   deontic_multilabel multi-label deontic modality over LEXDEMOD sentences
from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

SPLITS = ("train", "val", "test")
DEFAULT_SEED = 42
SPLIT_RATIOS = (0.8, 0.1, 0.1)


def loadJsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def splitRows(
    rows: Iterable[dict[str, Any]],
    seed: int = DEFAULT_SEED,
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
) -> dict[str, list[dict[str, Any]]]:
    """Deterministic shuffled split. Stratifies by 'type'/'deontic_types' key if present."""
    import random

    items = list(rows)
    rng = random.Random(seed)
    key_fn = None
    if items and ("type" in items[0] and items[0].get("type")):
        key_fn = lambda r: str(r.get("type", "none"))
    elif items and "deontic_types" in items[0]:
        key_fn = lambda r: ",".join(sorted(r.get("deontic_types", [])))
    if key_fn is not None:
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in items:
            buckets.setdefault(key_fn(row), []).append(row)
        out: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
        for group in buckets.values():
            rng.shuffle(group)
            n = len(group)
            n_tr, n_va = int(n * ratios[0]), int(n * ratios[1])
            out["train"].extend(group[:n_tr])
            out["val"].extend(group[n_tr : n_tr + n_va])
            out["test"].extend(group[n_tr + n_va :])
        return out
    rng.shuffle(items)
    n = len(items)
    n_tr, n_va = int(n * ratios[0]), int(n * ratios[1])
    return {
        "train": items[:n_tr],
        "val": items[n_tr : n_tr + n_va],
        "test": items[n_tr + n_va :],
    }