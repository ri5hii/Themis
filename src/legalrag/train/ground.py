"""Train the grounding relevance gate (audit-v3 fix, trainable artifact).

The gate is a logistic head over [dense cosine score, BM25 lexical score]
trained on positive pairs - each rule's statute_query paired with its
lexically anchor-matched statute chunk - versus random negative chunks.
grounding.py applies it: below-threshold dense hits fall through to the
static fallback instead of stamping an irrelevant citation.

Outputs:
    <out>/head.npz      -- logistic weights + intercept
    <out>/meta.json     -- model, threshold, features, BM25 corpus stats
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from legalrag.embeddings import encodeMeanPooled
from legalrag.risk.gate import BM25, HEAD_NPZ, META_JSON, Gate
from legalrag.train.artifacts import artifact_stamp, backup_existing
from legalrag.train.data import build_statute_pairs

DEFAULT_MODEL = "nlpaueb/legal-bert-base-uncased"


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-9, None)


def _dense_scores(
    queries: list[str],
    texts: list[str],
    model_name: str,
    encode_fn=encodeMeanPooled,
    batch_size: int = 32,
) -> np.ndarray:
    """Pairwise cosine similarity between query and chunk embeddings."""
    q_vecs = _normalize(encode_fn(queries, model_name, batch_size=batch_size))
    d_vecs = _normalize(encode_fn(texts, model_name, batch_size=batch_size))
    return (q_vecs * d_vecs).sum(axis=1)


def train_gate(
    chunks: list[dict],
    rules: list,
    model_name: str = DEFAULT_MODEL,
    out_dir: Path | None = None,
    backup_root: Path | None = None,
    seed: int = 42,
    neg_per_pos: int = 3,
    test_size: float = 0.2,
    encode_fn=encodeMeanPooled,
    verbose: bool = True,
) -> dict:
    """Fit and persist the gate; returns the metadata dict."""
    queries, texts, labels = build_statute_pairs(chunks, rules, neg_per_pos=neg_per_pos, seed=seed)
    if len(set(labels)) < 2 or len(labels) < 8:
        raise ValueError(
            f"need >=8 pairs with both labels for gate training, got {len(labels)} pairs "
            f"(labels={sorted(set(labels))})"
        )

    dense = _dense_scores(queries, texts, model_name, encode_fn=encode_fn)
    bm25 = BM25.fit([c["text"] for c in chunks])
    bm25_scores = np.asarray([bm25.score(q, t) for q, t in zip(queries, texts)])
    X = np.column_stack([dense, bm25_scores])
    y = np.asarray(labels)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    if verbose:
        print(f"[ground] {len(labels)} pairs ({sum(labels)} pos / {len(labels) - sum(labels)} neg)")
        print(f"[ground] gate accuracy: {acc:.4f}")

    gate = Gate(
        model_name=model_name,
        weights=clf.coef_[0],
        intercept=clf.intercept_[0],
        threshold=0.5,
        bm25=bm25,
    )
    out_dir = out_dir or Path.cwd() / "models" / "grounding"
    backed = backup_existing(
        out_dir,
        [HEAD_NPZ, META_JSON],
        backup_root or (out_dir.parent / "backups"),
        "grounding",
    )
    stamp = artifact_stamp(list(zip(queries, texts, labels)))
    gate.save(out_dir, extra_meta=stamp)
    if verbose:
        print(f"[ground] saved gate -> {out_dir}")
        if backed:
            print(f"[ground] previous gate backed up -> {backed}")
    meta = {
        "model": model_name,
        "features": list(Gate.FEATURES),
        "threshold": gate.threshold,
        "n_pairs": len(labels),
        "n_pos": int(sum(labels)),
        "test_accuracy": round(float(acc), 4),
        "head": str(out_dir / "head.npz"),
    }
    meta.update(stamp)
    return meta