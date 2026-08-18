"""Trained relevance gate for statute grounding.

Dense retrieval over mean-pooled LegalBERT embeddings tends to favor long,
generic chunks (e.g. a tribunal-administration section) regardless of the
query, so dense hits are often irrelevant. The gate is a logistic head over
two features - the dense cosine score and a BM25 lexical score - trained on
(rule query, lexically anchor-matched chunk) positives vs random negatives.
grounding.py applies it: below-threshold hits fall through to the static
fallback instead of stamping an irrelevant citation.

Artifacts (written by ``themis train ground``):
    <out>/head.npz      -- logistic weights + intercept
    <out>/meta.json     -- model, threshold, features, BM25 corpus stats
"""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

HEAD_NPZ = "head.npz"
META_JSON = "meta.json"
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens."""
    return TOKEN_RE.findall(text.lower())


class BM25:
    """Pure-python BM25 scorer with persisted corpus statistics."""

    def __init__(self, idf: dict[str, float], avgdl: float, n_docs: int, k1: float = 1.5, b: float = 0.75) -> None:
        self.idf = idf
        self.avgdl = avgdl
        self.n_docs = n_docs
        self.k1 = k1
        self.b = b

    @classmethod
    def fit(cls, texts: list[str], k1: float = 1.5, b: float = 0.75) -> BM25:
        df: dict[str, int] = {}
        total_len = 0
        for text in texts:
            tokens = set(tokenize(text))
            for tok in tokens:
                df[tok] = df.get(tok, 0) + 1
            total_len += len(tokenize(text))
        n_docs = len(texts)
        avgdl = total_len / max(n_docs, 1)
        idf = {
            tok: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
            for tok, freq in df.items()
        }
        return cls(idf, avgdl, n_docs, k1, b)

    def score(self, query: str, doc: str) -> float:
        q_tokens = tokenize(query)
        if not q_tokens:
            return 0.0
        doc_tokens = tokenize(doc)
        dl = len(doc_tokens)
        denom = 1 - self.b + self.b * (dl / max(self.avgdl, 1e-9))
        score = 0.0
        for tok in q_tokens:
            freq = doc_tokens.count(tok)
            if freq == 0 or tok not in self.idf:
                continue
            score += self.idf[tok] * (freq * (self.k1 + 1)) / (freq + self.k1 * denom)
        return score

    def to_dict(self) -> dict:
        return {"idf": self.idf, "avgdl": self.avgdl, "n_docs": self.n_docs, "k1": self.k1, "b": self.b}

    @classmethod
    def from_dict(cls, data: dict) -> BM25:
        return cls(data["idf"], data["avgdl"], data["n_docs"], data["k1"], data["b"])


class Gate:
    """Logistic relevance gate: P(relevant | dense_score, bm25_score)."""

    FEATURES = ("dense_score", "bm25_score")

    def __init__(
        self,
        model_name: str,
        weights: np.ndarray,
        intercept: float,
        threshold: float,
        bm25: BM25,
    ) -> None:
        self.model_name = model_name
        self.weights = np.asarray(weights, dtype="float32")
        self.intercept = float(intercept)
        self.threshold = float(threshold)
        self.bm25 = bm25

    def score(self, query: str, chunk_text: str, dense_score: float) -> float:
        """P(relevant) from dense cosine score and BM25 lexical score."""
        bm25_score = self.bm25.score(query, chunk_text)
        features = np.asarray([[dense_score, bm25_score]], dtype="float32")
        z = float((features @ self.weights + self.intercept)[0])
        return 1.0 / (1.0 + math.exp(-z))

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_dir / HEAD_NPZ,
            weights=self.weights,
            intercept=np.asarray([self.intercept]),
        )
        (out_dir / META_JSON).write_text(
            json.dumps(
                {
                    "model": self.model_name,
                    "features": self.FEATURES,
                    "threshold": self.threshold,
                    "bm25": self.bm25.to_dict(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, out_dir: Path) -> Gate:
        data = np.load(out_dir / HEAD_NPZ)
        meta = json.loads((out_dir / META_JSON).read_text(encoding="utf-8"))
        return cls(
            model_name=meta["model"],
            weights=data["weights"],
            intercept=float(np.asarray(data["intercept"])[0]),
            threshold=float(meta["threshold"]),
            bm25=BM25.from_dict(meta["bm25"]),
        )


@lru_cache(maxsize=1)
def _loadGate(gate_dir: Path) -> Gate | None:
    """Cached gate load; None when absent or corrupt (ungated behavior)."""
    if not (gate_dir / HEAD_NPZ).exists() or not (gate_dir / META_JSON).exists():
        return None
    try:
        return Gate.load(gate_dir)
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None