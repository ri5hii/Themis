# Trained clause classifier: fallback for fast-lane `unknown` sections.
#
# A linear softmax head (multinomial logistic regression) fit on frozen
# LegalBERT mean-pooled embeddings. `TrainedClassifier` wraps the fitted head
# with a refuse contract: predictions whose max softmax probability is below
# THRESHOLD, or whose top-two probability gap is below MARGIN, collapse to
# UNKNOWN so the engine never overrules a confident fast-lane label with a
# low-quality guess (docs/progress.md §6.14).
from __future__ import annotations

from pathlib import Path

import numpy as np

from .fast_lane import classifyClause, confidenceFromCount
from .taxonomy import UNKNOWN

# Confidence floor for the fallback: predictions below this become UNKNOWN.
THRESHOLD = 0.4

# Refuse margin: when the gap between the top two softmax probabilities is
# below this, the prediction is treated as UNKNOWN. OOD non-clause content
# (preamble, signatures, boilerplate) gets confidently-wrong softmax peaks;
# close top-two scores are the low-quality signal the fallback should refuse
# instead of guessing. Tuned on scripts/eval_ood.py: margin=0.5 lifts OOD
# hybrid overall accuracy 0.144 -> 0.340 with no clause-only regression.
MARGIN = 0.5


def refuseRow(
    row: np.ndarray,
    classes: list[str],
    threshold: float,
    margin: float,
) -> tuple[str, float]:
    """Argmax label with threshold + margin refusal; returns (label, prob).

    Shared by `TrainedClassifier.predict`/`classifyClause` and the analyze
    engine's fallback loop so the refusal rules live in one place. A row whose
    top probability is below `threshold`, or whose top-two gap is below
    `margin`, collapses to UNKNOWN (the fast-lane refuse signal).
    """
    order = np.argsort(row)[::-1]
    k = int(order[0])
    prob = float(row[k])
    second = float(row[order[1]]) if len(row) > 1 else 0.0
    if prob >= threshold and prob - second >= margin:
        return classes[k], prob
    return UNKNOWN, prob


class TrainedClassifier:
    """Linear softmax clause classifier over frozen embeddings."""

    def __init__(
        self,
        model_name: str,
        classes: list[str],
        weights: np.ndarray,
        intercept: np.ndarray,
        threshold: float = THRESHOLD,
        margin: float = MARGIN,
    ) -> None:
        self.model_name = model_name
        self.classes = classes
        self.weights = weights
        self.intercept = intercept
        self.threshold = threshold
        self.margin = margin
        self._class_index = {c: i for i, c in enumerate(classes)}

    @classmethod
    def from_sklearn(
        cls,
        model_name: str,
        clf,
        classes: list[str],
        threshold: float = THRESHOLD,
        margin: float = MARGIN,
    ) -> TrainedClassifier:
        """Wrap a fitted sklearn LogisticRegression (multinomial softmax)."""
        weights = np.asarray(clf.coef_, dtype="float32")
        intercept = np.asarray(clf.intercept_, dtype="float32")
        return cls(model_name, list(classes), weights, intercept, threshold, margin)

    def predict_proba(self, vectors: np.ndarray) -> np.ndarray:
        """Softmax over per-class logits for each row; shape (n, n_classes)."""
        logits = vectors @ self.weights.T + self.intercept
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, vectors: np.ndarray) -> list[str]:
        """Argmax label per row; below threshold or narrow top-two margin collapses to UNKNOWN."""
        probs = self.predict_proba(vectors)
        return [refuseRow(row, self.classes, self.threshold, self.margin)[0] for row in probs]

    def classifyClause(self, text: str) -> tuple[str, float]:
        """Fast-lane first; fall back to this classifier on UNKNOWN.

        Returns (clause_type, confidence). Confidence is the fast-lane
        confidence (evidence count mapped to [0, 1] via
        confidenceFromCount) when the fast lane fires, else the classifier's
        max softmax probability after threshold/margin refusal.
        """
        fast, count = classifyClause(text)
        if fast != UNKNOWN:
            return fast, confidenceFromCount(count)

        vec = encodeTexts([text], self.model_name)
        probs = self.predict_proba(vec)[0]
        return refuseRow(probs, self.classes, self.threshold, self.margin)

    def save(self, path: Path) -> None:
        """Persist weights + metadata to a .npz (stdlib-compatible load)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            weights=self.weights,
            intercept=self.intercept,
            classes=np.asarray(self.classes),
            model_name=self.model_name,
            threshold=self.threshold,
            margin=self.margin,
        )

    @classmethod
    def load(cls, path: Path) -> TrainedClassifier:
        data = np.load(path, allow_pickle=True)
        return cls(
            model_name=str(data["model_name"]),
            classes=[str(c) for c in data["classes"]],
            weights=data["weights"],
            intercept=data["intercept"],
            threshold=float(data["threshold"]),
            margin=float(data["margin"]) if "margin" in data else MARGIN,
        )


def _cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def _default_device() -> str:
    return "cuda" if _cuda_available() else "cpu"


def encodeTexts(
    texts: list[str],
    model_name: str,
    batch_size: int = 16,
    device: str | None = None,
) -> np.ndarray:
    """Mean-pooled LegalBERT embeddings for the linear head (raw, unnormalized)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = device or _default_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    vecs: list[np.ndarray] = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)
            out = model(**enc)
            token_embeds = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float().to(token_embeds.device)
            summed = (token_embeds * mask).sum(1)
            counts = mask.sum(1).clamp(min=1e-9)
            vecs.append((summed / counts).cpu().numpy())
    return np.vstack(vecs).astype("float32")
