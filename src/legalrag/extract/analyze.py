# Section-level analysis: hybrid clause classification across a document.
#
# `analyzeSections` is the Extract engine's entry point (docs M5). The
# fast-lane regex classifier is authoritative when it fires; sections it marks
# `unknown` are handed to the trained classifier fallback (classifier.py) in a
# single batched embedding pass. Output rows carry the clause type, the method
# that decided it, and a raw confidence (fast-lane evidence count or softmax
# probability).
from __future__ import annotations

from typing import Any, Protocol

from .classifier import encodeTexts
from .fast_lane import classifyClause
from .taxonomy import UNKNOWN

# Method labels for the per-section result.
METHOD_FAST_LANE = "fast_lane"
METHOD_CLASSIFIER = "classifier"
METHOD_UNKNOWN = "unknown"


class FallbackClassifier(Protocol):
    """The slice of TrainedClassifier analyzeSections depends on."""

    model_name: str
    classes: list[str]
    threshold: float

    def predict_proba(self, vectors: Any) -> Any: ...  # (n, n_classes) softmax


def _text(sec: dict[str, Any]) -> str:
    """Section text from either engine row shape."""
    return sec.get("text") or sec.get("raw_text") or ""


def analyzeSections(
    sections: list[dict[str, Any]],
    fallback: FallbackClassifier | None = None,
) -> list[dict[str, Any]]:
    """Classify every section; returns one row per input section.

    Rows: {id, text, clause_type, method, confidence}. Confidence is the
    fast-lane evidence count when the fast lane fires, else the classifier's
    max softmax probability (raw, unadjusted). Sections with neither yield
    clause_type=UNKNOWN, method="unknown", confidence=0.0.
    """
    results: list[dict[str, Any]] = []
    unknown_idx: list[int] = []

    for i, sec in enumerate(sections):
        text = _text(sec)
        clause_type, count = classifyClause(text)
        if clause_type == UNKNOWN:
            results.append(
                {
                    "id": sec.get("id", ""),
                    "text": text,
                    "clause_type": UNKNOWN,
                    "method": METHOD_UNKNOWN,
                    "confidence": 0.0,
                }
            )
            unknown_idx.append(i)
        else:
            results.append(
                {
                    "id": sec.get("id", ""),
                    "text": text,
                    "clause_type": clause_type,
                    "method": METHOD_FAST_LANE,
                    "confidence": float(count),
                }
            )

    if fallback is not None and unknown_idx:
        texts = [_text(sections[i]) for i in unknown_idx]
        probs = fallback.predict_proba(encodeTexts(texts, fallback.model_name))
        for j, row in enumerate(probs):
            k = int(row.argmax())
            label = fallback.classes[k]
            prob = float(row[k])
            out = results[unknown_idx[j]]
            if prob >= fallback.threshold:
                out["clause_type"] = label
                out["method"] = METHOD_CLASSIFIER
                out["confidence"] = prob
            else:
                out["confidence"] = prob

    return results
