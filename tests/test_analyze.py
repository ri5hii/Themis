# Unit tests for the hybrid section analysis engine (extract/analyze.py).
#
# Uses a stub fallback classifier and monkeypatches encodeTexts so no model
# is loaded during the tests.
from __future__ import annotations

import numpy as np
import pytest

from legalrag.extract import analyze
from legalrag.extract.analyze import (
    METHOD_CLASSIFIER,
    METHOD_FAST_LANE,
    METHOD_UNKNOWN,
    analyzeSections,
)
from legalrag.extract.taxonomy import UNKNOWN


class StubFallback:
    """Minimal fallback classifier with a canned softmax matrix."""

    def __init__(self, probs: np.ndarray, classes: list[str], threshold: float = 0.4):
        self.model_name = "stub/model"
        self.classes = classes
        self.threshold = threshold
        self._probs = probs

    def predict_proba(self, vectors: np.ndarray) -> np.ndarray:
        self.embedded = vectors
        return self._probs


def _call(sections, fallback=None):
    return analyzeSections(sections, fallback)


def _sec(text: str, sec_id: str = "s") -> dict:
    return {"id": sec_id, "text": text}


class TestAnalyzeSections:
    def test_schema_and_fast_lane(self):
        out = _call(
            [
                _sec("The term of this lease is twelve months."),
                _sec("Tenant shall pay a security deposit of Rs. 50,000."),
            ]
        )
        assert [r["clause_type"] for r in out] == ["term", "deposit"]
        assert [r["method"] for r in out] == [METHOD_FAST_LANE, METHOD_FAST_LANE]
        assert [r["confidence"] for r in out] == [1, 2]
        assert out[0]["id"] == "s"
        assert set(out[0]) == {"id", "text", "clause_type", "method", "confidence"}

    def test_unknown_without_fallback(self):
        out = _call([_sec("Parties hereby execute this instrument.")])
        assert out[0]["clause_type"] == UNKNOWN
        assert out[0]["method"] == METHOD_UNKNOWN
        assert out[0]["confidence"] == 0.0

    def test_fallback_decides_unknown(self, monkeypatch):
        def fake_encode(texts, model_name, **kw):
            return np.zeros((len(texts), 3), dtype="float32")

        monkeypatch.setattr(analyze, "encodeTexts", fake_encode)
        stub = StubFallback(np.array([[0.1, 0.8, 0.1]]), ["term", "pets", "rent"])
        out = _call([_sec("Parties hereby execute this instrument.")], stub)
        assert out[0]["clause_type"] == "pets"
        assert out[0]["method"] == METHOD_CLASSIFIER
        assert out[0]["confidence"] == pytest.approx(0.8)

    def test_fallback_low_confidence_stays_unknown(self, monkeypatch):
        def fake_encode(texts, model_name, **kw):
            return np.zeros((len(texts), 3), dtype="float32")

        monkeypatch.setattr(analyze, "encodeTexts", fake_encode)
        stub = StubFallback(np.array([[0.3, 0.2, 0.1]]), ["term", "pets", "rent"])
        out = _call([_sec("Parties hereby execute this instrument.")], stub)
        assert out[0]["clause_type"] == UNKNOWN
        assert out[0]["method"] == METHOD_UNKNOWN
        assert out[0]["confidence"] == pytest.approx(0.3)

    def test_fallback_not_called_when_no_unknowns(self, monkeypatch):
        called = {"n": 0}

        def fake_encode(texts, model_name, **kw):
            called["n"] += 1
            return np.zeros((len(texts), 3), dtype="float32")

        monkeypatch.setattr(analyze, "encodeTexts", fake_encode)
        stub = StubFallback(np.zeros((0, 3)), ["term", "pets", "rent"])
        out = _call([_sec("The term of this lease is twelve months.")], stub)
        assert out[0]["method"] == METHOD_FAST_LANE
        assert called["n"] == 0

    def test_mixed_sections_batch_fallback(self, monkeypatch):
        def fake_encode(texts, model_name, **kw):
            return np.zeros((len(texts), 3), dtype="float32")

        monkeypatch.setattr(analyze, "encodeTexts", fake_encode)
        stub = StubFallback(
            np.array([[0.1, 0.6, 0.3], [0.5, 0.4, 0.1]]),
            ["term", "pets", "rent"],
        )
        out = _call(
            [
                _sec("The term of this lease is twelve months."),
                _sec("Parties execute this instrument."),
                _sec("Tenant pays a security deposit of Rs. 10,000."),
                _sec("Another unknown clause body."),
            ],
            stub,
        )
        assert [r["clause_type"] for r in out] == ["term", "pets", "deposit", "term"]
        assert [r["method"] for r in out] == [
            METHOD_FAST_LANE,
            METHOD_CLASSIFIER,
            METHOD_FAST_LANE,
            METHOD_CLASSIFIER,
        ]

    def test_raw_text_fallback_shape(self, monkeypatch):
        """Sections from ingest rows carry text in raw_text, not text."""

        def fake_encode(texts, model_name, **kw):
            return np.zeros((len(texts), 3), dtype="float32")

        monkeypatch.setattr(analyze, "encodeTexts", fake_encode)
        stub = StubFallback(np.array([[0.1, 0.9, 0.0]]), ["term", "pets", "rent"])
        out = analyzeSections(
            [{"source": "s", "text": "", "raw_text": "Execute this instrument."}],
            stub,
        )
        assert out[0]["clause_type"] == "pets"
        assert out[0]["method"] == METHOD_CLASSIFIER
