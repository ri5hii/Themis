# Unit tests for the trained clause classifier (TrainedClassifier).
from __future__ import annotations

from pathlib import Path

import numpy as np

from legalrag.extract.classifier import MARGIN, THRESHOLD, TrainedClassifier
from legalrag.extract.taxonomy import UNKNOWN


def _clf(threshold: float = THRESHOLD, margin: float = MARGIN) -> TrainedClassifier:
    # Two classes; input dim 3. Row [1,0,0] -> "term", [0,1,0] -> "rent".
    classes = ["term", "rent"]
    weights = np.array([[2.0, -2.0, 0.0], [-2.0, 2.0, 0.0]], dtype="float32")
    intercept = np.array([0.0, 0.0], dtype="float32")
    return TrainedClassifier("fake-model", classes, weights, intercept, threshold, margin)


def test_softmax_columns_sum_to_one() -> None:
    probs = _clf().predict_proba(np.array([[1.0, 0.0, 0.0]], dtype="float32"))
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_argmax_class() -> None:
    c = _clf()
    vecs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype="float32")
    assert c.predict(vecs) == ["term", "rent"]


def test_below_threshold_becomes_unknown() -> None:
    # Low-confidence input collapses to UNKNOWN: logits [0.2, -0.2] -> p≈0.6,
    # below the 0.9 threshold.
    classes = ["term", "rent"]
    weights = np.array([[0.2, -0.2, 0.0], [-0.2, 0.2, 0.0]], dtype="float32")
    intercept = np.array([0.0, 0.0], dtype="float32")
    c = TrainedClassifier("fake-model", classes, weights, intercept, threshold=0.9)
    vecs = np.array([[1.0, 0.0, 0.0]], dtype="float32")
    assert c.predict(vecs) == [UNKNOWN]


def test_narrow_margin_refused() -> None:
    # p≈[0.525, 0.475]: above the 0.4 threshold but the top-two gap (0.05) is
    # below the 0.2 margin, so it collapses to UNKNOWN (OOD refuse path).
    classes = ["term", "rent"]
    weights = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype="float32")
    intercept = np.array([0.0, 0.0], dtype="float32")
    c = TrainedClassifier("fake-model", classes, weights, intercept, margin=0.2)
    vecs = np.array([[0.55, 0.45, 0.0]], dtype="float32")
    assert c.predict(vecs) == [UNKNOWN]

    wide = np.array([[0.9, 0.1, 0.0]], dtype="float32")
    assert c.predict(wide) == ["term"]


def test_save_load_roundtrip(tmp_path: Path) -> None:
    c = _clf()
    p = tmp_path / "classifier.npz"
    c.save(p)
    loaded = TrainedClassifier.load(p)
    assert loaded.classes == c.classes
    assert loaded.threshold == c.threshold
    assert loaded.margin == c.margin
    assert np.allclose(loaded.weights, c.weights)
    vecs = np.array([[0.0, 1.0, 0.0]], dtype="float32")
    assert loaded.predict(vecs) == ["rent"]


def test_load_defaults_margin_for_old_npz(tmp_path: Path) -> None:
    # npz written before the margin field existed must still load.
    c = _clf()
    p = tmp_path / "old.npz"
    c.save(p)
    np.savez(p, weights=c.weights, intercept=c.intercept, classes=c.classes, model_name=c.model_name, threshold=c.threshold)
    loaded = TrainedClassifier.load(p)
    assert loaded.margin == MARGIN


def test_from_sklearn_wraps_coefs() -> None:
    from sklearn.linear_model import LogisticRegression

    # Well-separated 3-class multinomial data (matches the real n-class head).
    X = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.1, 0.0],
            [1.0, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.1],
            [0.0, 1.0, 0.2],
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.2, 0.0, 1.0],
        ]
    )
    y = ["term"] * 3 + ["rent"] * 3 + ["deposit"] * 3
    clf = LogisticRegression(max_iter=1000, C=100.0).fit(X, y)
    c = TrainedClassifier.from_sklearn("fake", clf, list(clf.classes_))
    assert c.predict(X[:3]) == ["term", "term", "term"]
    assert c.predict(X[3:6]) == ["rent", "rent", "rent"]
    assert c.predict(X[6:]) == ["deposit", "deposit", "deposit"]


def test_threshold_contract() -> None:
    # Refusal must be reachable: threshold in (0, 0.5) keeps the refusal
    # path live while leaving room below for low-confidence predictions.
    assert 0.0 < THRESHOLD < 0.5


def test_margin_contract() -> None:
    # Refuse margin: non-negative, must not exceed the threshold (a margin
    # above threshold would veto every accepted prediction).
    assert 0.0 <= MARGIN <= THRESHOLD
