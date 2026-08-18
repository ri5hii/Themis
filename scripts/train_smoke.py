"""Train smoke: run each trainer on a tiny subset, assert artifacts load.

Guarded: skips (exit 0) when the embedding model is not in the local Hugging
Face cache, so verify_repo does not download models. Trains into a temp dir
and never touches the shipped artifacts in models/.

Usage:
    python scripts/train_smoke.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

ROOT = Path(__file__).resolve().parent.parent

MODEL = "nlpaueb/legal-bert-base-uncased"


def _model_cached(model: str) -> bool:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    if not hub.is_dir():
        return False
    slug = "models--" + model.replace("/", "--")
    return (hub / slug).exists()


def main() -> int:
    from legalrag.risk.rules import RULES
    from legalrag.train.data import load_auto_labels, load_statute_chunks
    from legalrag.train.ground import train_gate

    if not _model_cached(MODEL):
        print(f"[train-smoke] skipping: {MODEL} not in HF cache")
        return 0


    from legalrag.extract.classifier import TrainedClassifier
    from legalrag.risk.gate import Gate
    from legalrag.train.classify import train_classifier

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        auto = ROOT / "data" / "annotated" / "leivaditi_leases.jsonl"
        texts, labels = load_auto_labels(auto)
        texts = texts[:300]
        labels = labels[:300]
        meta = train_classifier(
            texts, labels, model_name=MODEL, test_size=0.2, seed=42,
            out_dir=tmp, verbose=False,
        )
        tc = TrainedClassifier.load(tmp / "classifier.npz")
        assert set(tc.classes) <= set(labels) and len(tc.classes) >= 2
        print(f"[train-smoke] classify ok: {len(texts)} rows, {meta['test_accuracy']:.3f} acc")

        chunks = load_statute_chunks(ROOT / "data" / "statutes")
        gmeta = train_gate(
            chunks, RULES, model_name=MODEL, out_dir=tmp / "grounding",
            seed=42, neg_per_pos=2, test_size=0.2, verbose=False,
        )
        gate = Gate.load(tmp / "grounding")
        assert gate.threshold == 0.5
        score = gate.score("rent offset", chunks[0]["text"], 0.8)
        assert 0.0 < score < 1.0
        print(f"[train-smoke] ground ok: {gmeta['n_pairs']} pairs, {gmeta['test_accuracy']:.3f} acc")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())