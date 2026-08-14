# Unit tests for dense retrieval over the content-hash index.
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from legalrag.retrieve import _mean_pool, _normalize, buildEmbeddings, queryEmbeddings


@pytest.fixture
def sections_index(tmp_path: Path) -> Path:
    p = tmp_path / "sections.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for text in ["subletting is not allowed", "rent shall be paid monthly", "pet policy clause here"]:
            fh.write(
                json.dumps(
                    {
                        "id": f"h_{text}",
                        "text": text,
                        "n_occurrences": 1,
                        "sources": ["doc_a"],
                    }
                )
                + "\n"
            )
    return p


def test_normalize_unit_vectors() -> None:
    v = _normalize(np.array([[3.0, 4.0], [0.0, 0.0]], dtype="float32"))
    assert np.allclose(np.linalg.norm(v[0]), 1.0)
    assert v[1, 0] == 0.0  # zero vector stays zero, no division blow-up


def test_mean_pool_shape() -> None:
    emb = np.random.rand(1, 5, 4).astype("float32")

    class FakeOutput:
        last_hidden_state = torch.from_numpy(emb)

    mask = torch.tensor([[1, 1, 1, 0, 0]])
    pooled = _mean_pool(FakeOutput(), mask)
    assert pooled.shape == (1, 4)
    assert np.allclose(pooled[0], emb[0, :3].mean(axis=0))


class TestBuildAndQuery:
    def test_build_writes_faiss_and_meta(
        self, sections_index: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "legalrag.retrieve.embedTexts",
            lambda texts, model_name, batch_size=16: np.ones((len(texts), 8), dtype="float32"),
        )
        out = tmp_path / "emb"
        meta = buildEmbeddings(sections_index, out, model_name="fake/model")
        assert meta["n_sections"] == 3
        assert meta["dim"] == 8
        assert (out / "sections.faiss").is_file()
        assert (out / "section_ids.jsonl").is_file()
        assert (out / "meta.json").is_file()

    def test_query_roundtrip(
        self, sections_index: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_embed(texts, model_name, batch_size=16) -> np.ndarray:
            v = np.zeros((len(texts), 3), dtype="float32")
            v[:, 0] = 1.0
            return v

        monkeypatch.setattr("legalrag.retrieve.embedTexts", fake_embed)
        out = tmp_path / "emb"
        buildEmbeddings(sections_index, out, model_name="fake/model")
        hits = queryEmbeddings("anything", out, model_name="fake/model", k=3)
        assert len(hits) == 3
        assert hits[0]["score"] == pytest.approx(1.0, abs=1e-4)
        assert "sources" in hits[0]
        assert hits[0]["rank"] == 1

    def test_query_k_bounds(
        self, sections_index: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "legalrag.retrieve.embedTexts",
            lambda texts, model_name, batch_size=16: np.ones((len(texts), 4), dtype="float32"),
        )
        out = tmp_path / "emb"
        buildEmbeddings(sections_index, out, model_name="fake/model")
        hits = queryEmbeddings("q", out, model_name="fake/model", k=1)
        assert len(hits) == 1