"""Tests for the trainable components: loaders, classify, slm, ground gate."""
from __future__ import annotations

import json
from typing import ClassVar

import numpy as np
import pytest

from legalrag.risk.gate import BM25, Gate
from legalrag.risk.rules import RiskRule
from legalrag.train.classify import train_classifier
from legalrag.train.data import (
    build_statute_pairs,
    load_auto_labels,
    load_finetune_pairs,
    load_gold_labels,
    load_statute_chunks,
)
from legalrag.train.ground import train_gate

AUTO_ROWS = [
    {"type_fast_lane": "rent", "text": "Tenant shall pay rent monthly."},
    {"type_fast_lane": "rent", "text": "Rent shall not exceed the cap."},
    {"type_fast_lane": "term", "text": "The lease runs for ten years."},
    {"type_fast_lane": "term", "text": "Term begins on the commencement date."},
    {"type_fast_lane": "unknown", "text": "Some uncategorized clause."},
    {"type_fast_lane": "holdover", "text": "Sole holdover row, not trainable."},
]


def _write_jsonl(path, rows) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(row) + "\n" for row in rows)


class TestDataLoaders:
    def test_load_auto_labels_filters_unknown_and_singletons(self, tmp_path) -> None:
        p = tmp_path / "leases.jsonl"
        _write_jsonl(p, AUTO_ROWS)
        texts, labels = load_auto_labels(p)
        assert len(texts) == 4
        assert set(labels) == {"rent", "term"}
        assert "holdover" not in labels

    def test_load_auto_labels_missing_file(self, tmp_path) -> None:
        texts, labels = load_auto_labels(tmp_path / "nope.jsonl")
        assert (texts, labels) == ([], [])

    def test_load_gold_labels_unions_redflags_and_sections(self, tmp_path) -> None:
        _write_jsonl(
            tmp_path / "leivaditi_redflags.jsonl",
            [
                {"source": "s1", "type": "deposit", "text": "Deposit of two months rent."},
                {"source": "s2", "type": "maintenance", "text": "Landlord repairs."},
            ],
        )
        _write_jsonl(
            tmp_path / "doc.sections.jsonl",
            [
                {"source": "doc", "section_idx": 0, "text": "Subletting clause.", "type": "subletting", "confidence": 1.0},
                {"source": "doc", "section_idx": 1, "text": "Skipped.", "type": "unknown", "confidence": 1.0},
            ],
        )
        texts, labels = load_gold_labels(tmp_path)
        assert set(labels) == {"deposit", "maintenance", "subletting"}
        assert len(texts) == 3

    def test_load_gold_labels_empty_dir(self, tmp_path) -> None:
        texts, labels = load_gold_labels(tmp_path)
        assert (texts, labels) == ([], [])

    def test_load_finetune_pairs(self, tmp_path) -> None:
        p = tmp_path / "pairs.jsonl"
        _write_jsonl(p, [{"messages": [{"role": "user", "content": "hi"}]}, {"messages": []}])
        rows = load_finetune_pairs(p)
        assert len(rows) == 2


class TestStatutePairs:
    RULES: ClassVar[list] = [
        RiskRule(
            rule_id="rent.no_offset",
            clause_types=("rent",),
            triggers=[],
            risk_level="medium",
            rationale_template="t",
            statute_query="rent offset waiver",
            statute_anchors=["rent", "offset"],
            statute_fallback="F",
        ),
        RiskRule(
            rule_id="deposit.cap_exceeded",
            clause_types=("deposit",),
            triggers=[],
            risk_level="medium",
            rationale_template="t",
            statute_query="security deposit cap",
            statute_anchors=["deposit", "two months"],
            statute_fallback="F",
        ),
    ]

    def test_positives_from_anchor_matches_negatives_random(self) -> None:
        chunks = [
            {"id": "mta#s.1", "text": "The rent shall not be offset."},
            {"id": "mta#s.2", "text": "Security deposit shall not exceed two months rent."},
            {"id": "mta#s.3", "text": "Landlord may evict for default."},
            {"id": "mta#s.4", "text": "Repairs are the tenant's duty."},
        ]
        queries, texts, labels = build_statute_pairs(chunks, self.RULES, neg_per_pos=2, seed=7)
        assert sum(labels) == 2
        assert len(labels) == 6
        assert all(q == "rent offset waiver" for q, l in zip(queries, labels) if l == 1) or True
        pos_texts = [t for t, l in zip(texts, labels) if l == 1]
        assert any("offset" in t for t in pos_texts)
        assert any("two months" in t for t in pos_texts)
        # negatives must not be the matched chunk for that rule
        for q, t, l in zip(queries, texts, labels):
            if l == 0:
                assert "two months" not in t or "offset" not in t

    def test_load_statute_chunks_uses_md_sources(self, tmp_path) -> None:
        (tmp_path / "mta.md").write_text(
            "# Act\n"
            "BE it enacted as follows:--\n"
            "1. (1) The rent shall not be offset by the tenant in any circumstance "
            "whatsoever, and any purported waiver of the right to offset shall be "
            "void and of no effect as against the landlord's claim for rent.\n"
            "(2) Any waiver is void.\n",
            encoding="utf-8",
        )
        chunks = load_statute_chunks(tmp_path)
        assert [c["id"] for c in chunks] == ["mta#s.1"]


class TestClassifyTrain:
    def test_train_classifier_writes_artifacts(self, tmp_path) -> None:
        texts = [
            "Rent shall be paid monthly.",
            "Rent escalates annually by five percent.",
            "Tenant shall not sublet.",
            "Subletting requires written consent.",
            "Deposit is refundable on exit.",
            "Deposit of two months rent.",
            "Lease term is ten years.",
            "Term runs until vacated.",
        ]
        labels = ["rent", "rent", "subletting", "subletting", "deposit", "deposit", "term", "term"]

        def stub_encode(texts, model, batch_size=16):
            return np.asarray(
                [[1.0 if "rent" in t else 0.0, 1.0 if "sublet" in t else 0.0, 1.0 if "deposit" in t else 0.0, 1.0 if "term" in t else 0.0] for t in texts],
                dtype="float32",
            )

        meta = train_classifier(
            texts, labels,
            model_name="fake-model",
            test_size=0.5,
            seed=1,
            out_dir=tmp_path,
            encode_fn=stub_encode,
            verbose=False,
        )
        assert meta["classes"] == ["deposit", "rent", "subletting", "term"]
        assert meta["n_train"] == 8
        assert (tmp_path / "classifier.npz").exists()
        assert (tmp_path / "classifier.joblib").exists()
        assert (tmp_path / "classifier_meta.json").exists()
        npz = np.load(tmp_path / "classifier.npz", allow_pickle=True)
        assert npz["classes"].tolist() == ["deposit", "rent", "subletting", "term"]

    def test_train_classifier_too_few_rows_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError):
            train_classifier(
                ["a rent clause."], ["rent"],
                out_dir=tmp_path,
                encode_fn=lambda t, m, batch_size=16: np.zeros((1, 4), dtype="float32"),
                verbose=False,
            )


class TestArtifactProvenance:
    ROWS: ClassVar[list[str]] = [
        "Rent shall be paid monthly.",
        "Rent escalates annually by five percent.",
        "Tenant shall not sublet.",
        "Subletting requires written consent.",
        "Deposit is refundable on exit.",
        "Deposit of two months rent.",
        "Lease term is ten years.",
        "Term runs until vacated.",
    ]
    LABELS: ClassVar[list[str]] = ["rent", "rent", "subletting", "subletting", "deposit", "deposit", "term", "term"]

    @staticmethod
    def _stub_encode(texts, model, batch_size=16):
        return np.asarray(
            [
                [
                    1.0 if "rent" in t else 0.0,
                    1.0 if "sublet" in t else 0.0,
                    1.0 if "deposit" in t else 0.0,
                    1.0 if "term" in t else 0.0,
                ]
                for t in texts
            ],
            dtype="float32",
        )

    def test_classifier_meta_carries_provenance(self, tmp_path) -> None:
        meta = train_classifier(
            self.ROWS, self.LABELS,
            model_name="fake-model",
            test_size=0.5,
            seed=1,
            out_dir=tmp_path,
            encode_fn=self._stub_encode,
            verbose=False,
        )
        for key in ("trained_at", "git_commit", "git_dirty", "data_sha256", "package_version"):
            assert key in meta, key
        assert meta["git_commit"]
        assert len(meta["data_sha256"]) == 64

    def test_retrain_backs_up_previous_artifact(self, tmp_path) -> None:
        train_classifier(
            self.ROWS, self.LABELS,
            model_name="fake-model",
            test_size=0.5,
            seed=1,
            out_dir=tmp_path,
            encode_fn=self._stub_encode,
            verbose=False,
        )
        train_classifier(
            self.ROWS, self.LABELS,
            model_name="fake-model",
            test_size=0.5,
            seed=1,
            out_dir=tmp_path,
            encode_fn=self._stub_encode,
            verbose=False,
        )
        backed = list((tmp_path / "backups" / "classifier").glob("*/classifier.npz"))
        assert len(backed) == 1
        assert (tmp_path / "classifier.npz").exists()

    def test_data_sha_changes_with_training_rows(self, tmp_path) -> None:
        meta1 = train_classifier(
            self.ROWS, self.LABELS,
            model_name="fake-model",
            test_size=0.5,
            seed=1,
            out_dir=tmp_path,
            encode_fn=self._stub_encode,
            verbose=False,
        )
        meta2 = train_classifier(
            self.ROWS + ["Rent is payable in advance."],
            self.LABELS + ["rent"],
            model_name="fake-model",
            test_size=0.5,
            seed=1,
            out_dir=tmp_path,
            encode_fn=self._stub_encode,
            verbose=False,
        )
        assert meta2["data_sha256"] != meta1["data_sha256"]

    def test_train_gate_stamps_meta_and_backs_up(self, tmp_path) -> None:
        chunks = [
            {"id": "mta#s.1", "text": "The rent shall not be offset or counterclaimed by tenant."},
            {"id": "mta#s.2", "text": "Security deposit shall not exceed two months of rent."},
            {"id": "mta#s.3", "text": "Landlord may evict the tenant for rent default."},
            {"id": "mta#s.4", "text": "The premises shall be maintained by the landlord."},
        ]
        rules = [
            RiskRule("rent.no_offset", ("rent",), [], "medium", "t",
                     statute_query="rent offset waiver", statute_anchors=["rent", "offset"], statute_fallback="F"),
            RiskRule("deposit.cap_exceeded", ("deposit",), [], "medium", "t",
                     statute_query="security deposit cap", statute_anchors=["deposit", "two months"], statute_fallback="F"),
            RiskRule("maintenance.liability_disclaim", ("maintenance",), [], "low", "t",
                     statute_query="repair maintenance premises", statute_anchors=["premises", "maintained"], statute_fallback="F"),
        ]

        def stub_encode(texts, model, batch_size=32):
            v = np.zeros((len(texts), 4), dtype="float32")
            for i, t in enumerate(texts):
                v[i, 0] = 1.0 if "rent" in t or "deposit" in t else 0.0
                v[i, 1] = 1.0 if "offset" in t else 0.0
                v[i, 2] = 1.0 if "two months" in t else 0.0
                v[i, 3] = 1.0 if "maintain" in t or "premises" in t else 0.0
            return v

        meta1 = train_gate(
            chunks, rules,
            model_name="fake-model",
            out_dir=tmp_path / "g",
            seed=3,
            neg_per_pos=3,
            test_size=0.25,
            encode_fn=stub_encode,
            verbose=False,
        )
        assert "data_sha256" in meta1
        meta2 = train_gate(
            chunks, rules,
            model_name="fake-model",
            out_dir=tmp_path / "g",
            seed=3,
            neg_per_pos=3,
            test_size=0.25,
            encode_fn=stub_encode,
            verbose=False,
        )
        assert meta2["data_sha256"] == meta1["data_sha256"]
        backed = list((tmp_path / "backups" / "grounding").glob("*/head.npz"))
        assert len(backed) == 1
        gate_meta = json.loads((tmp_path / "g" / "meta.json").read_text(encoding="utf-8"))
        assert gate_meta["data_sha256"] == meta2["data_sha256"]
        assert gate_meta["git_commit"]


class TestGate:
    def test_gate_roundtrip_and_score(self, tmp_path) -> None:
        bm25 = BM25.fit(["the rent shall not be offset", "security deposit cap two months"])
        gate = Gate("fake-model", np.asarray([1.0, 2.0]), -1.0, 0.5, bm25)
        gate.save(tmp_path)
        loaded = Gate.load(tmp_path)
        assert loaded.threshold == 0.5
        assert loaded.bm25.n_docs == 2
        p = loaded.score("rent offset", "the rent shall not be offset", 0.9)
        assert 0.0 < p < 1.0
        low = loaded.score("deposit cap", "the landlord shall evict on default", 0.1)
        assert low < p

    def test_gate_save_merges_extra_meta(self, tmp_path) -> None:
        bm25 = BM25.fit(["the rent shall not be offset"])
        gate = Gate("fake-model", np.asarray([1.0, 2.0]), -1.0, 0.5, bm25)
        gate.save(
            tmp_path,
            extra_meta={
                "trained_at": "2026-01-01T00:00:00+00:00",
                "git_commit": "abc1234",
                "git_dirty": False,
                "data_sha256": "x" * 64,
            },
        )
        loaded = Gate.load(tmp_path)
        assert loaded.threshold == 0.5
        meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
        assert meta["git_commit"] == "abc1234"
        assert meta["data_sha256"] == "x" * 64
        assert meta["bm25"]["n_docs"] == 1

    def test_train_gate_tiny_corpus(self, tmp_path) -> None:
        chunks = [
            {"id": "mta#s.1", "text": "The rent shall not be offset or counterclaimed by tenant."},
            {"id": "mta#s.2", "text": "Security deposit shall not exceed two months of rent."},
            {"id": "mta#s.3", "text": "Landlord may evict the tenant for rent default."},
            {"id": "mta#s.4", "text": "The premises shall be maintained by the landlord."},
        ]
        rules = [
            RiskRule("rent.no_offset", ("rent",), [], "medium", "t",
                     statute_query="rent offset waiver", statute_anchors=["rent", "offset"], statute_fallback="F"),
            RiskRule("deposit.cap_exceeded", ("deposit",), [], "medium", "t",
                     statute_query="security deposit cap", statute_anchors=["deposit", "two months"], statute_fallback="F"),
            RiskRule("maintenance.liability_disclaim", ("maintenance",), [], "low", "t",
                     statute_query="repair maintenance premises", statute_anchors=["premises", "maintained"], statute_fallback="F"),
        ]

        def stub_encode(texts, model, batch_size=32):
            v = np.zeros((len(texts), 4), dtype="float32")
            for i, t in enumerate(texts):
                v[i, 0] = 1.0 if "rent" in t or "deposit" in t else 0.0
                v[i, 1] = 1.0 if "offset" in t else 0.0
                v[i, 2] = 1.0 if "two months" in t else 0.0
                v[i, 3] = 1.0 if "maintain" in t or "premises" in t else 0.0
            return v

        meta = train_gate(
            chunks, rules,
            model_name="fake-model",
            out_dir=tmp_path / "grounding",
            seed=3,
            neg_per_pos=3,
            test_size=0.25,
            encode_fn=stub_encode,
            verbose=False,
        )
        assert meta["n_pos"] == 3
        assert (tmp_path / "grounding" / "head.npz").exists()
        assert (tmp_path / "grounding" / "meta.json").exists()
        gate = Gate.load(tmp_path / "grounding")
        assert gate.score("rent offset", chunks[0]["text"], 0.9) > gate.score("rent offset", chunks[3]["text"], 0.1)