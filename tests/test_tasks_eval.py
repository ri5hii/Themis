# Unit tests for task splits and eval metrics (stdlib + numpy only).
from __future__ import annotations

import json
from pathlib import Path

from legalrag import tasks
from legalrag.eval import features, metrics


class TestSplitRows:
    def test_deterministic(self):
        rows = [{"source": "s", "text": f"t{i}", "type": "x"} for i in range(100)]
        a = tasks.splitRows(rows, seed=42)
        b = tasks.splitRows(rows, seed=42)
        assert a["train"] == b["train"]
        assert a["test"] == b["test"]

    def test_different_seed_different_order(self):
        rows = [{"source": "s", "text": f"t{i}", "type": "x"} for i in range(100)]
        a = tasks.splitRows(rows, seed=1)
        b = tasks.splitRows(rows, seed=2)
        assert [r["text"] for r in a["train"]] != [r["text"] for r in b["train"]]

    def test_ratios(self):
        rows = [{"source": "s", "text": f"t{i}", "type": "x"} for i in range(1000)]
        out = tasks.splitRows(rows)
        assert len(out["train"]) == 800
        assert len(out["val"]) == 100
        assert len(out["test"]) == 100

    def test_stratified_by_type(self):
        rows = [{"source": "s", "text": f"t{i}", "type": "a" if i % 2 else "b"} for i in range(1000)]
        out = tasks.splitRows(rows)
        for split in ("train", "val", "test"):
            types = [r["type"] for r in out[split]]
            assert "a" in types and "b" in types

    def test_loadJsonl(self, tmp_path: Path):
        p = tmp_path / "x.jsonl"
        p.write_text(json.dumps({"a": 1}) + "\n" + "\n" + json.dumps({"a": 2}) + "\n")
        assert tasks.loadJsonl(str(p)) == [{"a": 1}, {"a": 2}]


class TestMetrics:
    def test_multiclass_stats_balanced(self):
        y_true = ["a", "a", "b", "b"]
        y_pred = ["a", "b", "b", "b"]
        out = metrics.multiclassStats(y_true, y_pred)
        assert out["n"] == 4
        assert out["accuracy"] == 0.75
        assert out["classes"]["a"] == {"precision": 1.0, "recall": 0.5, "f1": 0.6667}
        assert out["classes"]["b"] == {"precision": 0.6667, "recall": 1.0, "f1": 0.8}
        assert out["macro"]["f1"] == 0.7333
        assert out["micro"] == {"precision": 0.75, "recall": 0.75, "f1": 0.75}

    def test_multiclass_empty(self):
        out = metrics.multiclassStats([], [])
        assert out["n"] == 0
        assert out["accuracy"] == 0.0

    def test_multiclass_unknown_pred_label(self):
        out = metrics.multiclassStats(["a"], ["z"])
        assert "z" in out["classes"]
        assert out["accuracy"] == 0.0

    def test_multilabel_stats(self):
        labels = ["obl", "ent"]
        y_true = [[1, 0], [1, 1], [0, 1]]
        y_pred = [[1, 0], [1, 0], [0, 1]]
        out = metrics.multilabelStats(y_true, y_pred, labels)
        assert out["n"] == 3
        assert out["exact_match_ratio"] == 0.6667
        assert out["labels"]["obl"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        assert out["labels"]["ent"]["recall"] == 0.5

    def test_multilabel_micro(self):
        labels = ["x"]
        out = metrics.multilabelStats([[1]], [[0]], labels)
        assert out["micro"]["precision"] == 0.0
        assert out["micro"]["recall"] == 0.0
        assert out["micro"]["f1"] == 0.0


class TestFeatures:
    def test_triggerVector_obl(self):
        assert features.triggerVector("Tenant shall pay rent.")[0] == 1

    def test_triggerVector_pro(self):
        vec = features.triggerVector("Tenant shall not assign.")
        assert vec[2] == 1  # pro

    def test_triggerVector_none_hit(self):
        assert features.triggerVector("This is a plain sentence.") == [0] * 6

    def test_triggerVector_case_insensitive(self):
        assert features.triggerVector("LANDLORD SHALL HAVE THE RIGHT TO terminate.")[1] == 1

    def test_deonticGroupCounts(self):
        counts = features.deonticGroupCounts(["shall pay", "plain", "may"])
        assert counts["obl"] == 1
        assert counts["per"] == 1

    def test_trigger_groups_stable(self):
        assert list(features.DEONTIC_TRIGGERS) == ["obl", "ent", "pro", "per", "nen", "nobl"]

    def test_partyVector_tenant(self):
        assert features.partyVector("tenant") == [1]

    def test_partyVector_landlord(self):
        assert features.partyVector("landlord") == [0]

    def test_partyVector_unknown(self):
        assert features.partyVector("") == [0]
        assert features.partyVector(None) == [0]

    def test_partyVector_case_insensitive(self):
        assert features.partyVector("Tenant") == [1]