# Unit tests for the deterministic synthetic lease-corpus generator.
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from build_synthetic import (
    DEONTIC_ORDER,
    DEONTIC_PER_CATEGORY,
    REDFLAG_COUNTS,
    SPAN_RARE,
    _render,
    generate_multilabel,
    generate_redflag,
    generate_span,
)
from ingest_synthetic import (
    MULTILABEL_LABELS,
    validate_multilabel,
    validate_redflag,
    validate_span,
)


def test_spec_counts() -> None:
    dm = generate_multilabel(42)
    rf = generate_redflag(42)
    sp = generate_span(42)
    assert len(dm) == len(DEONTIC_ORDER) * DEONTIC_PER_CATEGORY == 1750
    assert len(rf) == sum(REDFLAG_COUNTS.values()) == 390
    assert len(sp) == len(SPAN_RARE) * 8 + (len(REDFLAG_COUNTS) - len(SPAN_RARE) - 1) * 4 == 116


def test_spec_distributions() -> None:
    dm = generate_multilabel(42)
    rf = generate_redflag(42)
    sp = generate_span(42)

    by_cat = {c: 0 for c in DEONTIC_ORDER}
    for r in dm:
        by_cat[DEONTIC_ORDER[r["label"].index(1)]] += 1
    assert all(by_cat[c] == DEONTIC_PER_CATEGORY for c in DEONTIC_ORDER)

    by_rf = {c: 0 for c in REDFLAG_COUNTS}
    for r in rf:
        by_rf[r["type"]] += 1
    assert by_rf == REDFLAG_COUNTS

    by_sp = {c: 0 for c in REDFLAG_COUNTS if c != "none"}
    for r in sp:
        by_sp[r["type"]] += 1
    for c, n in by_sp.items():
        assert n == (8 if c in SPAN_RARE else 4)


def test_determinism_same_seed() -> None:
    a = generate_multilabel(42)
    b = generate_multilabel(42)
    assert a == b
    assert generate_redflag(42) == generate_redflag(42)
    assert generate_span(42) == generate_span(42)


def test_determinism_diff_seed_changes_rows() -> None:
    assert generate_multilabel(1) != generate_multilabel(2)


def test_party_near_balanced() -> None:
    dm = generate_multilabel(42)
    parties = [r["party"] for r in dm]
    tenant = parties.count("tenant")
    landlord = parties.count("landlord")
    assert tenant + landlord == len(dm)
    assert abs(tenant - landlord) <= 0.05 * len(dm)


def test_no_placeholders_left_in_output() -> None:
    for rows in (generate_multilabel(42), generate_redflag(42), generate_span(42)):
        for r in rows:
            for key in ("text", "raw_text"):
                if r.get(key):
                    assert "{" not in r[key] and "}" not in r[key], r[key]


def test_schema_validates() -> None:
    dm = generate_multilabel(42)
    rf = generate_redflag(42)
    sp = generate_span(42)

    validate_multilabel(dm)
    real_vocab = {r["type"] for r in rf if r["type"] != "holdover"}
    kept_rf = validate_redflag(rf, real_vocab)
    kept_sp = validate_span(sp, real_vocab)
    assert len(kept_rf) < len(rf)
    assert len(kept_sp) < len(sp)


def test_write_jsonl_roundtrip(tmp_path: Path) -> None:
    from build_synthetic import write_jsonl

    rows = generate_redflag(42)
    dst = tmp_path / "redflag_paragraph.jsonl"
    assert write_jsonl(rows, dst) == len(rows)
    loaded = [json.loads(l) for l in dst.read_text().splitlines()]
    assert loaded == rows


def test_render_slots_resolve() -> None:
    out = _render("{L} shall repair the roof.", 7)
    assert "{L}" not in out
    assert out.startswith("Redwood Industrial Trust")
    out2 = _render("{T} shall give {days} days' notice.", 3)
    assert "{days}" not in out2


def test_multilabel_label_order_matches_ingest() -> None:
    assert DEONTIC_ORDER == MULTILABEL_LABELS
