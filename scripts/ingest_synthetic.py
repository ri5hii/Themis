"""Validate synthetic data and build deterministic train/val/test splits.

Usage:
    python scripts/ingest_synthetic.py [--seed 42] [--out data/synthetic/splits]

Reads data/synthetic/{deontic_multilabel,redflag_paragraph,deontic_span}.jsonl,
validates each row against the real engine schema/vocabulary, normalizes
deontic_multilabel rows to the full engine schema (adds spans/split keys),
and writes stratified per-class train/val/test splits consumed by
scripts/train_eval.py via --splits data/synthetic/splits.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legalrag import tasks

ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC = ROOT / "data/synthetic"
REAL_SPLITS = ROOT / "data/splits"
DEFAULT_OUT = SYNTHETIC / "splits"

MULTILABEL_LABELS = ["obl", "ent", "pro", "per", "oth", "nen", "none"]


def _load(path: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  [error] {path.name}:{i}: bad JSON: {e}")
            sys.exit(1)
    return rows


def _real_vocab(name: str) -> set[str]:
    vocab = set()
    for split in ("train", "val", "test"):
        p = REAL_SPLITS / f"{name}.{split}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            vocab.add(json.loads(line)["type"])
    return vocab


def _check_sources(rows: list[dict], label: str) -> None:
    sources = [r.get("source") for r in rows]
    dup = sorted({s for s in sources if sources.count(s) > 1})
    if dup:
        print(f"  [error] {label}: duplicate sources: {dup[:5]}")
        sys.exit(1)
    if any(s is None for s in sources):
        print(f"  [error] {label}: row missing 'source'")
        sys.exit(1)


def validate_redflag(rows: list[dict], vocab: set[str]) -> list[dict]:
    bad_types = sorted({r["type"] for r in rows if r["type"] not in vocab})
    if bad_types:
        print(f"  [warn] redflag_paragraph: types outside task vocab, dropping: {bad_types}")
    keep = [r for r in rows if r["type"] in vocab]
    dropped = len(rows) - len(keep)
    for r in keep:
        if not r.get("raw_text"):
            print(f"  [error] redflag_paragraph: row {r.get('source')} missing raw_text")
            sys.exit(1)
    _check_sources(rows, "redflag_paragraph")
    if dropped:
        print(f"  [warn] redflag_paragraph: dropped {dropped} rows outside task vocab")
    return keep


def validate_multilabel(rows: list[dict]) -> None:
    for r in rows:
        lab = r.get("label")
        if not isinstance(lab, list) or len(lab) != len(MULTILABEL_LABELS):
            print(f"  [error] deontic_multilabel: row {r.get('source')}: bad label dim {lab}")
            sys.exit(1)
        if sorted(set(lab)) not in ([0, 1], [1]):
            print(f"  [error] deontic_multilabel: row {r.get('source')}: non-binary label {lab}")
            sys.exit(1)
        if lab.count(1) != 1:
            print(f"  [error] deontic_multilabel: row {r.get('source')}: expected exactly one label (got {lab.count(1)})")
            sys.exit(1)
        if r.get("party") not in ("tenant", "landlord"):
            print(f"  [error] deontic_multilabel: row {r.get('source')}: bad party {r.get('party')!r}")
            sys.exit(1)
        types = r.get("deontic_types", [])
        active = MULTILABEL_LABELS[lab.index(1)]
        if active == "none":
            if types not in ([], ["none"]):
                print(f"  [error] deontic_multilabel: row {r.get('source')}: 'none' label with types {types}")
                sys.exit(1)
        elif types != [active]:
            print(f"  [error] deontic_multilabel: row {r.get('source')}: deontic_types {types} != [{active}]")
            sys.exit(1)
    _check_sources(rows, "deontic_multilabel")


def validate_span(rows: list[dict], vocab: set[str]) -> list[dict]:
    bad_types = sorted({r["type"] for r in rows if r["type"] not in vocab})
    if bad_types:
        print(f"  [warn] deontic_span: types outside task vocab, dropping: {bad_types}")
    keep = [r for r in rows if r["type"] in vocab]
    dropped = len(rows) - len(keep)
    for r in keep:
        if not r.get("raw_text"):
            print(f"  [error] deontic_span: row {r.get('source')} missing raw_text")
            sys.exit(1)
        if r.get("part") is None:
            print(f"  [error] deontic_span: row {r.get('source')} missing part")
            sys.exit(1)
    _check_sources(rows, "deontic_span")
    if dropped:
        print(f"  [warn] deontic_span: dropped {dropped} rows outside task vocab")
    return keep


def normalize_multilabel(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        norm = dict(r)
        norm.setdefault("spans", {})
        norm.setdefault("split", "synthetic")
        out.append(norm)
    return out


def _stratified_split(rows: list[dict], key, seed: int, fracs=(0.7, 0.15, 0.15)):
    import random

    rng = random.Random(seed)
    groups: dict[object, list[dict]] = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    train, val, test = [], [], []
    for g in groups.values():
        rng.shuffle(g)
        n1 = max(1, int(len(g) * fracs[0]))
        n2 = n1 + max(1, int(len(g) * fracs[1]))
        train.extend(g[:n1])
        val.extend(g[n1:n2])
        test.extend(g[n2:])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _write_split(out: Path, name: str, rows: list[dict]) -> None:
    dst = out / f"{name}.jsonl"
    dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=tasks.DEFAULT_SEED)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("[ingest_synthetic] validating against real schema/vocab...")

    rp = _load(SYNTHETIC / "redflag_paragraph.jsonl")
    rp = validate_redflag(rp, _real_vocab("redflag_paragraph"))
    print(f"  redflag_paragraph: {len(rp)} rows OK")

    dm = _load(SYNTHETIC / "deontic_multilabel.jsonl")
    validate_multilabel(dm)
    dm = normalize_multilabel(dm)
    print(f"  deontic_multilabel: {len(dm)} rows OK")

    ds = _load(SYNTHETIC / "deontic_span.jsonl")
    ds = validate_span(ds, _real_vocab("deontic_span"))
    print(f"  deontic_span: {len(ds)} rows OK")

    tr, va, te = _stratified_split(rp, lambda r: r["type"], args.seed)
    _write_split(out, "redflag_paragraph.train", tr)
    _write_split(out, "redflag_paragraph.val", va)
    _write_split(out, "redflag_paragraph.test", te)
    print(f"  redflag_paragraph splits: train={len(tr)} val={len(va)} test={len(te)}")

    tr, va, te = _stratified_split(dm, lambda r: MULTILABEL_LABELS[r["label"].index(1)], args.seed)
    _write_split(out, "deontic_multilabel.train", tr)
    _write_split(out, "deontic_multilabel.val", va)
    _write_split(out, "deontic_multilabel.test", te)
    print(f"  deontic_multilabel splits: train={len(tr)} val={len(va)} test={len(te)}")

    print(f"[ingest_synthetic] splits written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())