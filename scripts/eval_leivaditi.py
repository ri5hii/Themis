"""Evaluate the risk engine's triggers on the Leivaditi benchmark sentences.

The Leivaditi et al. 2020 benchmark (arXiv:2010.10386) labels red-flag
sentences in 179 real SEC EDGAR leases. The 738 sentences mapped into our
taxonomy live in data/annotated/leivaditi_redflags.jsonl. Each sentence
carries:
  - "type": our clause taxonomy type (classification is given, so this
    evaluates TRIGGERS, not the classifier)
  - "redflag_type": the Leivaditi red-flag annotation
  - "text": the sentence

This script reports per red-flag type: how many labeled sentences each of
our rules fires on (trigger recall on published drafting), plus the missed
sentences for trigger mining. Uses: progress.md §6.21.

Usage:
    python scripts/eval_leivaditi.py [--missed N] [--missed-for TYPE]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legalrag.risk import analyzeRisk
from legalrag.risk.rules import RULES

DATA = Path(__file__).resolve().parent.parent / "data" / "annotated" / "leivaditi_redflags.jsonl"


def load_sentences() -> list[dict]:
    sentences = []
    with open(DATA, encoding="utf-8") as fh:
        for line in fh:
            sentences.append(json.loads(line))
    return sentences


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--missed", type=int, default=0,
                        help="print up to N missed sentences per red-flag type")
    parser.add_argument("--missed-for", default="",
                        help="only print missed sentences for this red-flag type")
    args = parser.parse_args()

    sentences = load_sentences()

    per_type: dict[str, dict] = {}
    for s in sentences:
        rt = s["redflag_type"]
        if rt not in per_type:
            per_type[rt] = {"n": 0, "n_caught": 0, "fired": Counter(), "missed": []}
        per_type[rt]["n"] += 1

        result = analyzeRisk(
            [{"id": "s", "type": s["type"], "text": s["text"]}], RULES
        )
        if result.findings:
            per_type[rt]["n_caught"] += 1
            for f in result.findings:
                per_type[rt]["fired"][f.rule_id] += 1
        else:
            per_type[rt]["missed"].append(s)

    print(f"{'='*88}")
    print("LEIVADITI BENCHMARK: trigger recall on 738 labeled red-flag sentences")
    print(f"{'='*88}")
    print(f"{'redflag_type':<26} {'n':>4}  {'caught':>6}  {'recall':>6}  top rules")
    print("-" * 88)
    for rt, st in sorted(per_type.items(), key=lambda kv: -kv[1]["n"]):
        recall = st["n_caught"] / st["n"]
        top = ", ".join(f"{rid}x{c}" for rid, c in st["fired"].most_common(4))
        print(f"{rt:<26} {st['n']:>4}  {st['n_caught']:>6}  {recall:>6.2%}  {top}")

    total = sum(st["n"] for st in per_type.values())
    caught_total = sum(st["n_caught"] for st in per_type.values())
    fired_total = sum(sum(st["fired"].values()) for st in per_type.values())
    print(f"\ntotal: {caught_total}/{total} sentences caught ({caught_total/total:.2%}), "
          f"{fired_total} rule-fires (multiple rules can fire on one sentence)")

    # Missed sentences (for trigger mining from the published corpus)
    if args.missed > 0 or args.missed_for:
        print(f"\n{'='*88}")
        print("MISSED SENTENCES (published drafting not caught by current triggers)")
        print(f"{'='*88}")
        for rt, st in sorted(per_type.items(), key=lambda kv: -len(kv[1]["missed"])):
            if args.missed_for and rt != args.missed_for:
                continue
            if not st["missed"]:
                continue
            print(f"\n-- {rt} ({len(st['missed'])} missed of {st['n']}) --")
            for s in st["missed"][: args.missed if args.missed else 5]:
                print(f"  [{s['type']}] {s['text'][:180]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())