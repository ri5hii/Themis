"""Three-way comparison: Claude vs human vs Themis on 9 lease PDFs.

Loads:
  - claude_inference.json  (Claude's blind reading)
  - my_inference.json      (author's blind reading)
  - themis_out.json        (Themis extract engine)

Computes:
  1. Set-level: unique clause types per doc, Jaccard / P / R between pairs
  2. Section-level: matched sections by heading similarity, type agreement
  3. Aggregate metrics across all docs
"""
from __future__ import annotations

import json
import re
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval" / "claude_test_docs"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _extract_types(doc: dict) -> set[str]:
    """Flatten all clause types from per_section into a set."""
    types: set[str] = set()
    for section_types in doc.get("per_section", {}).values():
        types.update(section_types)
    return types


def _normalize_heading(h: str) -> str:
    """Strip article/section numbers and normalize for fuzzy matching."""
    h = h.lower().strip()
    h = re.sub(r"^(art\w*\.?\s*\d+[\.\d]*\s*|section\s*\d+[\.\d]*\s*|\d+[\.\d]*\s*)", "", h)
    h = re.sub(r"[\s\-_]+", " ", h).strip()
    return h


def _heading_similarity(a: str, b: str) -> float:
    """Character-level Jaccard on normalized headings."""
    na, nb = _normalize_heading(a), _normalize_heading(b)
    if not na or not nb:
        return 0.0
    sa, sb = set(na), set(nb)
    return len(sa & sb) / max(len(sa | sb), 1)


def _set_metrics(
    gold: set[str], pred: set[str]
) -> dict[str, float]:
    """Precision, recall, F1, Jaccard between two type sets."""
    intersection = gold & pred
    jaccard = len(intersection) / max(len(gold | pred), 1)
    precision = len(intersection) / max(len(pred), 1)
    recall = len(intersection) / max(len(gold), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "jaccard": round(jaccard, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_intersection": len(intersection),
        "n_gold": len(gold),
        "n_pred": len(pred),
        "missed": sorted(gold - pred),
        "extra": sorted(pred - gold),
    }


def set_level_comparison(
    claude_docs: dict, human_docs: dict, themis_docs: dict
) -> dict:
    """Per-doc and aggregate set-level comparison."""
    doc_keys = sorted(set(claude_docs) & set(human_docs) & set(themis_docs))
    pairs = [("claude_vs_human", "claude", "human"),
             ("claude_vs_themis", "claude", "themis"),
             ("human_vs_themis", "human", "themis")]
    results: dict = {"per_doc": {}, "aggregate": {}}

    agg: dict[str, list[dict]] = {p[0]: [] for p in pairs}

    for doc_key in doc_keys:
        c_types = _extract_types(claude_docs[doc_key])
        h_types = _extract_types(human_docs[doc_key])
        t_sections = themis_docs[doc_key].get("sections", [])
        t_types = {s["clause_type"] for s in t_sections if s["clause_type"] != "unknown"}
        all_map = {"claude": c_types, "human": h_types, "themis": t_types}

        doc_result: dict = {}
        for pair_name, a_key, b_key in pairs:
            metrics = _set_metrics(all_map[a_key], all_map[b_key])
            doc_result[pair_name] = metrics
            agg[pair_name].append(metrics)

        results["per_doc"][doc_key] = doc_result

    for pair_name, metrics_list in agg.items():
        n = len(metrics_list)
        results["aggregate"][pair_name] = {
            "mean_jaccard": round(sum(m["jaccard"] for m in metrics_list) / n, 4),
            "mean_precision": round(sum(m["precision"] for m in metrics_list) / n, 4),
            "mean_recall": round(sum(m["recall"] for m in metrics_list) / n, 4),
            "mean_f1": round(sum(m["f1"] for m in metrics_list) / n, 4),
            "n_docs": n,
        }
    return results


def section_level_comparison(
    claude_docs: dict, human_docs: dict, themis_docs: dict
) -> dict:
    """Match sections by heading/text similarity, compare type sets."""
    doc_keys = sorted(set(claude_docs) & set(human_docs) & set(themis_docs))
    results: dict = {"per_doc": {}, "aggregate": {"matched": 0, "type_agree": 0}}

    for doc_key in doc_keys:
        c_sections = claude_docs[doc_key].get("per_section", {})
        h_sections = human_docs[doc_key].get("per_section", {})
        t_sections = themis_docs[doc_key].get("sections", [])

        # Match Claude <-> Human by heading similarity
        c_headings = list(c_sections.keys())
        h_headings = list(h_sections.keys())
        matched_pairs: list[dict] = []

        for ch in c_headings:
            best_hh, best_sim = "", 0.0
            for hh in h_headings:
                sim = _heading_similarity(ch, hh)
                if sim > best_sim:
                    best_sim, best_hh = sim, hh
            if best_sim >= 0.3:
                c_types = set(c_sections[ch])
                h_types = set(h_sections[best_hh])
                agree = c_types == h_types
                matched_pairs.append({
                    "claude_heading": ch,
                    "human_heading": best_hh,
                    "similarity": round(best_sim, 3),
                    "claude_types": sorted(c_types),
                    "human_types": sorted(h_types),
                    "agree": agree,
                })

        # Match Themis sections to Claude headings by text containment
        themis_matches: list[dict] = []
        for ts in t_sections:
            text = ts["text"]
            best_ch, best_overlap = "", 0.0
            for ch in c_headings:
                # Check if key words from heading appear in text
                heading_words = _normalize_heading(ch).split()
                if not heading_words:
                    continue
                overlap = sum(1 for w in heading_words if w in text.lower()) / len(heading_words)
                if overlap > best_overlap:
                    best_overlap, best_ch = overlap, ch
            if best_overlap >= 0.3:
                c_types = set(c_sections[best_ch])
                t_type = ts["clause_type"]
                themis_matches.append({
                    "themis_text_start": text[:80] + "..." if len(text) > 80 else text,
                    "matched_claude_heading": best_ch,
                    "overlap": round(best_overlap, 3),
                    "claude_types": sorted(c_types),
                    "themis_type": t_type,
                    "agree": t_type in c_types,
                })

        agree_count = sum(1 for p in matched_pairs if p["agree"])
        results["per_doc"][doc_key] = {
            "claude_human_matches": matched_pairs,
            "claude_human_agreement": round(agree_count / max(len(matched_pairs), 1), 4),
            "n_matched": len(matched_pairs),
            "themis_matches": themis_matches,
            "n_themis_matched": len(themis_matches),
            "themis_agreement": round(
                sum(1 for m in themis_matches if m["agree"]) / max(len(themis_matches), 1), 4
            ),
        }
        results["aggregate"]["matched"] += len(matched_pairs)
        results["aggregate"]["type_agree"] += agree_count

    total = results["aggregate"]["matched"]
    results["aggregate"]["overall_agreement"] = round(
        results["aggregate"]["type_agree"] / max(total, 1), 4
    )
    return results


def risk_flag_comparison(
    claude_docs: dict, human_docs: dict
) -> dict:
    """Compare risk flags between Claude and the author."""
    doc_keys = sorted(set(claude_docs) & set(human_docs))
    results: dict = {"per_doc": {}, "aggregate": {}}
    total_c, total_h, total_overlap = 0, 0, 0

    for doc_key in doc_keys:
        c_flags = set(claude_docs[doc_key].get("risk_flags", []))
        h_flags = set(human_docs[doc_key].get("risk_flags", []))
        # Keyword overlap: count flags sharing ≥3 significant words
        c_words = [{w for w in re.split(r"\W+", f.lower()) if len(w) > 3} for f in c_flags]
        h_words = [{w for w in re.split(r"\W+", f.lower()) if len(w) > 3} for f in h_flags]
        overlap = 0
        for cw in c_words:
            for hw in h_words:
                if len(cw & hw) >= 3:
                    overlap += 1
                    break
        results["per_doc"][doc_key] = {
            "claude_n": len(c_flags),
            "human_n": len(h_flags),
            "keyword_overlap": overlap,
        }
        total_c += len(c_flags)
        total_h += len(h_flags)
        total_overlap += overlap

    results["aggregate"] = {
        "claude_total_flags": total_c,
        "human_total_flags": total_h,
        "total_keyword_overlap": total_overlap,
        "mean_claude_per_doc": round(total_c / max(len(doc_keys), 1), 1),
        "mean_human_per_doc": round(total_h / max(len(doc_keys), 1), 1),
    }
    return results


def main() -> int:
    claude = _load_json(EVAL_DIR / "claude_inference.json").get("documents", {})
    human = _load_json(EVAL_DIR / "my_inference.json").get("documents", {})
    themis = _load_json(EVAL_DIR / "themis_out.json")

    print("=" * 72)
    print("THREE-WAY COMPARISON: Claude vs Human vs Themis")
    print("=" * 72)

    # --- Set-level ---
    sl = set_level_comparison(claude, human, themis)
    print("\n## SET-LEVEL (unique clause types per document)\n")
    print(f"{'Document':<42} {'Pair':<22} {'J':>5} {'P':>5} {'R':>5} {'F1':>5}")
    print("-" * 84)
    for doc_key in sorted(sl["per_doc"]):
        for pair_name, metrics in sl["per_doc"][doc_key].items():
            short = pair_name.replace("_vs_", "/")
            print(f"{doc_key:<42} {short:<22} {metrics['jaccard']:>5.3f} "
                  f"{metrics['precision']:>5.3f} {metrics['recall']:>5.3f} {metrics['f1']:>5.3f}")

    print(f"\n{'AGGREGATE':<42}")
    for pair_name, metrics in sl["aggregate"].items():
        short = pair_name.replace("_vs_", "/")
        print(f"  {short:<20} J={metrics['mean_jaccard']:.3f}  P={metrics['mean_precision']:.3f}  "
              f"R={metrics['mean_recall']:.3f}  F1={metrics['mean_f1']:.3f}  (n={metrics['n_docs']})")

    # --- Section-level ---
    sec = section_level_comparison(claude, human, themis)
    print("\n\n## SECTION-LEVEL (matched sections, type agreement)")
    print(f"\n  Claude↔Human matched sections: {sec['aggregate']['matched']}")
    print(f"  Claude↔Human type agreement:   {sec['aggregate']['overall_agreement']:.1%}")
    for doc_key in sorted(sec["per_doc"]):
        d = sec["per_doc"][doc_key]
        print(f"\n  {doc_key}:")
        print(f"    Claude↔Human: {d['n_matched']} sections matched, "
              f"{d['claude_human_agreement']:.1%} type agreement")
        print(f"    Themis:       {d['n_themis_matched']} sections matched, "
              f"{d['themis_agreement']:.1%} type agreement")

    # --- Risk flags ---
    rf = risk_flag_comparison(claude, human)
    print("\n\n## RISK FLAGS")
    print(f"  Claude total: {rf['aggregate']['claude_total_flags']} "
          f"({rf['aggregate']['mean_claude_per_doc']:.1f}/doc)")
    print(f"  Human total:  {rf['aggregate']['human_total_flags']} "
          f"({rf['aggregate']['mean_human_per_doc']:.1f}/doc)")
    print(f"  Keyword overlap: {rf['aggregate']['total_keyword_overlap']}")
    for doc_key in sorted(rf["per_doc"]):
        d = rf["per_doc"][doc_key]
        print(f"    {doc_key}: Claude={d['claude_n']} Human={d['human_n']} overlap={d['keyword_overlap']}")

    # --- Write full results ---
    out = EVAL_DIR / "three_way_comparison.json"
    out.write_text(json.dumps({"set_level": sl, "section_level": sec, "risk_flags": rf}, indent=2) + "\n")
    print(f"\n\n[compare] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
