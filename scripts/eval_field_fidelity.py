"""Field-fidelity eval of the tuned SLM on the held-out finetune pairs.

Probes the shipped tuned GGUF on ``data/finetune/eval.jsonl`` (held-out
finding->explanation pairs, 15% split, seed 42). Per item it scores:

  - ``parse_ok``: grammar-constrained JSON parsed
  - field fidelity: clause_type / risk_level / statute agreement with the
    gold engine-stamped fields in the pair's assistant message
  - prose fidelity: ``plain_explanation`` exact/overlap match against the
    gold explanation (the training target is ``rationale[:300]``), and
    template detection on ``tenant_impact``

The gold prose IS the engine-rationale echo by construction, so "prose
fidelity" here measures how faithfully the model reproduces the target
behavior — which is the quality gap flagged in progress.md §6.22.

Usage:
    PYTHONUNBUFFERED=1 python scripts/eval_field_fidelity.py \
        [--model models/qwen2.5-1.5b/qwen2.5-1.5b-instruct-tuned-q8_0.gguf]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legalrag.slm.grammar import GRAMMAR as GBNF_GRAMMAR

FIELDS = ("clause_type", "risk_level", "statute")
DEFAULT_MODEL = (
    "models/qwen2.5-1.5b/qwen2.5-1.5b-instruct-tuned-q8_0.gguf"
)
EVAL_DATA = "data/finetune/eval.jsonl"


def load_pairs(path: str) -> list[dict]:
    pairs: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def gold_of(pair: dict) -> dict | None:
    """Engine-stamped gold fields from the pair's assistant message."""
    msgs = pair.get("messages", [])
    if len(msgs) < 3:
        return None
    try:
        return json.loads(msgs[2]["content"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    pairs = load_pairs(EVAL_DATA)
    if not pairs:
        print(f"[error] no pairs in {EVAL_DATA}")
        return 1

    from llama_cpp import Llama, LlamaGrammar

    grammar = LlamaGrammar.from_string(GBNF_GRAMMAR)
    llm = Llama(
        model_path=args.model,
        n_ctx=4096,
        n_threads=args.threads,
        n_gpu_layers=0,
        verbose=False,
    )

    results: list[dict] = []
    t0 = time.time()
    for pair in pairs:
        gold = gold_of(pair)
        item: dict = {
            "source": pair.get("source", "?"),
            "rule_id": pair.get("rule_id", "?"),
            "parse_ok": False,
        }
        if gold is None:
            item["skip"] = "malformed pair"
            results.append(item)
            continue

        out = llm.create_chat_completion(
            messages=pair["messages"][:2],
            grammar=grammar,
            max_tokens=args.max_tokens,
            temperature=0.6,
            top_k=40,
            top_p=0.95,
        )
        raw = out["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            item["_parse_error"] = str(e)
            item["_raw"] = raw[:200]
            results.append(item)
            continue

        item["parse_ok"] = True
        item["gold"] = {k: gold.get(k) for k in FIELDS}
        item["out"] = {k: parsed.get(k) for k in FIELDS}
        item["field_fidelity"] = [
            parsed.get(k) == gold.get(k) for k in FIELDS
        ]
        expl = str(parsed.get("plain_explanation", "")).strip()
        g_expl = str(gold.get("plain_explanation", "")).strip()
        impact = str(parsed.get("tenant_impact", "")).strip()
        g_impact = str(gold.get("tenant_impact", "")).strip()
        item["prose_exact"] = bool(expl) and expl == g_expl
        item["prose_overlap"] = bool(expl) and (
            g_expl in expl or expl in g_expl
        )
        item["template_impact"] = bool(impact) and (
            impact == g_impact or impact.startswith("This is a")
        )
        item["plain_explanation"] = expl
        item["tenant_impact"] = impact
        item["elapsed_s"] = round(time.time() - t0, 1)
        results.append(item)

    elapsed = time.time() - t0
    scored = [r for r in results if r.get("parse_ok")]
    n = len(results)
    parsed = len(scored)
    field_fidelity = sum(
        1 for r in scored if all(r["field_fidelity"])
    )
    prose_exact = sum(1 for r in scored if r["prose_exact"])
    prose_overlap = sum(1 for r in scored if r["prose_overlap"])
    template = sum(1 for r in scored if r["template_impact"])
    summary = {
        "n": n,
        "parse_rate": round(parsed / max(n, 1), 3),
        "field_fidelity": round(field_fidelity / max(parsed, 1), 3),
        "prose_exact_rate": round(prose_exact / max(parsed, 1), 3),
        "prose_overlap_rate": round(prose_overlap / max(parsed, 1), 3),
        "template_impact_rate": round(template / max(parsed, 1), 3),
        "elapsed_s": round(elapsed, 1),
        "per_item_s": round(elapsed / max(n, 1), 1),
        "items": results,
    }
    dst = Path("eval/finetune/field_fidelity.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps({k: v for k, v in summary.items() if k != "items"}, indent=2))
    print("--- per item (first 8)")
    for r in results[:8]:
        flag = "OK" if all(r.get("field_fidelity", [])) else "--"
        print(
            f"{r.get('source','?'):<42} fid={flag} exact={int(r.get('prose_exact', False))} "
            f"ovlp={int(r.get('prose_overlap', False))} tpl={int(r.get('template_impact', False))}"
        )
    print(f"[field_fidelity] wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())