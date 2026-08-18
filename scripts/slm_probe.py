"""Grammar-constrained SLM probe: plain-language clause explanation.

End-state design: deterministic engine handles classification/flagging; a
grammar-constrained SLM handles *prose* generation only. This probe evaluates
the LLM on a small sample of redflag sections from the full benchmark,
constraining output to the shipped grammar (legalrag.slm.grammar.GRAMMAR) —
the same strict JSON schema the risk stage uses for findings.

Outputs: JSON with {clause_type, risk_level, statute, plain_explanation,
tenant_impact}. Grammar guarantees the parse; the probe measures parse rate,
timing, and output quality.

Usage:
    PYTHONUNBUFFERED=1 python scripts/slm_probe.py --model models/...gguf \
        --n 20 [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legalrag import tasks
from legalrag.slm.grammar import GRAMMAR as GBNF_GRAMMAR
from legalrag.slm.grammar import SYSTEM_PROMPT, make_finding_prompt


def load_sample(n: int, seed: int, path: str) -> list[dict]:
    rows = tasks.loadJsonl(path)
    rows = [r for r in rows if r["type"] != "none" and r["text"]]
    rng = random.Random(seed)
    return rng.sample(rows, min(n, len(rows)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    from llama_cpp import Llama, LlamaGrammar

    grammar = LlamaGrammar.from_string(GBNF_GRAMMAR)
    sample = load_sample(
        args.n, args.seed, "data/cleaned/leivaditi_full_redflags.jsonl"
    )
    llm = Llama(
        model_path=args.model,
        n_ctx=4096,
        n_threads=args.threads,
        n_gpu_layers=0,
        verbose=False,
    )

    results: list[dict] = []
    t0 = time.time()
    for row in sample:
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": make_finding_prompt(row["text"], "", "", "")},
            ],
            grammar=grammar,
            max_tokens=256,
            temperature=0.6,
            top_k=40,
            top_p=0.95,
        )
        raw = out["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(raw)
            parsed["_gold_type"] = row["type"]
            results.append(parsed)
        except json.JSONDecodeError as e:
            results.append({"_gold_type": row["type"], "_parse_error": str(e), "_raw": raw[:200]})

    elapsed = time.time() - t0
    parsed_ok = sum(1 for r in results if "_parse_error" not in r)
    types_ok = sum(1 for r in results if "_parse_error" not in r and r.get("clause_type"))
    summary = {
        "n": len(results),
        "parse_rate": round(parsed_ok / len(results), 3),
        "valid_type_rate": round(types_ok / len(results), 3),
        "elapsed_s": round(elapsed, 1),
        "per_item_s": round(elapsed / len(results), 1),
        "items": results,
    }
    dst = Path("eval/artifacts/slm_probe.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "items"}, indent=2))
    for r in results[:5]:
        print("---")
        print(json.dumps(r, ensure_ascii=False))
    print(f"[slm_probe] wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())