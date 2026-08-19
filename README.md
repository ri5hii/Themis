# LegalRAG

RAG-based legal document analysis to narrow the access-to-justice gap.

Accepts PDF, scanned, image, and plain-text legal documents and returns
clause-level risk annotations with plain-language explanations and a
simplified summary, grounded on relevant statute context.

- **Parse** — pypdfium2 (Apache-2.0) text layer; rapidocr (Apache-2.0) lazy OCR for scanned pages; python-docx (MIT) for .docx
- **Extract** — fast lane: regex triggers + LegalBERT prototype classifier over a 16-type lease clause taxonomy
- **Retrieve** — LegalBERT embeddings + FAISS over a curated statute corpus (Model Tenancy Act 2021 + Delhi Rent Act 1958)
- **Risk** — rule-based engine with hybrid confidence scoring
- **Simplify** — grammar-constrained Qwen2.5-1.5B-Instruct (GGUF Q8, LoRA-tuned on aligned pairs) plain-language output as JSON

## Setup

```bash
uv sync --all-extras          # venv + base, slm, ocr, gui, dev deps (reproducible via uv.lock)
uv run themis --help          # run without activating
```

Or activate the venv first (then `themis` is on PATH):

```bash
source .venv/bin/activate       # bash/zsh
source .venv/bin/activate.fish  # fish
```

The `themis` CLI (and `legalrag-gui`) land on PATH from the venv:

## CLI commands

```bash
themis analyze <lease.pdf>                         # full pipeline -> console report
themis analyze <lease.pdf> --slm                   # + plain-language simplification
themis analyze <lease.pdf> -f json -o out.json     # machine-readable findings
themis analyze <lease.pdf> -f markdown -o out.md   # report for the client
themis analyze <lease.pdf> -i                      # interactive review of findings
themis annotate <lease.pdf> -o gold.jsonl          # interactive section re-annotation (gold labels)
themis train classify [--data auto|gold]           # clause-classifier fallback
themis train slm [--epochs 3 --lr 2e-4 --load-8bit]  # plain-language SLM (LoRA)
themis train ground                                # statute-grounding gate
legalrag-gui                                       # desktop GUI (Qt shell)
```

Every subcommand accepts `--help` (e.g. `themis analyze --help`) for the
full flag list.

## Training (`themis train`)

Trainable components, with their data sources:

| Component | Command | Data |
|---|---|---|
| Clause-classifier fallback | `themis train classify [--data auto\|gold]` | auto: fast-lane labels from `leivaditi_leases.jsonl`; gold: redflag sentences + `themis annotate` sections |
| Plain-language SLM (LoRA) | `themis train slm [--epochs 3 --lr 2e-4 --r 8 --alpha 16]` | aligned pairs in `data/finetune/{train,eval}.jsonl` |
| Statute-grounding gate | `themis train ground` | (rule query, anchor-matched statute chunk) positives vs random negatives |

Segment (heuristics) and risk (hand-authored trigger rules) are **not
trainable by design** — "training" those means editing `rules.py`
triggers/anchors, so the CLI never implies otherwise.

Every artifact carries provenance in its `meta.json` — training timestamp,
git commit (+ dirty flag), package version, and a SHA-256 of the training
rows — and `themis analyze` records which classifier/gate versions a run
consumed under `artifacts` in its JSON output. Retraining never overwrites:
the previous artifact is moved to `models/backups/<kind>/<stamp>/`.

## SLM retrain → deploy cycle

`themis train slm` writes a LoRA adapter (QLoRA 8-bit needs
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --max-length 384` on
~4 GB GPUs). The runtime model is a GGUF quant, so deploy is:

```bash
python scripts/merge_lora.py                 # adapter -> models/qwen2.5-1.5b-merged/
python tools/convert_hf_to_gguf.py models/qwen2.5-1.5b-merged \
    --outfile models/qwen2.5-1.5b/qwen2.5-1.5b-instruct-tuned-q8_0.gguf --outtype q8_0
python scripts/eval_field_fidelity.py        # held-out fidelity gate
```

The assistant targets come from curated plain-language golds
(`data/finetune/gold_prose.jsonl`, keyed by lease × rule) built from the
Claude `risk_flags`; uncurated findings fall back to the template target.
`scripts/verify_repo.py` gates on the fidelity eval (parse 1.0, no template,
no rationale echo). `tools/convert_hf_to_gguf.py` is pinned to llama.cpp
commit a290ce6 (matches the installed `gguf` 0.19.0).
