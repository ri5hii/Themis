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
