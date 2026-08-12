# Themis — External Dataset Candidates

Findings from web research (2026-08-14) for integrating better / larger
datasets into the v0.1.x line. Current in-repo corpora (gitignored, on disk):
`data/annotated/leivaditi_leases.jsonl` (8659 sections, 335 leases) and
`data/annotated/leivaditi_redflags.jsonl` (738 redflag sentences, 112 leases),
cleaned to `data/cleaned/` by `scripts/clean_dataset.py`.

## Summary table

| Dataset | Domain | Size | Labels | License/access | Relevance | Priority |
|---|---|---|---|---|---|---|
| Leivaditi redflag (UvA) | lease red-flag detection | 179 annotated leases; 53,232 clauses; 1,242 red flags / 19 types | red flags, entities | DOI 10.21942/uva.19732993; GitHub `j-rossi-nl/redflag`; GitLab mirror | **Origin of our current levi data — strictly richer** (full text + 19 types + entity spans) | **HIGH** |
| LEXDEMOD (Adobe) | deontic modality in leases | 7,092 clauses / 23 leases; 8,230 spans | obligation/entitlement/prohibition/permission per party | GitHub `adobe-research/LexDeMod` (MIT) | **Party-specific rights/duties — directly maps to our simplify/tenant_impact** | **HIGH** |
| CUAD v1 (Atticus) | commercial contract clauses | 510 contracts; 13k+ clause annotations | 41 clause categories | HF `dvgodoy/CUAD_v1_Contract_Understanding_clause_classification` | Broad clause taxonomy; useful for pretraining/zero-shot | MED |
| LEDGAR | legal provisions (SEC Exhibit-10) | 60,540 contracts; 846,274 provisions | 12k+ labels | Public (SEC/EDGAR); `sebischair/...` | Large-scale provisions; heavy, generic | LOW-MED |
| German tenancy law (TUM) | German rental law | 601 + 312 sentences | 3/6/9-type taxonomies | GitHub `sebischair/Legal-Sentence-Classification-Datasets-and-Models` | German-only; concept reference for clause taxonomy | LOW |
| Contractual clause retrieval (Isaacus) | IR retrieval benchmark | 45 clause types × 2 examples | retrieval pairs | HF `isaacus/contractual-clause-retrieval` | Evaluates retrieval/zero-shot clause ID | MED |

## Recommended integration (in order)

1. **Leivaditi redflag full benchmark** — replace/augment the extracted subset
   with the original annotated corpus. Gives full-text leases (no 600-char
   truncation), 19 red-flag types (vs our 7), entity spans, and ALeaseBERT
   weights. Sources:
   - DOI: <https://doi.org/10.21942/uva.19732993>
   - GitHub: <https://github.com/j-rossi-nl/redflag>
   - GitLab: <https://gitlab.com/spyretta.leiv/lease_contract_review>
2. **LEXDEMOD** — party-specific deontic labels (tenant vs landlord obligation/
   entitlement/prohibition) to supervise the plain-language layer and enrich
   `tenant_impact`. Source: <https://github.com/adobe-research/LexDeMod>
3. **CUAD v1 clause classification** — broad commercial clause taxonomy for
   zero-shot / pretraining support. Source:
   <https://huggingface.co/datasets/dvgodoy/CUAD_v1_Contract_Understanding_clause_classification>

## Why this matters (current data gaps)

- 47.5% of lease sections and 30.4% of redflag sentences are truncated at the
  scraper's 600-char cap — the original benchmark provides full documents.
- Truncation is concentrated in **classified** sections: known `type_fast_lane`
  sections are truncated at 60–100% (term 88%, premises 90%, pets 90%) while
  `unknown` sections are truncated only ~22%. Classifier labels thus carry the
  least complete text — a serious training-data bias.
- Redflag corpus sources are disjoint from the lease-section sources, so redflag
  labels cannot currently be joined to section text.
- Current redflags cover only 7 clause types; the original benchmark annotates
  19 red-flag types + entity spans.

> Note: truncation flags are computed on **cleaned** text (after whitespace
> normalization collapses runs of spaces), so they undercount raw truncation.
> Raw-level counts (47.5% / 30.4%) are the authoritative figures.
