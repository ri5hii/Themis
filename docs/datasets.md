# Themis — External Dataset Candidates

Findings from web research (2026-08-14) for integrating better / larger
datasets into the v0.1.x line. Current in-repo corpora (gitignored, on disk):
`data/annotated/leivaditi_leases.jsonl` (8659 sections, 335 leases) and
`data/annotated/leivaditi_redflags.jsonl` (738 redflag sentences, 112 leases),
cleaned to `data/cleaned/` by `scripts/clean_dataset.py`.

## Status

- **DONE** — Full Leivaditi redflag benchmark fetched and ingested (v0.0.4):
  raw CSVs at `data/annotated/leivaditi_full/`, canonical JSONL at
  `data/cleaned/leivaditi_full_*.jsonl` via `scripts/ingest_leivaditi_full.py`
  (see "Ingested corpora" below).
- **Pending** — LEXDEMOD (party-specific deontic labels), CUAD (commercial
  clause taxonomy).

## Ingested corpora (full benchmark)

| Corpus | Rows | Docs | Notes |
|---|---|---|---|
| `leivaditi_full_docs.jsonl` | 171 | 171 | Full-text lease documents, 1.4K–288K chars (no 600-char cap); classes: lease agreement 115, other 42, amendment 9, sublease agreement 5 |
| `leivaditi_full_redflags.jsonl` | 52,853 | 179 | Paragraph-level, 863 positives across 19 types + 51,990 'none' negatives |
| `leivaditi_full_easy_redflags.jsonl` | 1,242 | 115 | Sentence-level redflag spans (19 types), each verified present in its doc text |
| `leivaditi_full_entities.jsonl` | 2,101 | 123 | NER spans, 12 classes (term_of_payment 406, lessor 376, lessee 307, leased_space 190, ...) |
| `leivaditi_full_clauses.jsonl` | 32,951 | 120 | Clause structure: 8,426 clause starts; clause_title 2,131, sub_clause_title 1,554, clause_number 610, annex 125 |

The 19 redflag types: services_charges, sublease_permitted, break_option,
reinstatement_clause, landlord_repairs, damage, guarantee_transferable,
change_of_control, right_of_first_refusal_to_lease, right_of_first_refusal_to_purchase,
warrantees_of_the_owner, compalsory_reconstraction, expansion, special_stipulations,
extension_period, additional_remarks, assignment_indeplaatsstelling_permitted,
riders, no_obligation_to_operate.

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

1. **Leivaditi redflag full benchmark** — ✅ **fetched + ingested** (v0.0.4).
   Full-text leases (no 600-char truncation), 19 red-flag types, entity spans,
   clause structure. Sources:
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
