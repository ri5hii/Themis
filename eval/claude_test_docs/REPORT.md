# Themis vs independent reading: claudeTestDocs

Method: I independently read all 9 PDFs (7 text-layer via pdftotext, 2 scanned via
docling/RapidOCR — the scanned ones are copies of lease_03/lease_05) and recorded
my own per-section clause-type inference + risk flags in `my_inference.json`,
BEFORE running Themis. Then Themis's shipped Extract engine
(`extractText` -> `buildRows` -> `analyzeSections` with fast-lane + trained
LegalBERT classifier fallback) was run on each PDF; output in `themis_out.json`.
`comparison.json` holds per-doc type-set agreement.

Note: this compares *clause-type coverage* against my independent judgment; the
real gold-standard eval (hand-labeled) is a separate exercise.

## Headline numbers (clause-type set agreement)

Numbers are after the segmentation fix (`splitParagraphs` now also splits on
prose headings; see Fix 1). Pre-fix mean: J=0.472, P=0.761, R=0.610.

| document | Jaccard | precision | recall |
|---|---|---|---|
| lease_01_office_meridian | 0.667 | 0.750 | 0.857 |
| lease_02_retail_sundial | 0.625 | 0.625 | 1.000 |
| lease_03_industrial_ferrous | 0.333 | 0.600 | 0.429 |
| lease_03_industrial_ferrous_SCANNED | 0.400 | 0.571 | 0.571 |
| lease_04_mixeduse_silverline | 0.636 | 0.700 | 0.875 |
| lease_05_smallsuite_thornfield | 0.714 | 1.000 | 0.714 |
| lease_05_smallsuite_thornfield_SCANNED | 0.625 | 0.833 | 0.714 |
| lease_06_table_rentschedule_bellwood | 0.429 | 0.600 | 0.600 |
| lease_07_edgecase_stresstest_windmere | 0.875 | 1.000 | 0.875 |
| **mean** | **0.589** | **0.742** | **0.737** |

## Findings

### 1. Segmentation failure on prose-style leases — FIXED
lease_05 used to collapse to **2 sections** (both `termination`) because its
headings are prose-style ("Premises and Term", "Rent", ...) with single `\r\n`
line breaks, no blank lines, and no numbered clause starts. `splitParagraphs`
(`src/legalrag/ingest/segment.py`) now also splits on short title-case lines
with no terminal punctuation, so lease_05 segments into 11 sections and recall
rose 0.143 -> 0.714. Remaining gaps are inherent, not segmentation:
- `late_fee` appears only inside the Rent clause (no dedicated article);
- `premises` is subsumed by the "Premises and Term" heading that fast-lane
  labels `term`;
- the `Insurance and Indemnity` heading sits alone on a page boundary (its body
  is on the next page) and go-dark / insurance / miscellaneous sections correctly
  stay `unknown` — no taxonomy type exists for them.
The scanned copy (13 sections) tracked the text twin (0.625 vs 0.714 Jaccard).

### 2. Missing type coverage on risk-heavy clauses
lease_03 misses `holdover`, `maintenance`, `premises`, `dispute_resolution`
(recall 0.429) despite the source having an explicit "HOLDOVER" article with
150% rent + "tenant at sufferance". Root cause verified: the holdover triggers
DO fire (evidenceCounts = {term:2, rent:1, holdover:2}) but the tie-break rule
(`classifyClause` picks the earlier taxonomy index — `term` at index 0 beats
`holdover` at index 6) classifies the section as `term`. So "holdover missing"
is a **tie-break artifact**, not a trigger gap. Fast-lane only fired 4/18
sections; the classifier fallback (14 classes, trained on auto-labels) filled
the rest. The scanned variant fared slightly better (recall 0.571).

### 3. Risk flags vs Themis
Themis is a clause-type classifier, not a risk scorer — it does not surface
deal-risk judgments directly. Independent risk flags Themis has no signal for:
- lease_02 / lease_05: **go-dark / no-obligation-to-operate** (only detectable
  via raw text; no taxonomy type exists)
- lease_03: guaranty auto-transfer without new instrument, holdover 150%
- lease_01: assignment on merger without consent, no duty to mitigate

Themis did correctly catch the risk-related *types* in most docs (holdover in
lease_03 text is the notable miss).

### 4. scan parity
The 2 scanned copies (lease_03/05) track their text twins closely (Jaccard
0.400/0.625 vs 0.333/0.714). OCR degradation did not materially change the
result — extraction quality was adequate.

## Actionable fixes (if we want to close the gap)
1. **segment.py — DONE**: split on prose headings too (short title-case line, no
   terminal punctuation). Landed in `src/legalrag/ingest/segment.py` with unit
   tests (wrapped-fragment and lowercase-continuation guards). Raised lease_05
   recall 0.143 -> 0.714 and mean recall 0.610 -> 0.737.
2. **fast_lane tie-break**: the taxonomy-order tie-break lets `term` (index 0)
   swallow `holdover` sections (lease_03 "tenant at sufferance" + 150% rent). A
   data-driven fix: give `holdover` evidence a higher weight, or break ties by
   total distinctive-lexicon strength instead of raw count. Verify with
   `evidenceCounts` unit tests. Also reproduced on the shipped Leivaditi eval:
   holdover recall is 0.009 (1/110).
3. Scope note: go-dark/no-obligation clauses have no taxonomy type — a risk layer
   would need a new type (or a separate boolean flag), separate from the 15-type
   clause taxonomy.
