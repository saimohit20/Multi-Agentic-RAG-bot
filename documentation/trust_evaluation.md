# Trust Score — Evaluation

How we evaluated the document **trust score** used in the RAG re-ranking step. The score is
computed per document at ingestion and combines a **prior** (metadata + content quality) with
**corroboration** (a graph-based TrustRank):

```
final_trust = 0.7 · prior + 0.3 · trustrank
prior       = 0.45 · content + 0.25 · authority + 0.15 · format + 0.15 · freshness
```

The corpus (GROWNFIELD, 277 approved SAP S/4HANA training documents) is **uniformly approved**, so
there is no natural "untrustworthy" class to test against. Both evaluations therefore **manufacture
ground truth**. Two questions are answered:

- **Part 1 — Is the trust score correct?** (does it *detect* quality loss)
- **Part 2 — Does the trust score help the RAG?** (does it *improve* answers)

---

## Part 1 — Is the trust score correct? (perturbation sensitivity)

### Aim
Prove the score reacts *correctly* to degradation: when a document is damaged in one specific way,
the trust score should drop **in the matching pillar** and stay flat elsewhere. Since every real doc
is "good", we create the missing negative class by **controlled perturbation** of good docs.

### Implementation steps
1. **Select sample** — take 40 documents that have an LLM content summary (`content_source == "llm"`;
   the 94 videos are excluded as their content score is fixed at 1.0). Stratified, fixed seed (42).
   Baseline pillar values are read from the existing `metadata_trust.jsonl`.
2. **Perturb** — apply **5 controlled degradations**, one per copy. Three attack the *content* pillars
   via **gpt-5.2** (procedural = remove ~60 % of steps; technical = strip SAP specifics; filler =
   inject off-topic text); two attack *metadata* by fixed override (authority = author → unknown;
   freshness = backdate 4 years).
3. **Re-score** — feed each variant back through the **existing gpt-4o content scorer** (imported,
   not rewritten) and recompute the prior with the real weights. Using a different model to corrupt
   (gpt-5.2) and to score (gpt-4o) avoids circularity.
4. **Analyze** — compare each variant's pillars against its own baseline.

### Dataset: how and why
- **How:** 40 docs × (1 original + 5 perturbations) = **240 variants**. Content perturbations are
  saved as new summary files; metadata perturbations are inline overrides (no LLM needed).
- **Why perturbation:** it is the only way to obtain labelled "bad" documents in an all-approved
  corpus — we *know* the ground truth because we caused the damage.
- **Why one severity:** keeps the first pass clean (shows the score *responds*); dose–response
  (mild/severe grading) is a deliberate later extension.
- **Why score the summary:** it is the exact representation the production scorer consumes, so the
  result reflects the real pipeline.

### How we evaluated
- **(a) Descriptive:** for each perturbation, the **Δ of the targeted pillar** (headline), the
  **Δ of the overall prior** (secondary — small by design, since one pillar is diluted by the
  weighted average), and a **specificity check** that off-target pillars stay flat (the damage is
  localised, not a global collapse).
- **(b) Statistical:** because each perturbed doc is paired with its own original, we use the
  **paired Wilcoxon signed-rank test** (non-parametric, no normality assumption) on the pillar
  differences, reported with **Cohen's d** (effect size) and **Benjamini–Hochberg** correction for
  multiple comparisons.

---

## Part 2 — Does the trust score help the RAG? (3-arm A/B ablation)

### Aim
Prove the score is *useful*: when a trustworthy document and an untrustworthy near-copy give
**conflicting answers**, trust should make the RAG follow the trustworthy one. The approved corpus
has no such conflicts, so we manufacture a small **poisoned test corpus** and ablate trust on/off.

### Implementation steps
1. **Select docs** — 20 fact-dense grownfield documents (manually curated for rich, checkable
   content such as transaction codes).
2. **Make twins** — for each good doc, gpt-5.2 builds a **poisoned twin**: a near-identical copy with
   **~10 facts flipped** to wrong values, and metadata degraded (10 twins via freshness, 10 via
   authority). Each flipped fact becomes one test question (~190 total) whose answer *reveals* which
   document was used.
3. **Compute twin trust (graph)** — the twin's corroboration is derived by taking the good doc's
   graph entities **minus the flipped ones**, injecting the twin into the **full 277-doc graph in
   memory**, and running the same TrustRank PPR. This yields `test_trust_prior.jsonl` (prior only)
   and `test_trust_full.jsonl` (prior + corroboration).
4. **Build isolated index** — a separate Qdrant collection + docstore holding the 40 docs, so
   production is untouched. Retrieval is **vector-only** (the test corpus has no retrieval graph).
5. **Run 3 arms** — same questions, same index, only the trust config changes: **A0 off**,
   **A1 prior**, **A2 full**. Per question we record retrieval ranks + the generated answer.
6. **Score + analyze** — grade generation with RAGAS, then compare the arms.

### Dataset: how and why
- **How:** 20 good + 20 twins = 40 docs, ~190 questions (one per flipped fact), with a gold answer
  (correct value) and the twin's wrong value.
- **Why a poisoned twin:** it creates the **controlled conflict** the real corpus lacks — a good and
  a bad source that disagree on the *same* question, so the RAG must choose.
- **Why flip facts:** the answer text then directly betrays the source (correct value vs wrong
  value), giving automatic, unambiguous grading.
- **Why isolate + graph-in-memory:** the test must not touch production, so the vector index is
  separate; corroboration is a **corpus-level property** (it measures agreement with *other*
  documents), so the twin is scored inside the **full 277-doc graph**, not an artificial 40-doc one
  which would have no meaningful co-mention community.

### How we evaluated
Two layers, because trust acts on retrieval and the effect must survive into the answer:
- **Retrieval — `good_doc`:** did the good document rank **above** its twin? Binary and
  deterministic. Trust enters retrieval as a multiplier: `final_score = rerank × (1 + β · final_trust)`.
- **Generation — exact match (primary):** is the gold value present in the answer
  (case-insensitive)? Objective, because we authored the gold. We also track whether the answer
  followed the twin's wrong value.

**Generation is graded by three independent graders** (one agreement table):

| Grader | Role |
|---|---|
| **Exact match** (`correct_value in answer`) | ground-truth primary — drives the A/B result |
| **Own gpt-4o LLM judge** (temp 0; prompt built for short SAP codes/app names) | independent validation; confirms exact match |
| **RAGAS `answer_correctness`** | comparison exhibit — shown to *disagree* with ground truth |

Why three: RAGAS `answer_correctness` proved **unsuitable** here — it segments text into sentences
ending in a period and compares statements, but our answers/golds are bare codes ("MM01") or app
names with no period, so it returns empty statements and mis-scores (a wrong "MM04" scored 1.0, a
correct "…is QI03." scored 0.0). The own LLM judge and exact match agree closely; RAGAS is the
outlier. This is reported as an **evaluation-methodology finding** (metric selection matters for
short factual answers), not as the RAG result.

Statistics:
- **(a) Descriptive:** % `good_doc` (retrieval), % exact-match correct and % LLM-judge correct
  (generation) per arm, split by lever (freshness / authority); plus the **metric-agreement table**
  (each grader vs exact-match ground truth).
- **(b) Statistical:** **McNemar's test** (paired binary) on both the retrieval outcome and the
  exact-match generation outcome, for the three arm pairs (off↔prior, prior↔full, off↔full).

---

## Notes for the defense

- **Paired tests throughout.** Part 1 uses the paired **Wilcoxon signed-rank** test (continuous
  pillar deltas); Part 2 uses **McNemar's test** (paired binary outcomes) for both retrieval and
  exact-match generation. Both are paired because every variant/answer has a matched control.
- **Why exact match, not RAGAS, for generation.** RAGAS `answer_correctness` is built for prose; it
  segments answers into sentences and collapses on short transaction-code answers (empty statements
  → mis-scores). We therefore grade generation by exact match against the manufactured gold, and
  **validate it with an independent gpt-4o judge**. Reporting RAGAS's disagreement is a small
  methodological contribution: *off-the-shelf RAG metrics can fail on domain-specific short answers.*
- **Why the prior dominates (0.7 / 0.3).** Corroboration is **blind to a poisoned clone**: a twin
  shares almost all real entities with its original (the flipped values are made-up, appear in only
  one document, and are filtered as `df < 2` orphans), so its corroboration ≈ the original's. Only
  the **prior** separates them. Part 2's A2 ≈ A1 result is the evidence for this design choice, not
  a failure.
- **Limitations (all deliberate design choices).** Single severity (shows response, not yet
  dose–response); perturbation is applied at the summary level (skips the summarisation stage);
  corroboration cannot separate clones (motivates the prior-dominant blend); Part 2 retrieval is
  vector-only for isolation (full fusion is a later extension).
