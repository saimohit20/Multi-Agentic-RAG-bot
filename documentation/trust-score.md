# Trust Score

*Part of AVAILABLE 2.0 · `dc_available/trust_score/`*

---

## Aim

Not every training document is equally reliable. Some are current and authoritative; others are
outdated, thin, or come from an external vendor. Pure relevance retrieval can surface a weak document
just because its words match the question.

The aim is to compute an **automatic trust score** for each document — once, at ingestion time,
independent of any query — and use it to gently favour trustworthy sources when answering. This
replaces the **manual** `trust.json` ratings of the 1.0 baseline with a score derived from the
documents themselves.

---

## Solution

A trust score built in **three parts**:

```
SharePoint metadata  +  document summaries
        │   collect → align → content-score → merge
        ▼
  1) PRIOR (metadata)      freshness + authority + format + content quality
        │   propagate
        ▼
  2) TRUSTRANK (graph)     spread the prior across a document-to-document graph
        │   blend
        ▼
  3) FINAL TRUST   ──►   used at re-rank:   score × (1 + β · trust)
```

- The **prior** asks: *what do we know about this document on its own?*
- **TrustRank** asks: *do other reliable documents back it up?*
- The **final trust** blends the two into one number and applies it during retrieval.

---

## How we solved it

### Part 1 — Prior (from metadata)

A base score per document (0–1), combining four "pillars". Each pillar is a 0–1 sub-score and the
weights add up to 1:

| Pillar | Weight | What it measures |
|---|---|---|
| **Content quality** | 0.45 | an LLM rates the document's summary on how useful it is as training material (technical depth, complete procedure, usefulness, focus). Videos score full by policy. |
| **Authority** | 0.25 | who authored it — **internal** counts more than an **external** vendor. |
| **Format** | 0.15 | the file type — video/PowerPoint weigh higher, PDF/web-page lower. |
| **Freshness** | 0.15 | a gentle decay with age (recency is only a weak trust signal, so a small weight). |

Before this can be scored, the document **metadata is collected** from SharePoint (via Microsoft
Graph), **aligned** to the actual ingested documents, the **content quality** is scored by the LLM,
and everything is **merged** into one row per document.

### Part 2 — TrustRank (from the graph)

A document's trust should not depend on its own metadata alone: a document that many reliable
documents **corroborate** deserves more trust. TrustRank spreads the prior over a
**document-to-document graph**:

- **Nodes** are documents.
- Two documents are **connected** when they share **specific** (high-IDF) entities — generic,
  everywhere-terms are ignored, so only documents that discuss the same specific things are linked.
- The graph is **seeded** with the metadata prior, and the **same PageRank maths as the
  [PPR](graph.md) retriever** is run (`trust = α · S · trust + (1−α) · prior`). Trust therefore flows
  from reliable sources to the documents that back up their specific claims.
- Documents with no entities (e.g. text-less videos) simply keep their prior — there is nothing to
  corroborate them.

### Part 3 — Final trust

The prior stays in charge and corroboration nudges it:

```
final_trust = 0.7 · prior  +  0.3 · trustrank
```

The result is a single 0–1 value per document.

### Using it during retrieval

At query time, after the [hybrid](hybrid.md) reranker has scored the candidate passages, each score
is multiplied by `(1 + β · final_trust)` — so more trustworthy documents rise in the final ranking.
If the trust file is missing, every document is treated as neutral (no reordering), so retrieval
still works without it.

---

## How this fits into AVAILABLE 2.0

Automated trust scoring is a **new capability in AVAILABLE 2.0**, replacing the manual `trust.json`
of 1.0. It is computed **once at ingestion time**, so there is no query-time cost beyond a quick
lookup. It reuses two pieces the project already has: the knowledge-graph **entities** (to decide
which documents corroborate each other) and the **PageRank maths** from the PPR retriever. Trust is
currently built for the **GROWNFIELD** dataset.

---

## Configuration & outputs

The stages read and write a chain of files under
`data/Artifacts/resources/trust/grownfield/`:

```
metadata.jsonl -> metadata_aligned.jsonl -> content_scores.jsonl
             -> metadata_scored.jsonl -> metadata_prior.jsonl -> metadata_trust.jsonl
```

Key settings (tunable at the top of the scripts):

| Setting | Meaning |
|---|---|
| pillar weights | content 0.45 · authority 0.25 · format 0.15 · freshness 0.15 |
| `ALPHA` | how far trust spreads in the TrustRank walk |
| final blend | `0.7 · prior + 0.3 · trustrank` |
| trust boost `β` | how strongly trust reorders the final ranking |

Dependencies: collecting metadata uses the **SharePoint / Microsoft Graph** access (from
`pipeline/.env`), and the content score uses **Azure OpenAI**.

---

## How to run

The scripts run in order (from the `dc-available` directory):

```bash
python -m dc_available.trust_score.collect_metadata  # gather SharePoint metadata
python -m dc_available.trust_score.align_metadata    # match metadata to documents
python -m dc_available.trust_score.prior_score       # content, merge, and prior score
python -m dc_available.trust_score.trustrank         # corroboration propagation
python -m dc_available.trust_score.final_trust       # publish metadata_trust.jsonl
```

The automated AVAILABLE pipeline runs these stages after indexing:

```bash
python -m pipeline.run_pipeline --source available
```

`components/score_content.py`, `components/merge_scores.py`, and
`components/score_prior.py` are internal modules of `prior_score.py`. They are
kept because the evaluation suite tests the production scorer implementation.
