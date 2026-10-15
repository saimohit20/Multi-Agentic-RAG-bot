# AVAILABLE 2.0 — Final Evaluation Plan

## Datasets

| Dataset | Type | Role |
|---|---|---|
| Grownfield | real questions (exact-code answers) | component selection (dev) |
| Hydraulics | real questions (prose answers) | component selection (dev) |
| Synthetic | generated (grownfield + hydraulics + **digital transformation**) | final held-out test |

## Metrics

- **Retrieval:** MRR, Hit@1, Hit@3, Hit@10 (document-level, post-rerank)
- **Generation:**
  - **LLM-judge** (primary) — correct / partial / incorrect
  - **Human validation** (validates the judge — agreement / Cohen's κ on a sample)
  - **RAGAS** (secondary) — faithfulness, answer_relevancy, context_precision, answer_correctness

---

## The story (two phases)

**Phase 1 — Build the best system** (on real data: Grownfield + Hydraulics, combined)
Sequential selection — each step carries the winner forward:

> graph engine → retrieval eye → reranker

**Phase 2 — Prove it** (AVAILABLE 1.0 vs 2.0)
On **both** real (Grownfield + Hydraulics) *and* synthetic (held-out, incl. unseen digital-transformation domain).

### Why this design (justifications for the thesis)

- **Dev vs held-out test split.** Real questions are the development set used to *choose* components; the final 1.0-vs-2.0 comparison is also run on synthetic data, which was not used for any selection decision — so the final verdict is not overfit to the tuning data.
- **Sequential (greedy) component selection.** Components are optimized one step at a time, each carrying the previous winner forward (the eye study uses a graph leg; the reranker study holds retrieval fixed at hybrid + PPR-merge). Full factorial search is N×M×K runs; we assume locally-optimal components compose — standard practice in RAG ablation.
- **Final on real + synthetic.** Real data = credible proof on authentic questions; synthetic = scale + an unseen domain (digital transformation) → generalization. Two mutually-reinforcing results.
- **Graph engine = retrieval only.** The graph engine only changes which documents surface, not answer quality; a generator score on a deliberately single-leg (graph-only) input would be meaningless. Answer quality is evaluated once the full pipeline is assembled (E2 onward).

---

## PHASE 1 — Component selection (Grownfield + Hydraulics, real)

### E1. Graph engine — `entity_1hop` vs `PPR` (× seeding legacy/merge)

*Retrieval only* (graph-only leg — changes which docs surface, not answer quality).

| | x-axis | y-axis | group by |
|---|---|---|---|
| Retriever | {Hit@1, Hit@3, Hit@10, MRR} | 0–1 | {entity_1hop_legacy, entity_1hop_merge, ppr_legacy, ppr_merge} |

### E2. Retrieval eye — `vector` vs `graph` vs `hybrid`

| | x-axis | y-axis | group by |
|---|---|---|---|
| Retriever | {Hit@1, Hit@3, Hit@10, MRR} | 0–1 | {vector, graph, hybrid} |
| Generator | {judge, RAGAS 4} | 0–1 | {vector, graph, hybrid} |

### E3. Reranker — `no_rerank` vs `base` vs `v2m3` (quality **vs latency**)

| | x-axis | y-axis | group by |
|---|---|---|---|
| Latency (motivation) | {vector, graph, reranker, generation} | mean latency (s) | dataset |
| Retriever | {MRR, Hit@1, Hit@3} | 0–1 | {no_rerank, base, v2m3} |
| Generator | {judge, faithfulness} | 0–1 | {no_rerank, base, v2m3} |
| Decision | end-to-end latency (s) | judge correctness | 1 point per config |

---

## PHASE 2 — Final: AVAILABLE 1.0 vs 2.0

Run on **real** (Grownfield + Hydraulics) and **synthetic**.

| | x-axis | y-axis | group by |
|---|---|---|---|
| Retriever | {Hit@1, Hit@3, Hit@10, MRR} | 0–1 | {1.0, 2.0} — facet per dataset |
| Generator | {judge, RAGAS 4} | 0–1 | {1.0, 2.0} — facet per dataset |
| Human validation | — | judge↔human agreement (κ) | sample across systems |

---

## Plotting style (canonical)

All figures across every experiment (E1–E3 + Phase 2) must follow one consistent
publication style — the one already used in
`evaluation/available/plot_results.py` (output folder
`evaluation/results/reranker_ablation/grownfield/figures/`). When writing any new
plotting script, reuse that style:

- **Palette:** Wong colour-blind-safe (Nature Methods, Wong 2011) — see the `WONG`
  dict in `plot_results.py`.
- **Type / chrome:** sans-serif, coupled font sizes, no top/right spines, light
  y-grid, white background (see `set_style()`).
- **Labels:** value labels on bars, border-less legend placed above the axes when it
  would overlap bars.
- **Export:** 300 dpi **PNG** only (no PDF).

> Rule of thumb: copy `set_style()` + the `WONG` palette + `save()` from
> `plot_results.py` into any new figure script rather than re-styling by hand.

---

## Locked decisions

1. Final 1.0 vs 2.0 → real **+** synthetic ✅
2. Human validation of judge → yes ✅
3. Graph engine → retrieval only ✅
4. Exact-identifier metric → **removed** ✅
5. Generator metrics → judge (primary) + human (validation) + RAGAS (secondary) ✅
