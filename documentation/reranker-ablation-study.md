# Reranker Ablation and the Decision to Disable It in AVAILABLE 2.0

## 1. Purpose

This document describes the investigation into whether the cross-encoder reranker
earns its place in the AVAILABLE 2.0 retrieval pipeline. A controlled ablation was
run over three configurations — no reranker, a smaller reranker, and the original
(heavier) reranker — holding the rest of the retrieval path fixed. The study shows
that the reranker improves the ordering of retrieved passages (context precision)
but does not translate that improvement into better document hit rate, answer
correctness, or judged answer quality, while adding a large query-time cost.

The outcome was therefore to **disable the reranker by default**
(`RERANKER_ENABLED = False`), keeping it as a switchable component for future
workloads rather than a mandatory stage. This is a measurement-driven design
decision, not a removal of capability.

---

## 2. The Reranker in the Retrieval Pipeline

The reranker is the final reordering stage of the hybrid retrieval path:

```text
User question
    |
    v
Vector leg (hybrid dense + sparse chunk search)   --+
                                                     |-- union fusion -> candidate pool
Graph leg (PPR over the knowledge graph)          --+
    |
    v
Cross-encoder reranker  (optional)
    |- reads (question, passage) pairs together
    `- reorders the fused candidate pool by relevance
    |
    v
Top FUSION_OUTPUT_K chunks -> LLM answer generation
```

Fusion builds the candidate pool by **rank agreement** between the vector and graph
legs: it is strong for recall but blind to the actual passage text. The reranker is
a cross-encoder that reads the query and each passage **together** and rescores
them, aiming to raise precision at the top of the list. Fusion for recall, the
reranker for precision.

Relevant implementation files are:

- `dc-available/dc_available/RAG/reranker.py` — the cross-encoder wrapper (a lazy
  `FlagReranker` singleton; scores normalised to 0–1).
- `dc-available/dc_available/RAG/qa_functions.py` — fusion and the optional rerank
  step (`_cross_encoder_rerank`, gated by `RERANKER_ENABLED`).
- `dc-available/dc_available/config/data_config.py` — `RERANKER_ENABLED`,
  `RERANKER_MODEL`, `FUSION_OUTPUT_K`.
- `dc-available/evaluation/available/reranker_ablation.py` — the ablation harness.

---

## 3. The Question

A cross-encoder reranker is a standard precision stage in RAG systems, but it is
also the single most expensive query-time component here because the model runs on
the CPU and scores every candidate passage individually. The question was concrete:

> Does the cross-encoder reranker improve end-to-end retrieval and answer quality
> enough to justify its added latency on this enterprise-handbook workload?

Because the workload is dominated by single-hop lookups where the fused pool
already tends to contain the correct document, it was not obvious that reordering
the pool would change the final outcome.

---

## 4. Ablation Design

### 4.1 Configurations

The production retrieval path was held constant (hybrid: vector + PPR-merge graph);
only the reranker was varied:

| Config | Setting | Role |
|---|---|---|
| `no_rerank` | `RERANKER_ENABLED = False` | union order only — the latency floor |
| `base` | `BAAI/bge-reranker-base` | smaller multilingual cross-encoder (~2× faster) |
| `v2m3` | `BAAI/bge-reranker-v2-m3` | the original model — the quality ceiling |

The reranker model is a singleton, so it was reset between configurations to force
the swap, and an untimed warm-up question paid the one-time model load per config.

### 4.2 Dataset and matching

- Dataset: the Grownfield question set (29 questions with golden document codes and
  golden answers).
- Document matching: golden codes such as `O2C-02.010`, matched at document level
  after reranking.
- Ranks evaluated: Hit@1, Hit@3, Hit@10.

### 4.3 Metrics

- **Retrieval:** MRR and Hit@1/@3/@10 at document level, measured on the final
  (post-rerank) order.
- **Generation:** RAGAS `answer_correctness` against the golden answer; a
  supplementary LLM-judge correctness label (correct / partial / incorrect); and
  the RAGAS `faithfulness`, `answer_relevancy`, and `context_precision` metrics.
- **Latency:** per-query `vector`, `graph`, `rerank`, and total retrieval time from
  `perf_counter`, with Opik tracing force-disabled so span overhead does not
  pollute the timings.

---

## 5. Results — Retrieval Metrics

Grownfield, 29 questions (`evaluation/new_results/reranker_ablation/grownfield/summary.json`):

| Config | Hit@1 | Hit@3 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| no_rerank | 0.6552 | 0.7586 | 0.9310 | 0.7288 |
| base | 0.6552 | **0.7931** | 0.9310 | **0.7425** |
| v2m3 | **0.6207** | 0.7586 | **0.9655** | 0.7299 |

- The smaller `base` reranker gives the best mid-rank ordering (Hit@3 and MRR),
  but the gain over `no_rerank` is marginal (Hit@3 +0.034, MRR +0.014).
- The heavier `v2m3` model **does not** dominate: it improves Hit@10 slightly but
  **lowers Hit@1** (0.6207 vs 0.6552) — reordering occasionally pushes a correct
  top-1 document down.
- Hit@1 — the metric that matters most for a single-answer handbook assistant — is
  **not improved by any reranker**.

---

## 6. Results — Generation and Judged Quality

Grownfield, 29 questions (`judge_summary.csv`):

| Config | Judge correctness | correct / partial / incorrect | RAGAS answer_correctness | Faithfulness | Answer relevancy | Context precision |
|---|---:|---|---:|---:|---:|---:|
| no_rerank | **0.6724** | 15 / 9 / 5 | 0.2764 | 0.9049 | **0.9183** | 0.5575 |
| base | 0.6897 | 14 / 12 / 3 | 0.2451 | 0.8883 | 0.8721 | 0.6619 |
| v2m3 | 0.6207 | 12 / 12 / 5 | 0.2834 | 0.9101 | 0.9022 | **0.7328** |

The one metric that improves **monotonically** with the reranker is
**context precision** (0.5575 → 0.6619 → 0.7328): the cross-encoder does what it is
designed to do — it pushes relevant passages higher in the pool. However, that
ordering gain does **not** carry through to the answer:

- **Judged answer correctness** is highest with **no reranker on the strict count**
  (15 fully-correct answers vs 14 for `base` and 12 for `v2m3`); `v2m3` is clearly
  the worst.
- **RAGAS answer_correctness** does not separate the configurations meaningfully
  (all ≈ 0.25–0.29) and does not favour the reranker.
- **Faithfulness and answer relevancy** are essentially flat; `no_rerank` has the
  highest answer relevancy.

In short: better context ordering did not produce better answers on this workload,
because the LLM already receives the correct passage within the top-k regardless of
whether it sits at rank 1 or rank 3.

---

## 7. Results — Latency

Mean per-query time (seconds), same runs:

| Config | Vector | Graph | Rerank | Total retrieval |
|---|---:|---:|---:|---:|
| no_rerank | 2.294 | 4.896 | **0.000** | **7.192** |
| base | 2.322 | 4.933 | 4.412 | 11.669 |
| v2m3 | 2.316 | 4.975 | **15.314** | **22.609** |

The reranker is the single most expensive stage when enabled. The original `v2m3`
model adds **~15 s** per query and roughly **triples** total retrieval time
(7.2 s → 22.6 s); even the smaller `base` model adds ~4.4 s. The high cost is
because the cross-encoder scores each candidate passage individually on the CPU.

> The latency figures come from one profiled run and should be treated as
> profiling evidence, not a statistically complete benchmark. Median and 95th-
> percentile results over repeated warm queries are required for final
> quantitative claims (see §10).

---

## 8. Interpretation and Decision

The ablation produces a clear cost/benefit picture:

- **Benefit:** the reranker improves context precision (passage ordering), and the
  smaller `base` model gives a marginal Hit@3 / MRR gain.
- **No downstream benefit:** Hit@1, judged answer correctness, RAGAS
  answer_correctness, faithfulness, and answer relevancy are **not** improved — and
  the heavier `v2m3` model actively hurts Hit@1 and judged correctness.
- **Large cost:** +4.4 s (`base`) to +15.3 s (`v2m3`) per query.

The decision was therefore to **disable the reranker by default**
(`RERANKER_ENABLED = False`). On this single-hop enterprise-handbook workload the
fused candidate pool already places the correct document within the top-k, so
reordering it changes the position of a passage the LLM would have seen anyway. The
reranker pays a large latency cost for an improvement the end user does not
perceive, and the heavier model can even degrade the top result.

The component is retained in the codebase and remains switchable, so it can be
re-enabled for workloads where top-rank precision matters more (for example
relational or multi-hop questions, which are under-represented in the current
evaluation set).

---

## 9. Why Not Keep the Smaller `base` Model?

`base` was the strongest reranker on the ordering metrics (best Hit@3 and MRR) at
roughly a third of `v2m3`'s latency, so it is the natural fallback if a reranker is
wanted. It was still not enabled by default because:

- its retrieval gains over `no_rerank` are marginal (Hit@3 +0.034, MRR +0.014) and
  do not reach Hit@1 at all;
- it does not improve — and on RAGAS answer_correctness slightly lowers — the final
  answer quality;
- it still adds ~4.4 s per query to an already multi-second path.

The marginal ordering gain did not justify the fixed per-query cost for the
default configuration.

---

## 10. Recommended Evaluation Protocol

To turn this study into a final quantitative claim:

1. Restart the application; record the first request separately as cold start.
2. Run one warm-up question per configuration to pay the one-time model load.
3. Execute the full question set (ideally expanded with relational / multi-hop
   questions, where the reranker is most likely to help).
4. Repeat for `no_rerank`, `base`, and `v2m3` on the same questions, hardware,
   model versions, and Qdrant collection.
5. Report median, mean, standard deviation, and 95th percentile for rerank time and
   total retrieval time.
6. Report Hit@k, MRR, context precision, RAGAS answer_correctness, and judged
   correctness side by side.
7. Confirm that disabling the reranker does not regress Hit@1 or judged answer
   correctness.

Cold and warm measurements must not be mixed, because the first request pays the
cross-encoder model load.

---

## 11. Threats to Validity

- The evaluation set is small (29 questions) and dominated by single-hop lookups,
  which is exactly where reordering a recall-complete pool helps least. The
  reranker's expected advantage on relational / multi-hop questions is
  under-represented.
- The cross-encoder runs on the CPU here; on a GPU its latency cost would be far
  smaller and could change the cost/benefit balance.
- RAGAS answer_correctness and the LLM judge are themselves model-based metrics with
  their own variance; the judged counts (15 / 14 / 12) differ by only a few
  questions.
- Document-level matching on golden codes measures whether the right document was
  retrieved, not whether the single best chunk was ranked first.
- Results are for this corpus and embedding/graph configuration; a different corpus
  or a noisier fused pool could make reranking pay off.

---

## 12. Thesis-Ready Summary

A cross-encoder reranker is a standard precision stage in retrieval-augmented
generation, but it was also the most expensive query-time component in AVAILABLE
2.0. A controlled ablation over three configurations — no reranker, a smaller
`bge-reranker-base`, and the original `bge-reranker-v2-m3` — holding the hybrid
vector-plus-graph retrieval path fixed, showed that the reranker improves passage
ordering (RAGAS context precision rose monotonically from 0.56 to 0.73) but does
not improve the metrics that matter to the user: document Hit@1, judged answer
correctness, and RAGAS answer_correctness were not improved by any reranker, and
the heavier model lowered both Hit@1 (0.655 → 0.621) and the count of fully-correct
answers (15 → 12). The cost was large: the original model added approximately 15
seconds per query and roughly tripled total retrieval time, while the smaller model
still added about 4.4 seconds.

The reranker was therefore disabled by default. On this single-hop
enterprise-handbook workload the fused candidate pool already contains the correct
document within the top-k, so reordering it does not change the passage the language
model ultimately reads. The component remains in the system and switchable, so it
can be re-enabled for workloads — such as relational or multi-hop retrieval — where
top-rank precision is more likely to affect the answer. As with the graph-retrieval
latency work, the guiding principle was to let measurement, rather than the
assumed value of a standard component, drive the configuration.
