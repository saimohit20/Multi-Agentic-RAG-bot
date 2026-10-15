# Hybrid

*Part of AVAILABLE 2.0 · query time (`dc_available/RAG/qa_functions.py` — fusion + reranker)*

---

## Aim

The system has **two eyes** that find passages in different ways:

- the **vector** eye — similarity by meaning and keywords (see [vector](vector.md)),
- the **graph** eye — connections through shared concepts and structure (see [graph](graph.md)).

Each eye catches things the other misses. The aim of the **hybrid** step is to combine them into one
ranked list of passages, pick the truly most relevant ones, and hand them to the LLM.

> "Hybrid" here means **vector eye + graph eye**. It is different from the dense + keyword blend that
> already happens *inside* the vector eye.

---

## How we solved it

The two legs run, their results are pooled, and the pool is re-scored more carefully before the top
passages go to the LLM.

```
question
   ├─ vector eye  →  top chunks
   └─ graph eye   →  top chunks
          │  UNION  (merge both into one candidate pool)
          ▼
   cross-encoder RERANK   (question + each chunk scored together)
          │
          │  × (1 + β · trust)      fold in document trust
          ▼
   keep top-K chunks  →  LLM answer   (+ the documents behind them)
```

1. **Run both legs** — the vector eye returns its top chunks and the graph eye returns its top
   chunks. (When the graph eye uses PPR, it can also take the vector eye's top chunks as extra
   starting seeds.)
2. **Union** — the chunks from both legs are merged into a single candidate pool. A passage found by
   both eyes is naturally a strong candidate. *(A weighted alternative, RRF, exists for comparison,
   but union is the default.)*
3. **Rerank** — a **cross-encoder** model (`BAAI/bge-reranker-v2-m3`) reads the question and each
   candidate passage **together** and scores how well the passage actually answers the question. This
   is more accurate than the first-stage similarity, but it only runs on the small pooled set.
4. **Fold in trust** — each score is multiplied by `(1 + β · trust)`, so more trustworthy documents
   are nudged up (see [trust](trust-score.md)).
5. **Cut** — the pool is sorted, the top-K passages are kept and sent to the LLM, and the unique
   documents behind them become the "recommended documents" shown to the user.

---

## Why a cross-encoder reranker

The first-stage vector search embeds the question and the passages **separately** — fast, and good
enough to shortlist candidates, but only approximate. A **cross-encoder** instead looks at the
question and one passage **at the same time**, so it judges relevance much more precisely. It is too
slow to run over the whole corpus, but running it over the small fused pool gives a sharp final
ordering for little cost.

---

## How this fits into AVAILABLE 2.0

The hybrid step is where the pipeline's retrieval signals come together at query time:

```
vector eye ┐
graph eye  ├─► union → rerank → trust → top-K → LLM
trust      ┘
```

The two-eye combination and the reranker are **new capabilities in AVAILABLE 2.0** — the 1.0 baseline
had a single vector leg with no fusion, no reranking, and no automated trust. Switching the graph eye
between its approaches (entity hop / PPR) or turning a leg off changes only this step; the vector eye,
the LLM prompt, and the UI stay the same.

---

## Configuration

| Variable | Purpose |
|---|---|
| `GRAPH_RETRIEVAL_MODE` | which graph engine feeds the pool (`entity_1hop` / `ppr`) |
| `FUSION_STRATEGY` | how the pool is built (`union` default / `rrf`) |
| `FUSION_VECTOR_K`, `FUSION_GRAPH_K` | how many chunks each eye contributes |
| `FUSION_OUTPUT_K` | how many chunks are sent to the LLM after reranking |
| `RERANKER_ENABLED`, `RERANKER_MODEL` | the cross-encoder reranker (`BAAI/bge-reranker-v2-m3`) |

---

## How to run

The hybrid step runs automatically whenever the app answers a question:

```bash
python -m pipeline.serve --mode qa    # or --mode assistant
```
