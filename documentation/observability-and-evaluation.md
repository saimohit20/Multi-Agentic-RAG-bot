# Observability & Evaluation

*Part of AVAILABLE 2.0 · Opik tracing (`dc_available/`) + the `evaluation/` harness*

This document covers two things that watch over the system: **Opik**, which lets us see inside a live
request, and **Evaluation**, which measures how well the system retrieves and answers.

---

# Part 1 — Opik (Observability)

## Aim

A RAG pipeline does a lot of hidden work per question — search the vector store, walk the graph,
rerank, call the LLM. When an answer is wrong, we need to see **what happened inside**: what was
retrieved, what context the LLM got, how long it took, and what it cost. The aim of Opik is to make
every request transparent.

## How it works

Each step of the pipeline is recorded as part of a small hierarchy:

```
one question (request)
   ├─ span:  vector_retrieval
   ├─ span:  graph_retrieval
   ├─ span:  reranker
   └─ span:  generation (LLM)
        │  all spans roll up into one TRACE (the whole request)
        ▼
   TRACE  →  grouped into a THREAD (a whole conversation / session)
```

- A **span** is one step, a **trace** is one full request, and a **thread** is a whole session.
- Each step is tagged in code so its **inputs, outputs, token usage, and cost** are captured
  automatically and nested correctly.
- An **LLM judge** additionally scores each answer and attaches the scores to its trace as feedback
  (for example: hallucination, answer relevance, context precision).
- Everything is viewed in a **self-hosted Opik server** (a separate project the app sends traces to).

## Configuration

| Variable | Purpose |
|---|---|
| `OPIK_URL_OVERRIDE` | where the Opik server is (e.g. `http://localhost:5173/api`) |
| `OPIK_PROJECT_NAME` | the project traces are grouped under (`dc-available-rag`) |
| `OPIK_WORKSPACE` | the workspace name (`default`) |

## How this fits into AVAILABLE 2.0

Observability is a **new capability in 2.0** — the 1.0 baseline had no way to trace or monitor the
pipeline. Opik is what makes the system debuggable and is also where the per-answer quality scores
show up during normal use.

---

# Part 2 — Evaluation

## Aim

To improve the system we have to **measure** it objectively — not by spot-checking answers, but by
running a fixed set of questions through the real pipeline and scoring the results. This lets us
compare versions and retrieval approaches on equal footing.

## Two parts of evaluation

Evaluation asks two separate questions:

1. **Retrieval** — *did the system find the right document?*
   Metrics: **Hit@k** and **MRR** (also Recall@k and nDCG@10). These are **deterministic** (no LLM):
   each question has a known **golden source document**, and we check where it appears in the ranked
   list of retrieved documents.
2. **Generation** — *is the written answer good?*
   Metrics (scored by **RAGAS**, using an LLM as judge): **faithfulness** (the answer sticks to the
   retrieved context), **context relevance**, **answer relevance**, and **answer correctness** (versus
   the ground-truth answer).

## The datasets

Evaluation runs on **four Bosch-internal QA datasets**, each created by a different team, each a set
of questions with a ground-truth answer and a golden source document:

1. **Grownfield**
2. **Hydraulics**
3. **Digital Transformation**
4. **Synthetic** — a combined set drawn from all three topics above.

## How it works (two phases)

```
QA dataset  (question · ground-truth answer · golden source)     ← one of the 4 datasets
      │  Phase 1 — run each question through the REAL pipeline
      ▼
  predictions  (retrieved documents + generated answer + contexts)
      │  Phase 2 — score
      ├─ retrieval :  Hit@k · MRR · Recall@k · nDCG@10        (deterministic)
      └─ generation:  faithfulness · context relevance ·
                      answer relevance · answer correctness   (RAGAS — LLM judge)
```

- **Phase 1 (predict)** replays each question through the same retrieval + generation the app uses,
  and records what was retrieved, the answer, and the contexts. It saves as it goes, so a run can be
  resumed.
- **Phase 2 (score)** computes the retrieval metrics (by matching golden document codes) and the
  RAGAS generation metrics.

## What we compare

Beyond scoring one system, the harness is set up to run specific comparisons:

- **AVAILABLE 1.0 vs AVAILABLE 2.0** — the old baseline against the new system, on the same datasets,
  to show what the 2.0 changes achieved.
- **Graph approaches** — the **vector**, **graph** (entity hop / PPR), and **hybrid** legs side by
  side, to compare retrieval approaches under identical conditions.
- **Trust score** — whether folding document trust into re-ranking changes the results (and how the
  computed trust compares to a human-rated reference).

Each comparison reuses the same two-part measurement (retrieval + generation) so the numbers are
directly comparable.

## How this fits into AVAILABLE 2.0

A systematic evaluation harness is a **new capability in 2.0** — the 1.0 baseline had no quality
measurement. It is what turns "this feels better" into evidence, and it is how the graph approaches
and the trust score are judged.

## How to run

```bash
# whole AVAILABLE eval (retrieval + generation)
poetry run python -m evaluation.available.run_pipeline

# generic QA-dataset RAGAS flow (predict + score) on one xlsx
poetry run python -m evaluation.available.run_pipeline \
    --qa-dataset evaluation/datasets/test/qa_dataset.xlsx
```

Results are written to a timestamped folder under `evaluation/results/`.
```
