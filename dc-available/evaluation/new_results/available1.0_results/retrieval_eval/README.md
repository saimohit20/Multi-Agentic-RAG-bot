# Retrieval evaluation

Evaluates the **retrieval** stage of the AVAILABLE RAG system on two topics:
**Grownfield** and **Hydraulics**. It runs the existing FAISS index (no rebuild)
and, for each question, records the top-10 retrieved chunks and whether the
golden source document was found.

## What a "hit" means (document level)

| Topic | Golden source | Hit = |
|-------|---------------|-------|
| Grownfield | `golden_codes` (list, e.g. `O2C-02.010`) | a retrieved chunk's document code ∈ golden codes |
| Hydraulics | `source` PDF name | a retrieved chunk belongs to that document |

`golden_rank` = 1-based rank of the **first** matching chunk in the top-10
(`null` if the golden document is absent).

## Metrics

- **identification** — golden document present in top-10 (counts per topic)
- **hit@5** — `golden_rank <= 5`
- **MRR** — mean over all questions of `1/golden_rank` (0 for misses)

`TOP_K` (retrieved/stored) and `HIT_K` (hit@k) live in `src/paths_config.py`.

## How to run (from the repo root `dc-available/`)

Single command — both topics + plots (embeddings/index load once):

```powershell
.venv/Scripts/python.exe evaluation/retrieval_eval/src/run_all.py
```

Or step by step:

```powershell
.venv/Scripts/python.exe evaluation/retrieval_eval/src/run_grownfield.py
.venv/Scripts/python.exe evaluation/retrieval_eval/src/run_hydraulics.py
.venv/Scripts/python.exe evaluation/retrieval_eval/src/make_plots.py
```

## Outputs (`results/`)

- `grownfield_results.json`, `hydraulics_results.json` — one record per question
- `plots/identification.png` — identified vs not identified, per topic
- `plots/retrieval_metrics.png` — hit@5 and MRR, per topic

## Result record schema

Common: `question_id, question, question_type, level, type, golden_source,
golden_answer, retrieved_docs, golden_rank, retrieved_contexts,
generated_answer (null)`.

Grownfield additionally: `module, submodule, generated_module (null),
generated_submodule (null)`.

`retrieved_contexts` (the top-10 chunk texts) and the `generated_*` null fields
are placeholders for the later generation / RAGAS evaluation.

## Note

The embedding model is English (`BAAI/bge-small-en-v1.5`) while the Hydraulics
material is German — expect lower retrieval scores there; that is a genuine
finding, not a harness bug.
