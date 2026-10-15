# Generation evaluation (RAGAS) — placeholder

Reserved for the **generation** stage evaluation (RAGAS), to be implemented later.

Planned flow:
- Reuse the result JSON files produced by `retrieval_eval` (they already carry
  `question`, `golden_answer`, and `retrieved_contexts`).
- Fill `generated_answer` (and, for Grownfield, `generated_module` /
  `generated_submodule`) by running the QA pipeline.
- Score with RAGAS (faithfulness, answer relevancy, context precision/recall).

Structure:
- `src/`     — RAGAS scripts (empty for now)
- `results/` — scores + plots (empty for now)
