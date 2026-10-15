# Trust Score Evaluation

This package evaluates the production implementation in
`dc_available.trust_score`; it is not called by the AVAILABLE pipeline.

## Suites

- `perturbation/` checks whether controlled content and metadata degradations
  lower the intended trust-score pillars.
- `rag_ab/` compares RAG behavior with trust disabled, with the prior only,
  and with prior plus TrustRank.

Both suites read the production `metadata_trust.jsonl` as a baseline. Their
generated documents, result files, figures, and isolated Qdrant collection
configuration remain under this evaluation package.

## Running

Run both experiments (or one) end-to-end with the consolidated pipeline:

```bash
python -m evaluation.trust_score.run_pipeline               # both experiments
python -m evaluation.trust_score.run_pipeline perturbation  # Experiment 1 only
python -m evaluation.trust_score.run_pipeline rag_ab        # Experiment 2 only
python -m evaluation.trust_score.run_pipeline rag_ab --limit 6   # quick dry run
```

Or run a single stage with its module path, for example:

```bash
python -m evaluation.trust_score.perturbation.select_sample
python -m evaluation.trust_score.rag_ab.run_ab
```