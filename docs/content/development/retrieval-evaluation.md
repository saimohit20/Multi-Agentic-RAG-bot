# Retrieval Evaluation

Retrieval evaluation checks whether a question's reference source appears in
the documents selected for answering, and how early it appears. This guide
describes the experimental procedure, controls, and metrics without reporting
results. See [Hybrid Retrieval Design](hybrid-retrieval.md) for the serving path.

## Experimental Design

The evaluation plan separates component selection on real Grownfield and
Hydraulics questions from held-out assessment on synthetic questions. The
synthetic dataset includes Digital Transformation as well as the development
domains. This is a declared dataset separation, not a random split performed by
the retrieval loop. Preserve dataset versions and avoid choosing settings on
the held-out questions.

Component studies examine graph behavior, retrieval legs, and reranking
sequentially rather than searching every combination. This controls experiment
size but does not estimate all component interactions or establish a globally
optimal configuration. The final baseline comparison uses common questions and
reference matching across systems.

## Reference Sources

Each real-question record carries a question ID, question text, acceptable source
keys, and available question-type/domain metadata. Row-position IDs are used
when an ID column is absent; preserve ordering across reruns.

| Matching mode | Reference normalization | Successful match |
| --- | --- | --- |
| Grownfield code | Extract supported source-code patterns and uppercase them. | A retrieved document contains any acceptable reference code. |
| Hydraulics filename | Lowercase filename stem without extension. | The retrieved source stem equals the reference stem. |
| Synthetic document name | Remove a trailing file extension, lowercase, and collapse non-alphanumeric runs to spaces. | Normalized retrieved name equals normalized `source_document`. |

For multiple acceptable code references, finding any one is a hit. This does not
measure whether every necessary source was found. Document-code matching is a
source-identity rule, not an exact-answer correctness metric. Audit missing
references, ambiguous filenames, and unsupported code formats before aggregation.

## Measurement Boundary

For each question/configuration pair, run retrieval through the serving answer
path. Convert returned passages to unique owning document names in first
occurrence order, up to ten. The first matching document's position is the
reference rank. Ten passages can represent fewer than ten unique documents.

The real-question leg-comparison runner explicitly enables reranking and sets
the final passage limit to ten. It measures **post-selection, document-level**
ranking rather than raw vector recall. Trust and other inherited settings must
also be recorded. Save the selected passage texts for later generation so that
answer evaluation can reuse the same evidence. Retrieval evaluation itself does
not generate answers.

## Metrics and Failure Handling

| Metric | Definition |
| --- | --- |
| Hit@k | Mean of a binary indicator: any acceptable source occurs within the first k documents. |
| MRR | Mean of one divided by the first acceptable source's rank; absent references contribute zero. |

The main leg comparison uses Hit@1, Hit@3, Hit@10, and MRR. The current graph
ablation uses Hit@1, Hit@5, Hit@10, and MRR; its Hit@5 must not be relabeled as
Hit@3. Reciprocal rank is limited to the returned evidence list, not the complete
corpus ordering. Hit@k is not set recall for multi-source questions.

In these runners, retrieval exceptions become empty lists and count as misses,
not excluded questions. Track technical failures separately from valid searches
that did not retrieve the reference. Rows without valid reference keys cannot
score a hit and require explicit review.

## Graph and Retrieval-Leg Studies

Graph-only evaluation removes vector candidate retrieval and vector passage
seeds. The current graph runner compares one-hop, three-hop, and PPR with merge
seeding. The earlier evaluation plan describes a different two-engine by
legacy/merge-seeding study. Identify the protocol version and effective
configuration rather than inferring them from an older module description.

Hold corpus, entity index, topic isolation, and downstream ranking settings
constant. The graph runner inherits reranker/trust defaults rather than fixing
every option explicitly; record those values for each run.

The leg comparison evaluates vector-only, graph-only, and combined retrieval on
the same questions. Vector-only still uses dense/sparse Qdrant fusion. Combined
PPR can receive vector passage seeds, while graph-only PPR cannot: changing legs
therefore changes graph seeding as well as candidate combination.

Reranker comparisons should hold the retrieval pool policy fixed and record
relevance model, trust, and final output limits. Baseline comparisons need a
common document-matching policy even when internal retrieval differs. Separate
retrieval, reranking, and generation timing and identify cold/warm resources.

## Final-System Assessment

The dedicated AVAILABLE 2.0 final runner fixes combined vector/PPR retrieval,
merge seeding, and **reranking disabled**. It supports Hydraulics and synthetic
datasets, preserves the ten-passage selection boundary, and records source rank,
contexts, answer, and retrieval/generation durations. It skips blank question
rows; reference validation and configuration capture remain separate concerns.
Grownfield comparison inputs are prepared separately rather than run through
this runner's topic interface.

This fixed configuration is different from the three-leg evaluators, which
explicitly enable reranking. Do not mix their records under one configuration
label. A baseline comparison should retain each input artifact's lineage rather
than treating a comparison/plotting script as a fresh end-to-end rerun of both
systems. This guide makes no claim about which configuration performed better.

## Reproducibility

Preserve question/configuration records, ordered sources, reference ranks,
contexts, errors, and effective settings. Record dataset and corpus versions,
collection/graph builds, embedding/reranker models, candidate limits, fusion,
trust, and seed/topic policies.

The real-question evaluator reuses an existing topic JSON instead of retrieving
again; graph ablation similarly reuses `comparison.json`. Recomputed metrics or
plots are not a fresh retrieval experiment. Use distinct output locations or
deliberately manage cached inputs when changing questions or configuration.

The synthetic leg evaluator writes separate vector/graph/hybrid JSON files.
The older synthetic generation reader expects a combined `retrieval.json`.
These are not an automatic producer/consumer match: verify the chosen handoff
artifact and row schema before generation rather than assuming the consolidated
entry point reconciles them.

The consolidated real-question retrieval entry point is:

```powershell
poetry run python -m evaluation.available.run_pipeline --only retrieval --topics grownfield hydraulics --skip-synthetic
```

This requires datasets, indexes, model access, and relevant credentials; it is a
live integration experiment, not a documentation check. The consolidated
`--limit` flag applies to answer-correctness backfill, not retrieval.

The owning surfaces are `evaluation/available/retrieval/`, shared dataset and
matching helpers, and component/baseline runners. This guide intentionally
excludes result tables, plots, and selected winners.