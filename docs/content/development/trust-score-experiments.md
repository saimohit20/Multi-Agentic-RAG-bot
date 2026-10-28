# Trust Score Experiments

Trust evaluation separates score sensitivity from downstream usefulness. The
first experiment tests controlled quality degradations; the second compares
RAG behavior with trust disabled, prior-only trust, and full trust. This guide
documents experimental procedures without presenting outcomes.

See [Trust Scoring Design](trust-scoring.md) for prior components, document
TrustRank, lookup behavior, and query-time ranking adjustment.

## Why Controlled Variants

The source corpus is approved training material rather than a naturally labeled
mixture of reliable and unreliable documents. Controlled variants provide known
interventions: the experiment records which source was changed, which property
was targeted, and which original serves as its matched control.

This establishes ground truth about the intervention, not universal ground truth
about document reliability. Summary corruption, altered metadata, and injected
fact conflicts test specific assumptions; they do not represent every real-world
source failure.

## Experiment 1: Score Sensitivity

### Sampling and Pairing

The configured sample target is forty documents with `content_source=llm` and a
resolvable summary. Videos/transcripts with fixed policy content scores are not
eligible. Candidate lists are shuffled using seed `42`, then sampled round-robin
across topic groups until the target is reached or candidates are exhausted.
This balances topic coverage rather than sampling proportional to corpus size.

Retain the original metadata, content subscores, source key, and summary path.
Each document has one original and five independently targeted variants; compare
every variant to its own original rather than to an unrelated document.

### Controlled Changes

| Variant | Intervention | Targeted component |
| --- | --- | --- |
| Procedural | Remove procedural detail from the summary. | Process completeness; training usefulness is a secondary check. |
| Technical | Strip domain-specific technical detail. | Technical depth. |
| Filler | Introduce off-topic content. | Focus. |
| Authority | Override source-authority metadata. | Authority score. |
| Freshness | Backdate modification metadata by four years. | Freshness score. |

Content variants use `gpt-5.2`; metadata changes use deterministic overrides.
The content assessor is the production `gpt-4o` scorer. Separating corruption and
scoring models avoids using an identical model for both roles, but does not
remove shared model biases or prove that every generated perturbation succeeded.
Retain changed summaries and intervention manifests for inspection.

Perturbations operate on the summary representation consumed by trust scoring,
not raw source files. This isolates the scorer but does not test extraction or
summarization robustness. The design uses one intervention severity per type,
not a dose-response study.

### Rescoring and Controls

Content variants are passed through the production content scorer. Original and
metadata variants reuse baseline content subscores. Metadata pillars and the
weighted prior are recomputed using production formulas and a common reference
time for newly scored rows. Freshness comparisons therefore share a date within
a scoring pass.

The output preserves the four content subscores, content aggregate, authority,
format, freshness, and prior. Document TrustRank and `final_trust` are not
recomputed in this experiment: this is a prior/component-sensitivity test, not a
test of graph propagation or answer quality.

For each paired variant, calculate the changed score minus its original score.
Inspect the targeted component first, the overall prior second, and off-target
components as a specificity check. A weighted prior can change less than its
targeted component by construction.

### Paired Statistical Analysis

For each of the three content interventions, the analysis calculates:

- The count/proportion of paired documents whose targeted score decreases.
- Mean and median paired differences.
- A percentile 95% confidence interval for the mean difference, from 10,000
  bootstrap resamples with seed `42`.
- A one-sided paired Wilcoxon signed-rank test with the alternative that the
  perturbed score is lower than the original.
- Paired Cohen's d: mean difference divided by the sample standard deviation
  of differences.
- Benjamini-Hochberg adjustment across the three content-intervention tests.

Authority/freshness overrides are deterministic formula changes and are not
included in those significance tests. The implementation uses zero-difference
handling from SciPy's Wilcoxon test, returning a neutral p-value if that call
raises for a degenerate case. Preserve eligible pair counts and invalid cases;
do not describe generated variants as independent samples of new documents.

## Experiment 2: RAG with Conflicting Sources

### Source Selection and Altered Twins

This experiment uses a locked, manually curated list of twenty fact-dense
Grownfield summaries, chosen for concrete codes, identifiers, and values rather
than broad overview prose. It is not a random corpus sample. The configured
design assigns alternating documents to freshness or authority interventions,
targeting ten originals per lever. Missing source files can reduce the realized
sample; retain the selection manifest rather than assuming the target was met.

For each original, `gpt-5.2` proposes ten distinct, checkable facts with a correct
value, plausible wrong value, question, and reference answer. Validation keeps
records whose correct value appears verbatim in the summary, whose replacement
differs and is absent, and whose question/reference are present. Duplicate
correct values are removed. Accepted counts need not equal the requested count.

The altered twin is made by deterministic, alphanumeric-boundary-aware
replacement of accepted values in the original summary. Each accepted fact
becomes a question linked to the original and twin. Keep both versions and
inspect the replacements; model instructions and string checks do not guarantee
that every generated question has a uniquely valid answer.

Twin trust is lowered through the assigned metadata lever: backdate modification
time four years or use unknown authority. Original content quality and format
are retained in its prior. The corrupted text is **not** independently reassessed
for content quality in this experiment. This isolates a metadata-induced trust
difference rather than claiming the content scorer detected factual corruption.

### Trust Tables and Graph Assumption

Build two tables keyed by test document IDs such as `g01` and `g01_twin`:

| Table | Published lookup value |
| --- | --- |
| Prior only | The original or twin's prior. |
| Full trust | The prior/TrustRank blend for the original or twin. |

For full trust, derive twin entity membership from its original's graph entities,
removing entities matched to changed correct values. Novel wrong values are not
added as extracted graph entities. Inject the derived twins into an in-memory
copy of the broader production document graph, then run the production
transition and propagation algorithms.

Both originals and twins receive the extended-graph blend where graph membership
is available; absent graph documents fall back to their priors. This differs
from simply copying every original's previously published final score unchanged.
It also differs from fresh entity extraction over altered text. Shared entities
represent overlap, not a check that two sources agree on the changed facts.

### Index and Treatment Arms

The intended test index uses a separate Qdrant collection,
`available_trust_test`, and a separate document-store cache. It indexes original
and twin summary texts through the usual BGE-M3 dense/sparse representation and
splitter. Test IDs connect retrieved document metadata to the experimental
trust tables.

Use the same questions and index for all three arms:

| Arm | Treatment |
| --- | --- |
| Off | Disable trust adjustment. |
| Prior | Enable adjustment with the prior-only table. |
| Full | Enable adjustment with the prior/TrustRank table. |

Retrieval is vector-only because the isolated test corpus has no query-time
retrieval graph. Offline full-corpus TrustRank does not make this a hybrid
retrieval experiment. Generate an answer from each arm's selected passages using
the application's structured-answer path. Save source/chunk ordering, base and
adjusted scores, trust values, answer, contexts, reference, and changed values.

!!! warning "Verify effective configuration before execution"
  The arm runner inherits reranker and fusion settings. Under current union
  fusion without reranking, base scores are zero and multiplicative trust
  cannot reorder them. Record the actual scoring configuration instead of
  assuming that enabling trust produces a ranking intervention.

Collection names and retrievers are also cached/imported configuration. The
consolidated orchestrator imports index-building modules before the arm module's
collection environment override. Verify the resolved collection and cache before
search/build, preferably in a fresh dedicated process; an intended test collection
name is not proof of isolation. Index construction recreates its target
collection. This guide does not claim that a live isolated run was verified.

### Retrieval and Answer Outcomes

The binary retrieval outcome `good_doc` is true if the original appears and
ranks before the twin, or the twin is absent. It is false if the twin ranks
first or neither appears. Ranks stored in this experiment are zero-based passage
positions, unlike the one-based document rank in the general retrieval evaluator.
The source-comparison rule uses each source's first passage occurrence.

Generation is assessed by three complementary checks:

| Check | Procedure |
| --- | --- |
| Correct-value presence | Case-insensitive substring search for the authored correct value in the answer. |
| Dedicated LLM judge | `gpt-4o` at temperature zero judges whether the answer conveys the reference value; formatting differences are allowed, identifier/value changes are not. |
| RAGAS answer correctness | Separately score reference agreement and compare its judgments with the deterministic check. |

Wrong-value presence is tracked separately. Classify answers into correct only,
wrong only, both, or neither. Substring presence is not semantic entailment: a
negated value or a value inside a longer token can match. An answer mentioning
both values is not necessarily a correct resolution of the conflict.

For the grader-agreement comparison, RAGAS values are thresholded at `0.5` and
null evaluator outputs are excluded. Compare graders on common scored records,
preserving coverage. This is a metric-consistency check, not evidence that the
graders are statistically independent or a claim about their observed agreement.

### Paired Arm Comparisons

Compare off/prior, prior/full, and off/full on shared question IDs. The analysis
uses paired McNemar comparisons for source ordering and correct-value presence,
retaining both-success, each-arm-only success, and both-failure counts. It
reports a continuity-adjusted chi-square calculation and an exact binomial
p-value based on discordant pairs. No discordant pairs yield a neutral p-value.

The generation test uses correct-value presence, which includes the "both"
category; keep that separate from the stricter correct-only descriptive count.
The current arm analysis does not apply the sensitivity experiment's
Benjamini-Hochberg adjustment to its arm pairs. Questions from one original
document share content and are not independent document samples; retain document
grouping when interpreting question-level inference or extending the analysis.

## Experiment Boundaries and Reproducibility

Experiment artifacts are separate from production documents. Preserve the
sample, variant summaries/manifests, paired scores, effective model/rubric,
scoring date, and statistical settings. Resume behavior reuses existing variant
IDs rather than validating full input/model identity. Mixing resumed rows from
different scoring dates weakens the common-time freshness control.

Both trust suites depend on the production baseline table. Verify resolved
paths before running: the current component-relative artifact-path mismatch is
described in [Trust Scoring Design](trust-scoring.md#current-path-compatibility-limitation)
and also affects evaluation imports that derive paths from those components.

The sensitivity entry point is:

```powershell
poetry run python -m evaluation.trust_score.run_pipeline perturbation --no-plot
```

This still generates variants and calls the content-assessment model. Disabling
plots does not make the experiment offline or free of API costs. Implementation
ownership is `evaluation/trust_score/perturbation/`; the companion RAG comparison
is under `evaluation/trust_score/rag_ab/`.

For the RAG suite, the consolidated entry point is
`poetry run python -m evaluation.trust_score.run_pipeline rag_ab --no-plot`.
Resolve the isolation/path caveats above before using it. Its `--limit` option
caps arm execution and grading stages, not source selection, twin generation,
TrustRank preparation, or index recreation; it is not a side-effect-free dry run.

Keep the three arm files, both trust tables, dataset/twin manifests, grader
outputs, and effective settings together. Resume keys are question/arm IDs, not
hashes of answers or scoring policy. Null judge rows may be considered completed
by that cache. The consolidated suite runs generation grading before RAGAS, so
the unified grading file may not contain later RAGAS values automatically;
check both score artifacts before a grader-agreement analysis.

These experiments test controlled summary/metadata interventions in a selected
domain. They do not establish full-corpus accuracy, resistance to every source
attack, or the effect of trust in all graph/vector/frontend workflows.