# Generator Evaluation

Generator evaluation examines answers produced from retrieved evidence. It
separates reference-answer correctness from support in the supplied context,
and keeps a record of the question, context, answer, and evaluator configuration.
This guide describes the process without reporting scores or conclusions.

## From Retrieval to Answers

The standard real-question and synthetic generation passes read saved retrieval
records and select hybrid rows. They reuse `retrieved_contexts` rather than
retrieving again. This makes the evidence used for generation traceable to the
[retrieval evaluation](retrieval-evaluation.md).

Contexts are numbered and concatenated into a prompt that asks for a concise,
self-contained answer based only on the supplied evidence. The standard
generation model is `gpt-4o-mini`, despite its configuration constant being named
`JUDGE_MODEL`. The evaluation prompt is not a claim that every frontend,
curriculum, or multi-turn workflow was exercised.

No-context rows retain an empty answer. Generation exceptions also produce an
empty answer. Successful cached answers are reused by question ID. Changing
contexts, prompt, or model does not automatically invalidate that cache; use
versioned inputs and output locations to avoid mixing experiments.

The standard generation pass evaluates hybrid answers only. Leg-comparison and
reranker studies have their own orchestration. Do not describe a hybrid-only
generation file as an answer comparison across all retrieval configurations.

The dedicated final-system runner instead retrieves and generates together with
fixed hybrid/PPR-merge retrieval and reranking off, then reuses the standard
RAGAS scorer. It records both stage durations. Keep those records separate from
post-reranker leg studies and from reranker-study answers produced with the
application QA chain.

For synthetic runs, verify the input artifact explicitly: the generation reader
expects `retrieval.json`, while the current leg evaluator writes separate
per-leg files. The [retrieval guide](retrieval-evaluation.md#reproducibility)
describes this handoff limitation. Do not silently score a stale combined file
against a newer retrieval configuration.

## Reference-Based LLM Judge

The declared evaluation protocol uses a reference-based LLM judge as the primary
answer-correctness assessment. The judge receives the question, reference
answer, and generated answer; this assessment does not itself establish that
the answer was supported by retrieved context.

The reusable judge uses `gpt-4o`, temperature zero, and structured verdicts:

| Verdict | Rule | Numeric encoding |
| --- | --- | --- |
| Correct | Required reference facts are present and accurate, without contradictions. | `1.0` |
| Partial | The core is correct, but required facts are missing or imprecise. | `0.5` |
| Incorrect | A key fact is wrong, absent, or contradicts the reference. | `0.0` |

Extra detail is not penalized merely for being longer; incorrect or contradictory
detail still matters. When a reference requires a transaction code, identifier,
or exact value, the rubric requires that exact item. A correct procedure that
omits the required identifier can be partial; an incorrect identifier is incorrect.

The saved record includes the verdict and a short rationale. Empty generated or
reference answers and judge failures receive null scores, not an automatic
incorrect label. Aggregation averages non-null judge scores. Always report
scoring coverage separately so skipped failures are not mistaken for successful
answers or silently compared with a different denominator.

The standalone judge entry point expects the reranker-study configuration files.
Other experiment runners can reuse its scoring components. Selecting the
consolidated generation stage does not by itself run every judge comparison.

## RAGAS Assessment

RAGAS adds complementary diagnostics using question, generated answer, ordered
contexts, and reference answer. Context metrics assess the evidence pipeline as
well as the generator; they are not pure measurements of language-model ability.

| Metric | Assessment focus |
| --- | --- |
| Faithfulness | Whether answer claims are supported by the supplied contexts. |
| Answer relevancy | Whether the answer addresses the question. |
| Context precision | Whether relevant context is prioritized in the retrieved list. |
| Context recall | Whether reference-answer information is covered by retrieved context. |
| Answer correctness | Agreement with the reference through factual and semantic assessment. |

Faithfulness and correctness are different: an answer can faithfully repeat a
bad source, or contain a correct fact unsupported by its supplied context.

### Runner-Specific Metric Sets

| Evaluation path | Evaluator configuration | Metric set |
| --- | --- | --- |
| Standard real/synthetic RAGAS pass | `gpt-4o-mini` with project BGE-M3 embeddings. | Faithfulness, answer relevancy, context precision, context recall. |
| Reusable LLM-judge companion scoring | `gpt-4o` with project BGE-M3 embeddings. | Faithfulness, answer relevancy, context precision, answer correctness. |
| Answer-correctness backfill | `gpt-4o` with project BGE-M3 embeddings. | Answer correctness on previously generated answers. |

The backfill does not retrieve or generate again. It uses RAGAS's default
answer-correctness weighting: `0.75` factual and `0.25` semantic. Record the RAGAS
version and actual metric configuration; do not treat context recall and answer
correctness as interchangeable fourth metrics or compare judge-model changes
as though the evaluator were held fixed.

The standard RAGAS pass excludes rows without answers or contexts, and averages
available values per metric. Companion scoring and correctness backfill also
require a reference answer. The standard pass can supply an empty reference
when one is absent; reference-dependent metrics then need explicit validity
review. Preserve per-metric coverage and nulls, not just aggregate means.

## Human Review of the Judge

The evaluation plan calls for human validation of automated verdicts through
label agreement and Cohen's kappa. Use the same correct/partial/incorrect rubric
and pair each human label with the same question, reference, generated answer,
and system configuration. Kappa adjusts observed agreement for agreement
expected from the two label distributions.

The current worksheet generator takes the first records from each configuration
to approximate balanced coverage; it is not random or stratified sampling by
difficulty/verdict. It includes the automated verdict and rationale, so review
of that sheet is not blinded. Record this limitation or create a separate
blinded review copy for an independent validation protocol.

Creating a worksheet does not establish that labels were completed or agreement
was calculated. Preserve the completed annotations, sampling rule, label rubric,
and agreement procedure before describing human validation as performed. This
guide does not report any agreement values.

## Controlled Component Comparisons

The reranker study compares no reranker, `BAAI/bge-reranker-base`, and
`BAAI/bge-reranker-v2-m3` while fixing hybrid/PPR-merge retrieval policy. It
uses the application QA chain rather than the standard saved-context prompt.
Avoid attributing cross-run differences solely to the reranker if generation
model, prompt, trust, or candidate limits also changed.

That runner resets the reranker singleton between configurations and performs
an untimed warm-up to separate loading from steady-state measurements. It
measures retrieval, cross-encoder, and generation timing. Warm-up does not
eliminate every cache/order effect; record hardware, model state, and execution
order. Missing parsed stage timings must not be interpreted as measured zero
latency.

For baseline comparisons, align question IDs, references, generation settings,
judge/rubric, and RAGAS definitions. Reusing the same metric name alone is not
enough when evaluator models or eligible-row filters differ.

## Reproducibility and Entry Points

Retain retrieved context in its original order, generated answers, reference
answers, prompt/model settings, judge verdicts/rationales, per-metric scores,
errors, eligible-row counts, and configuration manifests. Successful answer and
correctness-score caches are keyed by IDs rather than full experiment identity.
Use separate versioned outputs when changing the experiment.

The standard generation/scoring stage can be invoked with:

```powershell
poetry run python -m evaluation.available.run_pipeline --only generation --topics grownfield hydraulics --skip-synthetic --skip-answer-correctness
```

This still generates answers and makes evaluator calls; it is not an offline
validation command. Correctness backfill and judge studies are separate stages
and can incur substantial API/model costs. Do not overwrite reference runs.

Implementation ownership is `evaluation/available/generation/`,
`evaluation/available/llm_judge.py`, and the component/baseline runners. These
boundaries identify the experiment surfaces, not a function walkthrough.