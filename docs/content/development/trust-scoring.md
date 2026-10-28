# Trust Scoring Design

Trust scoring builds a query-independent document signal from content-quality
assessment, source metadata, and graph connectivity. Retrieval can then use it
to adjust candidate ranking. It is not a probability that a document or answer
is correct, and it does not replace query relevance or evidence inspection.

![Metadata prior, document-graph propagation, and retrieval adjustment](../assets/figures/trust/trust%20score.png)

*[Open the full-size trust diagram](../assets/figures/trust/trust%20score.png).
The diagram expresses the scoring design; artifact-path and ranking caveats
below describe the current implementation.*

## Two Different Graph Uses

| System | Nodes | Seeds | When calculated |
| --- | --- | --- | --- |
| Query-time PPR | Corpus hierarchy, chunks, and entities. | Question-matched entities and optional vector passages. | Per question. |
| Document TrustRank | Documents connected through shared entities. | Document metadata/content priors. | Offline artifact generation. |

Both propagate weight with restart, but their graphs, personalization, readout,
and intended meaning differ. Document TrustRank does not evaluate whether two
passages agree on a claim. Shared specific entities indicate topical overlap,
not factual corroboration or source independence.

## Artifact Flow and Source Scope

The intended preparation flow is:

```text
source metadata -> aligned metadata
                       + summary content scores
                   -> scored metadata -> prior
                                          + document connectivity
                                      -> final trust table
```

The production collector/alignment defaults are GROWNFIELD-specific. Metadata
collection follows SharePoint source items, including resolved shortcut targets;
alignment joins them to local corpus files. It is not a generic local-file
metadata extractor for every AVAILABLE data source.

The join key is a normalized lowercase filename stem. Duplicate source rows
prefer resolved metadata over unresolved shortcuts, then the most recently
modified row. Alignment retains only keys present in local storage and attaches
local path/topic. Local same-stem files are collapsed by the index's first entry;
stem uniqueness is an assumption, not a validated global identity guarantee.

### Current Path Compatibility Limitation

!!! warning "The default build paths are inconsistent"
    Collection, alignment, TrustRank, and the default serving lookup use the
    `dc_available/data` artifact tree. The modules currently under
    `dc_available/trust_score/components/` calculate their data root relative to
    that deeper location and resolve under `dc_available/trust_score/data`
    instead. This affects content scoring, score merging, and prior output.

The orchestrator calls these stages, but does not reconcile their paths. Do not
assume an unmodified end-to-end run produces a new serving table in the intended
location. Verify resolved inputs/outputs before regeneration. This guide records
the mismatch rather than changing the application or claiming that artifact
generation has been executed successfully.

`TRUST_FILE` selects the serving lookup table. It does not redirect all metadata,
summary, prior, and graph-build stages. A multi-source extension needs an explicit
artifact-path contract throughout the build as well as a per-source lookup path.

## Content-Quality Assessment

Text documents are assessed from their pipeline-generated summary, not the raw
document. The current model is `gpt-4o`, requested with temperature zero and JSON
output. Four equally weighted subscores are averaged:

| Criterion | What the rubric rewards |
| --- | --- |
| Technical depth | Concrete SAP specifics rather than general statements. |
| Process completeness | Prerequisites, followable steps, and an outcome. |
| Training usefulness | Actionable instructions sufficient to perform a task. |
| Focus | Coherent task coverage rather than mixed topics or visual filler. |

The prompt explicitly does not assess factual correctness. Scoring a generated
summary also inherits summarization omissions and errors. The rubric is tailored
to SAP S/4HANA; applying it unchanged to another domain changes what "quality"
means and may introduce systematic bias.

Numeric subscores are clamped to the range zero to one; missing/unparseable
subscores become zero. Videos and transcripts (`mp4`, `srt`) instead receive
`1.0` by policy without an LLM assessment. This is a declared preference, not a
measured finding that every video is high quality.

Existing successful scores are reused by document key. Error and missing-summary
rows are retried; changing the underlying summary, prompt, or model does not
automatically invalidate a successful cached score. Version those inputs when
rebuilding an evaluation or publishing a revised policy.

## Four-Pillar Prior

The prior is the weighted sum of four component scores:

| Pillar | Weight | Current scoring policy |
| --- | --- | --- |
| Content | `0.45` | Average of the four content subscores, or policy fallback. |
| Authority | `0.25` | Internal `1.0`, external `0.7`, unknown `0.3`. |
| Format | `0.15` | `mp4`, `srt`, `pptx`: `1.0`; `docx`: `0.7`; `pdf`: `0.6`; `url`: `0.5`; other: `0.6`. |
| Freshness | `0.15` | Exponential age decay with a three-year half-life and `0.3` floor; missing/unparseable date: `0.4`. |

Authority checks creator first, then modifier if creator is absent. A nonempty
name containing `EXTERNAL`, case-insensitively, is external; other nonempty names
are classified internal. This is a naming heuristic, not verified organizational
identity, expertise, or document approval status.

Freshness uses modification time relative to scoring time. Future timestamps
are treated as age zero. Three-year half-life means an age of three years gives
`0.5` before the floor. Comparable reruns should record their reference date;
recency alone is not evidence that a source is more correct.

!!! important "Missing content does not receive a conservative prior"
    Missing summaries and assessment failures produce a null content score.
    The prior builder replaces a missing/null content score with `1.0`.
    Consequently, absence of an assessment can receive the maximum content
    contribution. Track the content-source status separately from its final
    numeric prior when auditing coverage.

With valid component values and weights summing to one, the intended prior range
is zero to one. The prior builder does not provide general validation of every
externally edited numeric field, so validate imported tables before publication.

## Construct the Document Graph

Build document/entity membership from Chunk-to-Entity mentions, using each
chunk's owning document name normalized to lowercase. Repeated mentions within
a document become one membership. Typed Entity-to-Entity relations and hierarchy
links do not directly become TrustRank edges.

Let `N` be the number of documents with entity membership, and `df` the number
of those documents mentioning an entity. Retain entities with `df >= 2` and
`df <= 0.5 * N`. Each retained entity adds weight `log(N / df)` to every pair of
documents sharing it. Contributions from different entities sum. This IDF is
based on **document frequency**, unlike retrieval seed specificity based on
chunk frequency.

The result is a symmetric, nonnegative document adjacency. Column-normalize by
weighted degree to obtain its transition matrix. The current implementation
uses a dense matrix, suitable for its original small corpus; a much larger
corpus needs memory/complexity measurements and likely sparse construction.
Pair generation also grows quadratically with an entity's document membership.

## Propagation and Final Blend

Normalize document priors over graph documents to form seed vector `p`.
Graph documents without matching prior keys get zero seed weight; fail if total
seed weight is zero. Start at `p`, then repeatedly combine:

- `0.5` times the transition matrix applied to the previous trust vector.
- `0.5` times the prior seed vector.

Stop below L1 change `1e-6` or at `100` iterations. Zero-degree columns are not
redistributed. An isolated document that is present in the entity-derived graph
can therefore behave differently from a document entirely absent from it.

Divide propagated values by the largest propagated value to create the
relative `trust` signal. It is maximum-normalized, not a calibrated probability
or a stable absolute scale across corpus versions. The final score is the sum
of `0.7` times the prior and `0.3` times this graph signal.

Documents outside the entity-derived graph keep `trust = prior`, so the blend
reduces to their original prior. This is not a penalty for being disconnected.
For every published row, preserve `in_graph`, raw propagation, normalized trust,
prior, final trust, and content-source fields so consumers can distinguish
measurement from fallback.

## Lookup and Query-Time Ranking

The serving lookup tries the normalized owning document name, then the storage
path's filename stem. An unmatched key receives the table mean. A missing/empty
table gives a default of `1.0`, so that value is not proof of a successful match.
Malformed JSON or invalid row values are not covered by this missing-file
fallback and can still fail loading.

Tables are cached by file path, up to eight entries. Replacing a table at the
same path does not automatically invalidate the cached contents; restart the
serving process or provide an explicit invalidation/version policy.

For a candidate base relevance score, serving applies the multiplier
`1 + TRUST_BETA * final_trust`, with beta currently `0.5`, then sorts and truncates.
For valid trust in zero to one, the multiplier ranges from `1.0` to `1.5`.
It boosts more-trusted candidates relative to others rather than subtracting a
fixed penalty from less-trusted ones. It cannot retrieve a document absent from
the candidate pool.

In the current default **union fusion without reranking**, base scores are zero,
so this multiplier cannot change order. With nonzero RRF or cross-encoder scores
it can change selection. See [Hybrid Retrieval Design](hybrid-retrieval.md#trust-adjustment-and-the-no-reranker-case)
before interpreting a "trust enabled" experiment.

## Extending and Validating

Keep source identity, content quality, graph connectivity, and lookup coverage
separate in reports. Evaluate prior ranking against independent human judgments,
then isolate propagation and query-time adjustment in separate ablations. Hold
candidate pools fixed when testing trust's ranking effect and record fallback
rates, document coverage, modality, and domain.

Version metadata snapshots, summaries, model/rubric, component weights, graph
artifacts, scoring date, and published table together. Test missing summaries,
unknown creators, malformed/undated timestamps, duplicate stems, isolated
documents, missing tables, and same-path cache replacement. New data sources
need a domain-appropriate rubric and explicit authority/format policies rather
than inheriting SAP/video preferences silently.

Preparation ownership is `dc_available/trust_score/` and its `components/`
subdirectory. `trustrank.py` owns document propagation, `final_trust.py` owns
publication/lookup, and the Q&A path owns the ranking multiplier. Operational
entry points are described in [Pipeline Operations](../operations/pipeline.md).