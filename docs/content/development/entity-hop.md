# Entity-Hop Retrieval

Entity-hop retrieval is the alternative graph engine selected by
`GRAPH_RETRIEVAL_MODE=entity_1hop`. It scores passages reached through matched
entities rather than diffusing mass across the complete semantic/structural
graph. Its single-hop, multi-hop, and module-local variants are distinct from
[PPR](ppr.md).

## Shared Entity Entry Points

Current `GRAPH_ENTITY_HOP_SEED_MODE=merge` reuses PPR's threshold, backstop,
boilerplate filtering, and exact-label concept grouping. Group weight is split
among member nodes. Shared topic isolation filters seed entities to the selected
topic graphs. With no kept entities, this engine abstains; it does not accept
vector passage seeds as an alternative starting point.

The `legacy` seed policy keeps hybrid-ordered matches above the configured
signal floor (`0.5`), capped at eight. It rejects the query if none remain.
If more than `1%` of fetched entities lack IDF, it raises a data-contract error
rather than silently trusting the index. The current signal selector is `max`,
using the maximum available hybrid/dense signal. Unsupported query-phrase modes
fall back to the raw query rather than enabling a separate implemented parser.

## Single-Hop Passage Scoring

For each retained entity, find the distinct passages linked to it by mentions
in its own topic. Add the entity's seed weight to each reached passage. Evidence
from different seed entities accumulates; repeating an edge to the same entity
does not create another contribution for that seed.

For example, if seeds have weights `0.8` and `0.4`, a passage mentioning both
gets raw weight `1.2`. A passage mentioning only the first gets `0.8`. These are
query-specific relevance weights, not document trust or probabilities.

Normalize each raw passage weight by the square root of the passage's **total**
entity count, including entities not matched by the question. This reduces the
advantage of broad, entity-heavy passages. In the example, a passage with four
entities changes from `1.2` to `0.6`; a two-entity passage changes from `0.8` to
about `0.566`. If the count is zero, retain the raw score.

## Document Prior and Selection

Within each topic/document pair, take the best normalized passage score (MaxP).
Divide it by the strongest document's MaxP to produce a relative document prior.
Each passage is multiplied by `1 + GRAPH_DOC_PRIOR_BETA * prior`, currently with
beta `0.3`. This is a relevance-derived boost, **not** metadata trust or TrustRank.
Using max instead of sum avoids rewarding a document solely for having more
retrieved passages.

Sort the adjusted passages, apply the relative floor (`0.1` of the top score),
cap each topic/document at two passages, and retain up to thirty candidates.
If the best seed signal is below `0.6`, use the three-passage fallback instead.
The hybrid caller then applies its own graph-leg limit, currently ten.

The graph result's internal document ranking uses pre-prior MaxP scores. The
final hybrid recommendation list is separately derived from selected answer
passages. Those two lists can legitimately differ.

## Bounded Multi-Hop Variant

`GRAPH_ENTITY_HOP_K=1` is the current default. Larger values add rounds of:

```text
matched entity -> passage                         first hop
reached passage -> its entities -> new passages   additional hop
```

This is expansion through the mentions bipartite graph. It does not mean
following the LLM's typed Entity-to-Entity relations as a logical chain.

Each additional round propagates the source passage's raw weight multiplied by
`GRAPH_ENTITY_HOP_DECAY`, default `0.5`. If several routes reach the same new
passage during a round, keep the strongest gain rather than summing all paths.
Already-seen passages are not rescored. Entities are tracked by topic/identifier
and expanded once across the extra rounds. Stop early when a round adds no new
passages.

`GRAPH_ENTITY_HOP_HUB_CAP` can skip entities reaching too many passages during
additional hops. Its current value `0` disables that cap; it is not an active
protection in the default configuration. After expansion, the same
normalization, document-prior, and selection rules apply.

More hops increase coverage but can introduce generic or neighboring procedures.
Compare recall gains with wrong-document hits and answer quality instead of
assuming longer traversal is better.

## Optional Module-Local Expansion

`GRAPH_HIERARCHICAL_ENABLED=False` currently disables this variant. When enabled,
module evidence and confidence checks can select a same-module expansion path:

1. Choose promising modules and seed documents from direct passage evidence.
2. Optionally boost document selection with query/title term overlap.
3. Inspect passages in those documents and expand through matched entities and
   explicit chunk links, staying inside the module.
4. Normalize candidate scores within each module before combining them.
5. Apply the usual passage normalization, document prior, and cutoffs.

If local expansion produces nothing, retain the direct path. This is a
module-constrained heuristic, not PPR over hierarchy nodes. Its module inference
depends on document naming, so a new corpus naming convention needs validation
before enabling it. Disabling the feature does not imply all module diagnostics
are absent from the request path.

## Engine Comparison and Extension

| Property | Entity hop | PPR |
| --- | --- | --- |
| Main evidence path | Mentions, plus optional bounded/local expansion. | Weighted semantic and structural adjacency. |
| Vector passage seeding | Not used in the standard engine. | Optional in hybrid mode. |
| Score normalization | Total-entity square root and MaxP prior. | Per-node weighted transition normalization. |
| Additional reach | Explicit hop rounds or module-local policy. | Repeated diffusion with restart. |
| Main risk | Generic entities and expansion drift. | Seed/topic errors, graph weighting, and diffusion drift. |

For a fair comparison, hold seed mode, topic isolation, corpus, and output caps
fixed. Entity-hop versus hybrid PPR also differs in passage-seed availability.
Record fallback and abstention rates and inspect matched-but-empty results.

Implementation ownership is `dc_available/graph/retrieval/entity_hop_retriever.py`;
the shared seed and topic policies are also used from `ppr_retriever.py`. Preserve
the [graph result contract](graph-retrieval.md#downstream-contract) when adding a
new engine so fusion and evidence hydration remain compatible.