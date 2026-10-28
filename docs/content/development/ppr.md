# Personalized PageRank Retrieval

PPR ranks passages by repeatedly propagating query-seeded weight through each
selected topic graph while restarting toward the question's evidence. It is the
default graph engine. This guide describes the implemented retrieval variant,
including departures from an idealized probability-preserving PageRank model.

## 1. Build the Weighted Retrieval Graph

Every stored node type participates: chunks, entities, documents, and hierarchy
nodes. Each accepted edge becomes two weighted adjacency entries, so retrieval
is **undirected** even when an extracted relation such as `requires` is stored
with a source and target. The walk uses connectivity, not directional logical
entailment.

| Edge family | Default weight | Treatment |
| --- | --- | --- |
| `mentions` | `1.0` | Connect chunk and entity. |
| Other semantic relation types | `0.8` | Multiply by stored confidence; missing/unparseable confidence falls back to `1.0`. |
| `related_to` | `0.5` | Represent explicit/sequence relationships. |
| `contains`, `part_of` | `0.3` | Connect corpus hierarchy nodes. |

Edges with unknown endpoints, self-links, and nonpositive weights are skipped.
Parallel edges accumulate. The structural builder writes both `contains` and
inverse `part_of`; symmetrizing both accumulates two structural contributions
for that pair. Edge-generation policy therefore affects transition strengths
even when family weights are unchanged.

For adjacency `A`, each nonempty column is divided by its total outgoing weight
to form transition matrix `S`. A node distributes its current weight across its
weighted neighbors. Broad hubs split their mass over more links, but graph
normalization does not guarantee that generic entities become harmless.

## 2. Select Entity Seeds

The default `merge` seed pipeline is:

1. Fetch up to `50` hybrid entity matches with joined dense scores.
2. Keep matches whose maximum hybrid/dense signal is at least `0.6`.
3. If fewer than `5` survive, replace that set with the best available five by
   signal. The threshold is therefore not a strict abstention boundary.
4. Remove entities mentioned in more than `150` chunks, but retain the original
   set if filtering would remove everything.
5. Group entities by lowercased, whitespace-normalized label, ignoring entity
   type for this query-time grouping.

A member's raw weight is signal times IDF. A group's weight is its strongest
member's weight, not the sum. Divide the group weight equally across its member
entity nodes. This lets same-label fragments contribute their linked evidence
without multiplying seed mass merely because extraction created several types.
It is exact normalized-label grouping, not semantic synonym resolution.

The `legacy` mode instead uses the original confidence floor and capped
hybrid-ordered seed list. Its strict missing-IDF guard and abstention policy are
not identical to merge mode. Evaluate a seed-mode change independently of an
engine change.

## 3. Choose Topics and Form Personalization

Select topic graphs using the shared isolation policy described in
[Graph Retrieval Design](graph-retrieval.md#topic-selection). For each topic,
keep only seed identifiers that exist in that graph with positive weight.

Vector hits can additionally seed Chunk nodes. They use the actual vector-leg
scores, which may be dense/sparse fusion scores, not necessarily raw dense
cosines. Only positive scores whose identifiers exist in the topic graph
contribute.

When both seed families exist, the default personalization mass is:

- `80%` on entity nodes, normalized within that group.
- `20%` on vector-retrieved passage nodes, normalized within that group.

`PPR_PASSAGE_SEED_WEIGHT` controls the passage share. If only one family maps
into a topic, it receives the full mass. With no valid seed in a topic, skip it.
With no entity matches but vector hits available, PPR can run passage-only.
The personalization vector `e` is normalized to sum to one.

This split is per topic. Cross-topic strength is calculated separately from raw
entity seed weight, falling back to raw passage weight only in a topic without
entity seeds. It is not simply an equal allocation among selected topics.

## 4. Iterate with Restart

Start the rank vector at `e`. Each iteration forms a new vector from:

- `alpha` times the propagated previous vector, `S` multiplied by that vector.
- `1 - alpha` times the personalization vector `e`.

With `PPR_ALPHA=0.5`, each update combines equal propagated and restarted
contributions. Lower alpha keeps ranking closer to seeds; higher alpha increases
the opportunity for indirect evidence and drift. This alpha is the propagation
coefficient, **not** the restart probability.

Stop when the sum of absolute changes is below `1e-6`, or after `100`
iterations. The current implementation does not publish an explicit convergence
status, so reaching a result is not proof that the tolerance was met.

!!! note "Dangling nodes"
    Zero-degree columns remain zero. Their outgoing mass is not redistributed
    back to the seed vector, and the final rank vector is not renormalized.
    Mass can therefore be below one. Treat output as retrieval weights, not
    calibrated probabilities or a guaranteed textbook stationary distribution.

## 5. Combine Topics and Read Out Passages

Multiply each topic's chunk-node rank by that topic's share of raw seed strength.
Retain positive scores. If a chunk identifier appears in more than one topic,
the larger weighted score is kept rather than summed. Shared identifiers must
still denote the same passage.

The passage list is then processed in this order:

1. Sort descending by score.
2. Remove scores below `0.1` times the best passage score.
3. Keep at most `2` passages per owning document.
4. Keep up to `30` passages in normal mode, or `3` in low-confidence fallback.

Fallback is selected when the best entity signal is below `0.6`, or when no
entities were available. The outer hybrid path subsequently takes only
`FUSION_GRAPH_K=10` graph passages. A `30`-passage graph result is not a
`30`-passage contribution to the default hybrid pool.

PPR does not apply the entity-hop engine's square-root entity-count normalization
or MaxP document-prior boost. Its graph-internal document ranking uses the best
pre-cut passage score. The current diversity/readout key is document name rather
than a topic-qualified key, so same-named documents across topics need care.

## Worked Intuition

For a question about a specific procedure, entity matching might identify its
transaction and process. Restart repeatedly places weight on those concepts;
mentions carry it to explanatory passages. Semantic relations can connect a
prerequisite concept, while hierarchy edges distribute some weight through the
owning document and training. A vector-matched passage can anchor the walk even
when entity extraction missed part of the wording.

The useful outcome is a better candidate passage, not a generated reasoning
chain. The language model still needs evidence-bearing text, and graph
connections alone do not verify a claim.

## Tuning and Extension

Inspect seed labels, IDF, topic decisions, landed seed mass, passage coverage,
and relative-cut survivors before tuning alpha. Compare entity-only versus
mixed seeding separately from edge weights. Changing relation directionality,
dangling-mass treatment, duplicate-edge handling, or document keys changes the
algorithm and requires quality/latency regression measurements.

The matrix is sparse and cached per topic/weight configuration; iteration cost
depends on active graph edges and iteration count. Cold loading and repeated
entity searches can dominate small walks. Version caches when implementing
hot-reload; the current cache does not include artifact identity.

Implementation ownership: `dc_available/graph/retrieval/ppr_graph.py` assembles
the retrieval matrix; `ppr_retriever.py` handles seeds, topic runs, and readout.
These are subsystem boundaries rather than a code walkthrough.