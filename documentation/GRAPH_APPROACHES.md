# Graph-Based Retrieval Approaches — Literature-Review Source Material

> **Purpose of this file.** This is *source material* for the literature-review chapter of the
> thesis, not an implementation log. It records (a) the graph-retrieval **approaches actually
> tried** in this work, (b) the **published methods** each one derives from or relates to, and
> (c) how this thesis **positions itself** against that literature. It is written to be handed to
> an LLM as grounding when drafting the review.
>
> **⚠ Citations must be verified.** Every reference below is marked `[ref: …]`. Author/year/venue
> are given from memory and **must be checked against the primary source before they go into the
> thesis.** Do not let a drafting LLM invent or "fix" a citation from this list — verify it.
>
> **⚠ Numbers are point-in-time.** The empirical figures in §5 come from a small (17-question)
> internal eval and a one-off diagnosis run; regenerate them from the current evaluation harness
> before quoting them in the final text.

---

## 1. Setting and motivation

The retrieval target is an **enterprise process handbook** (Bosch/SAP-style business-process
documentation — Order-to-Cash, Purchase-to-Pay, Record-to-Report, etc.), ingested as text/video
chunks, embedded, and served by a **dense vector leg** (Qdrant + a cross-encoder reranker,
`bge-reranker-v2-m3` `[ref: Chen et al. 2024, BGE-M3 / BAAI]`). Dense retrieval is the working
baseline `[ref: Lewis et al. 2020, RAG; Karpukhin et al. 2020, DPR]`.

Graph retrieval was explored as a **second, parallel leg** to capture what pure vector similarity
misses: (i) *entity-level* connections between chunks that share a specific concept, and (ii) the
*document/module structure* of the handbook. The motivating hypothesis (from the graph-RAG
literature) is that propagating relevance over an explicit knowledge structure can reach the right
passage when it is a hop away from the query's own words, and can keep activation inside the
correct process area.

### 1.1 The shared graph substrate (what every approach below runs on)

All approaches operate on one graph built at ingestion time (`graph_schema.md`,
`dc_available/graph/`). Its layers:

- **Structural / hierarchy layer** — `Topic → Subtopic → Level → Training → Document → Chunk`,
  connected by `part_of` / `contains`. This encodes the handbook's table-of-contents structure.
- **Semantic layer** —
  - `Chunk —mentions→ Entity` (entities extracted per chunk; the core signal),
  - `Entity ↔ Entity` **typed relations** (~70 LLM-extracted relation types: *requires*,
    *precedes*, *prerequisite_for*, *leads_to*, …), enabling multi-hop reasoning,
  - `Chunk ↔ Chunk` **sequence** links (`related_to`, from next/previous/related metadata).
- **Evidence layer** — `supports` / `contradicts` edges and a per-node `trust_score` field
  (present in the schema; used by the trust extension in §6, not by the retrievers in §3).

Every entity carries an **IDF** (inverse document/chunk frequency) value — a specificity weight
`[ref: Spärck Jones 1972]` — which several approaches use to suppress generic hub concepts.

---

## 2. The graph-RAG landscape (related work)

A compact map of the field the thesis sits in. Each entry names the mechanism and the property the
thesis borrows or contrasts with.

| Family | Representative work | Core mechanism | Relation to this thesis |
|---|---|---|---|
| Vanilla / dense RAG | `[ref: Lewis et al. 2020]`, `[ref: Karpukhin et al. 2020, DPR]` | embed query + passages, cosine top-k | the **baseline leg** the graph legs must beat |
| Community / global GraphRAG | `[ref: Edge et al. 2024, Microsoft "From Local to Global"]` | build KG, detect communities, hierarchically summarise for global queries | analogue of the **hierarchical/module-scoped** approach (§3.3) |
| PPR over a KG | `[ref: Gutiérrez et al. 2024, HippoRAG]` | OpenIE KG + **Personalized PageRank** seeded on query entities, weighted by node specificity | the **direct basis** of the PPR approach (§3.4) |
| PPR + dense integration | `[ref: Gutiérrez et al. 2025, HippoRAG 2]` | adds dense passage nodes / "recognition memory" to the PPR walk | basis of the **dense passage-seed** variant (§3.4) |
| Dual-level graph+vector | `[ref: Guo et al. 2024, LightRAG]` | low-level (entity) + high-level (theme) keyword retrieval over a graph, incremental updates | contrast: not adopted; noted as an alternative routing scheme |
| Hierarchical summary tree | `[ref: Sarthi et al. 2024, RAPTOR]` | recursive clustering + summarisation into a tree, retrieve across levels | conceptual analogue of the hierarchy layer / module scoping |
| KG-guided chunk organisation | `[ref: KG²RAG — verify authors/year]` | use a KG to expand and **organise** retrieved chunks (e.g. via an MST) | parked; cited as related structuring idea |
| Subgraph retrieval w/ GNN | `[ref: He et al. 2024, G-Retriever]` | PCST subgraph extraction + GNN for graph QA | contrast: heavier, learning-based; not used |
| Trust / claim propagation | `[ref: ClaimTrust — verify]` | propagate reliability over a claim graph | basis of the **trust-score** future work (§6) |
| Ranking foundations | `[ref: Page et al. 1999, PageRank]`, `[ref: Haveliwala 2002, Topic-Sensitive PageRank]` | stationary distribution of a random walk; **personalised** teleport vector | the math underneath every PPR variant |

---

## 3. Approaches implemented and evaluated in this thesis

Four core approaches were built. Three (§3.1–3.3) share one engine
(`entity_hop_retriever.retrieve`, mode `entity_1hop`); the fourth (§3.4) is a separate PPR engine
(`ppr_retriever.ppr_retrieve`, mode `ppr`). A `GRAPH_RETRIEVAL_MODE` switch selects the engine so
they can be compared head-to-head; PPR is the **current default**.

All approaches return the **same `GraphResult`** and plug into the same downstream:
**union fusion** with the vector leg → **cross-encoder rerank**. The legs differ only in *how they
score chunks from the graph*.

### 3.1 Entity-seeded 1-hop retrieval — *the baseline graph leg*

- **Idea.** Match the query to seed entities, then take a single hop to the chunks that mention
  them.
- **Mechanism.** Query → seed entities (kept above a confidence floor, capped in number), each
  weighted `signal × idf`; **one hop** `entity —mentions→ chunk` accumulates weight per chunk;
  `√(entity-count)` length normalisation; a **MaxP** document prior (a document scores as its best
  chunk); score floor + per-document cap + top-k.
- **Graph used.** Semantic `mentions` layer only. The structural hierarchy is built but ignored.
- **Published analogue.** The classic entity-linking / one-hop KG-RAG pattern; the "local"
  precursor to community GraphRAG `[ref: Edge et al. 2024]`.
- **Known limits (motivating the later approaches).** Generic, low-IDF entities ("billing",
  "sales order") route to many sibling documents; a single hop cannot reach a passage that sits a
  hop away; MaxP favours long documents over thin overview pages.

### 3.2 Entity-seeded k-hop traversal — *multi-hop extension of §3.1*

- **Idea.** Let relevance propagate further than one hop.
- **Mechanism.** Same seeding, but iterate `chunk → its entities → more chunks` for `k` hops
  (`GRAPH_ENTITY_HOP_K`), scaling each additional hop by a **decay** factor
  (`GRAPH_ENTITY_HOP_DECAY`), skipping generic **hub** entities above a degree cap
  (`GRAPH_ENTITY_HOP_HUB_CAP`), and deduplicating so each chunk is scored once. `k = 1` is
  byte-identical to §3.1.
- **Published analogue.** Multi-hop / path-based graph traversal for retrieval; a discrete,
  decay-weighted approximation of what PPR does continuously.
- **Purpose in the study.** Tests whether reach (not just seeding) is the bottleneck, and provides
  a fair multi-hop non-PPR comparison point.

### 3.3 Hierarchical (module-scoped) expansion — *structure-first variant*

- **Idea.** Use the handbook's structure: find the right **module**, then retrieve within it.
- **Mechanism.** Rank modules for the query; if a module clearly wins
  (`hierarchical_would_switch` gate on score/ratio thresholds), expand within that module's
  documents (`part_of`/`contains`), with a **query-title overlap boost**
  (`GRAPH_HIERARCHICAL_DOC_QUERY_BOOST`) and a cap on documents/chunks kept. Currently an optional
  path (`GRAPH_HIERARCHICAL_ENABLED`).
- **Graph used.** Structural hierarchy layer.
- **Published analogue.** Community/global GraphRAG `[ref: Edge et al. 2024]` and hierarchical
  tree retrieval `[ref: Sarthi et al. 2024, RAPTOR]` — retrieve by first localising to a region of
  the corpus.
- **Purpose in the study.** Directly exploits the structure the baseline ignores; a control for
  "does knowing the right module help?"

### 3.4 Personalized PageRank (HippoRAG-style) — *the main approach / current default*

- **Idea (water & pipes).** Pour "water" at the query's seed nodes; let it flow through the
  graph's pipes; retrieve the chunks where the most water pools. How much water each seed gets is
  its **IDF** (specific concepts get a full bucket, generic ones a few drops); a node with many
  pipes splits its water thinly, so hubs can't flood the graph; structural pipes make water pool in
  the right document.
- **Mechanism.** Build **one** sparse, weighted graph over *entities ∪ chunks ∪ documents* with
  four tunable edge families — `mentions` (1.0), `Entity↔Entity` relations (0.8, scaled by
  extraction confidence), `related_to` sequence (0.5), `part_of`/`contains` structural (0.3).
  Seed a personalization vector `e` from `idf × signal` entities (and, optionally, dense passage
  seeds). Run power iteration **`π = α·S·π + (1−α)·e`** (`α ≈ 0.5`) to convergence; read out the
  chunk nodes by `π`, then apply the same floor / per-doc cap / top-k as the baseline.
- **Graph used.** Both layers at once — the key departure from §3.1.
- **Published analogue.** **HippoRAG** `[ref: Gutiérrez et al. 2024]` (PPR over a KG with node
  specificity), built on Personalized/Topic-Sensitive PageRank
  `[ref: Haveliwala 2002; Page et al. 1999]`.

**Sub-variants of the PPR approach that were tried (the design space):**

- **Seed pipeline** (`PPR_SEED_MODE`):
  - *legacy* — top-N entities by fused score (the §3.1 gate);
  - *merge* (**current default**) — cast a wide net by a dense-score **threshold**, drop
    **boilerplate** entities by chunk-frequency, then **concept-merge** same-label entity
    fragments (e.g. "Sales Order [transaction]" + "Sales Order [process]") into one pooled seed
    weighted by its strongest member. This is the thesis's own seeding contribution on top of
    HippoRAG-style specificity.
- **Hybrid dense seeds** (`PPR_PASSAGE_SEED_WEIGHT`): mix the vector leg's top chunks into the seed
  vector — the **HippoRAG 2** idea `[ref: Gutiérrez et al. 2025]`. A toggle, so the leg can also
  run fully entity-independent for ablation.
- **Edge-family ablation:** structural edges on/off and per-family weights, to measure how much
  structure vs. semantics drives the walk.
- **Topic isolation** (`PPR_TOPIC_ISOLATION`): confine the walk to the dominant seed-topic(s) by
  summed seed mass, so a same-named entity in a foreign topic (e.g. a stray "SAP S/4HANA") cannot
  hijack the walk.

---

## 4. Side-by-side comparison

| Property | 3.1 Entity 1-hop | 3.2 Entity k-hop | 3.3 Hierarchical | 3.4 PPR (default) |
|---|---|---|---|---|
| Seeds | entities (`signal×idf`) | same | module ranking | entities (`idf×signal`) + optional dense seeds |
| Reach | 1 hop | k hops (decayed) | within a module | full multi-hop propagation |
| Graph layer used | semantic `mentions` | semantic `mentions` | structural hierarchy | **both** (semantic + structural + relations) |
| Generic-entity handling | IDF weight only | + hub cap | n/a (module-level) | IDF + degree-splitting + concept-merge + topic isolation |
| Doc scoring | MaxP (favours long docs) | MaxP | module-then-doc | pooled by connectivity |
| Published basis | local KG-RAG | multi-hop traversal | GraphRAG / RAPTOR | HippoRAG (+ HippoRAG 2 for dense seeds) |
| Output / fusion / rerank | shared `GraphResult` → union → cross-encoder (identical across all four) |

---

## 5. Empirical findings (summary — regenerate numbers before quoting)

On the internal 17-question evaluation, **both graph legs lost to the plain vector leg**, and PPR
did not overturn that. The value of the study is therefore as much *diagnostic* as it is a win —
useful framing for the thesis.

- **PPR's recall was frozen** across ranks (Hit@3 = Hit@5 = Hit@10), and its found-set was a
  *subset* of the entity-1hop leg's; it improved only Hit@1 slightly, by re-ordering.
- **Confirmed cause — read-out, not the walk.** A tiny seed-local pool combined with the deployed
  read-out cut (relative score floor + a low-confidence fallback-k + per-document cap) capped
  recall. Generic query entities act as extreme hubs and spread mass across sibling documents.
- **Refuted hypotheses (important negative results).**
  - "Removing structural edges will fix it" — structural edges barely rerouted real queries
    (they mostly concentrate the *seed-free* baseline), so removing them is **not** the fix.
  - "The walk collapses to a uniform distribution" — seeded results differ completely from a
    uniform walk; the seed *does* move mass, just to the wrong sibling documents.
- **Evaluation caveat.** The question set is almost entirely **single-hop lookups**, exactly where
  dense retrieval is strongest; the graph legs' expected advantage (relational / multi-hop
  questions) is under-represented. Adding relational questions is required before judging PPR.

*(These correct two predictions made in the earlier planning document `GRAPH_LEG_PLAN.md`, which
expected structural edges and IDF weighting alone to close the gap.)*

---

## 6. Positioning and contribution (how the thesis relates to the literature)

- **Adopted from the literature:** Personalized PageRank retrieval and node-specificity seeding
  from **HippoRAG** `[ref: Gutiérrez et al. 2024]`; the dense-passage seed integration from
  **HippoRAG 2** `[ref: Gutiérrez et al. 2025]`; structure-first localisation in the spirit of
  **GraphRAG** `[ref: Edge et al. 2024]` / **RAPTOR** `[ref: Sarthi et al. 2024]`.
- **Adapted / added by this thesis:** a **unified semantic+structural** PPR graph (most graph-RAG
  systems use one or the other); a **concept-merge seed pipeline** with boilerplate filtering on
  top of IDF specificity; **topic isolation** to prevent cross-area leakage; and a clean
  **engine-swap harness** (1-hop / k-hop / hierarchical / PPR behind one interface + fusion) that
  makes the approaches directly comparable — itself a methodological contribution.
- **Honest finding as contribution:** a **diagnosis** that, for this single-hop enterprise-handbook
  workload, the graph leg's limitation is *read-out and generic-hub seeding*, not the walk or the
  structural edges — a negative/clarifying result that guides where graph retrieval does and does
  not pay off.

---

## 7. Future direction — trust-aware retrieval (parked, design only)

The same PPR machinery can be **re-purposed for document trust**: swap the seed vector from
query-entities (relevance) to **reliable documents** (a metadata prior over type/author/freshness),
put document↔document edges weighted by shared **high-IDF** entities, and run the same power
iteration so trust flows from reliable sources to the documents that corroborate their *specific*
claims. Final score becomes `relevance × (1 + β·doc_prior) × trust`, computed at ingestion time
(no query-time cost). This is inspired by claim-trust propagation `[ref: ClaimTrust — verify]`
but deliberately avoids its expensive pairwise-contradiction core (kept as a further tier only if
extracted entities turn out to carry numeric attributes). **Designed, not implemented** — cite as
future work, not as a method evaluated here.

---

## References (⚠ verify all before use)

- `[ref: Lewis et al. 2020]` — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020.
- `[ref: Karpukhin et al. 2020]` — Dense Passage Retrieval for Open-Domain QA. EMNLP 2020.
- `[ref: Spärck Jones 1972]` — A Statistical Interpretation of Term Specificity and Its Application in Retrieval. (IDF.)
- `[ref: Page et al. 1999]` — The PageRank Citation Ranking: Bringing Order to the Web.
- `[ref: Haveliwala 2002]` — Topic-Sensitive PageRank. WWW 2002. (Personalised teleport vector.)
- `[ref: Edge et al. 2024]` — From Local to Global: A Graph RAG Approach to Query-Focused Summarization. (Microsoft GraphRAG.)
- `[ref: Gutiérrez et al. 2024]` — HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs. NeurIPS 2024.
- `[ref: Gutiérrez et al. 2025]` — HippoRAG 2 (From RAG to Memory / continual). *Verify title & venue.*
- `[ref: Guo et al. 2024]` — LightRAG: Simple and Fast Retrieval-Augmented Generation.
- `[ref: Sarthi et al. 2024]` — RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. ICLR 2024.
- `[ref: He et al. 2024]` — G-Retriever: Retrieval-Augmented Generation for Textual Graph QA. NeurIPS 2024.
- `[ref: KG²RAG]` — Knowledge-Graph-guided RAG with chunk organisation (e.g. MST). *Verify authors/year/title.*
- `[ref: ClaimTrust]` — trust/reliability propagation over a claim graph. *Verify — may not be a canonical citation.*
- `[ref: Chen et al. 2024, BGE-M3 / BAAI]` — `bge-reranker-v2-m3` cross-encoder. *Verify.*
</content>
</invoke>
