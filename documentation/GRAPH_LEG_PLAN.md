# Plan: Upgrade the Graph Retrieval Leg (1-hop → Personalized PageRank)

**Status:** proposed · **Scope:** improve the existing system (swap one component, not a rewrite)
**Owner:** thesis work · **Chosen approach:** HippoRAG-style Personalized PageRank (PPR) graph leg

---

## TL;DR

The current graph leg **loses to plain vector search on every metric** because it seeds on
generic entities and only walks **one hop**, and it ignores the structural graph we already
built. We replace its engine with **Personalized PageRank** over both the **semantic**
(`mentions`) and **structural** (`part_of` / `related_to`) graph, seeded by **IDF-weighted
entities** (+ optional dense chunk seeds). Everything else — the hybrid vector leg, the
header/card embedding fix, `union` fusion, and the `bge-reranker-v2-m3` cross-encoder — stays.

---

## 1. Previous approach (what we have today)

**Code:** `dc_available/graph/retrieval/entity_hop_retriever.py` → `retrieve()` and `_GraphIndex`.

**How it works (entity-seeded, 1-hop):**
1. Match the query to **seed entities** via the entity index (`search_entities_scored`), keep
   those above `GRAPH_MIN_FLOOR`, cap at `GRAPH_MAX_SEED_ENTITIES`.
2. Weight each seed entity = `signal_score × idf`.
3. **One hop:** for each seed entity, add its weight to every chunk that `mentions` it
   (`raw[chunk] += weight`).
4. Length-normalize by `sqrt(entity_count)`, apply a MaxP document prior, floor + per-document
   cap + top-k.
5. Return `GraphResult(chunks=[(chunk_id, score)])` → `union` fusion → cross-encoder rerank.

**What it uses:** only the **semantic layer** (`mentions`), plus `related_to` in the optional
hierarchical path. The **structural graph** (`part_of` / `contains` hierarchy:
chunk→document→module) is built but **not used for scoring**.

**Measured result (17-question eval, graph-only leg):**

| metric | vector | **graph (current)** |
|---|---|---|
| Hit@1 | 0.47 | **0.29** |
| Hit@5 | 0.82 | **0.59** |
| Hit@10 | 0.82 | **0.59** |
| MRR | 0.60 | **0.43** |
| nDCG@10 | 0.57 | **0.46** |

Per-question vs vector: **6 regressions** (graph loses a doc vector found:
gf_001,002,006,008,009,013), **2 wins** (graph finds a doc vector missed: gf_007 `O2C-04.020`,
gf_017 `R2S-04.000`), rest tie/miss. **It never beats vector on any aggregate metric.**

**Why it fails:**
- **Generic-entity mis-routing** — seeds like "sales order", "billing", "quotation" are
  low-IDF and point to many sibling documents, so the wrong sibling wins.
- **Only one hop** — cannot reach a correct chunk that sits a hop away via a shared entity.
- **MaxP favours fat documents** — thin overview/`.000` docs lose to detailed siblings.
- **Structural graph unused** — nothing keeps activation inside the right document/module.

---

## 2. New approach (Personalized PageRank graph leg)

**Intuition (water & pipes):** pour water at the query's seed nodes; let it flow through the
graph's pipes; retrieve the chunks where the most water pools.
- **How much water each seed gets = IDF** (rare/specific words get a full bucket; generic words
  get a few drops).
- **A node with many pipes splits its water thinly** (PPR's degree normalization), so hub words
  can't flood the network.
- **Structural pipes** (chunk→document→module) make the water **pool in the right document**,
  not leak to unrelated ones.

**How it works:**
1. **Build one graph** over three node types — **entities ∪ chunks ∪ documents** — with edges:
   `chunk—mentions—entity` (semantic), `chunk—related_to—chunk` (sequence), and
   `chunk—part_of—document—part_of—module` (**structural, newly switched on**).
2. **Seed** the personalization vector from:
   - **entities** matched to the query, weighted by **node specificity (IDF) × signal** (this
     suppresses generic entities), and
   - *(toggle)* **dense chunk seeds** = the vector leg's top chunks at a small weight
     (HippoRAG-2 style; keeps it switchable so the leg can also run fully independent).
3. **Run Personalized PageRank** `π = α·Sᵀπ + (1−α)·e`, `α≈0.5`, capped iterations.
4. **Rank chunk nodes** by `π`; apply the existing floor / per-doc cap / top-k.
5. Return the **same** `GraphResult(chunks=…)` → **same** `union` fusion + cross-encoder.

---

## 3. Previous vs New — at a glance

| | Previous (1-hop) | New (PPR) |
|---|---|---|
| Seeds | entities only | IDF-weighted entities **+ optional dense chunk seeds** |
| Generic-entity handling | none (mis-routes) | **IDF seed weighting + PPR degree-splitting** |
| Reach | 1 hop | **multi-hop propagation** |
| Structural graph | unused | **used as pipes (pool in right doc/module)** |
| Doc scoring | MaxP (favours fat docs) | water pools by connectivity, not chunk count |
| Output / fusion / reranker | `GraphResult` → union → rerank | **unchanged** |

---

## 4. Why the new approach is better (failure → fix)

| Current failure | Fix in new approach |
|---|---|
| Generic entities mis-route (the 6 regressions) | **IDF seed weighting** — specific words dominate; generic ones get drops |
| Fat siblings drown the right doc | **PPR degree-splitting** — hub words split their water thinly |
| Right module, wrong sibling (low Hit@1) | **Structural `part_of` edges** pool water in the right document/module |
| Can't reach a doc a hop away (why vector already beats it) | **Multi-hop propagation** reaches connected chunks (keeps the 2 wins, helps relational Qs) |
| Blind to the strong (card-boosted) text signal | **Dense passage seeds** ground the graph in the vector leg |
| Risk of a big rewrite | **Drop-in**: same `GraphResult`, same fusion + reranker |

**Net expectation:** the 6 regressions mostly disappear (IDF + structure), the 2 wins are kept
(multi-hop), and **hybrid ≥ vector** on Hit@1 / MRR — which the current graph leg never achieves.

---

## 5. Staged implementation plan

**Stage 0 — Honest baseline (prerequisite)**
- Re-embed with the header/card fix: `poetry run python -m pipeline.index --vectordb --refresh`.
- Tag the 15 missing-video golden codes as "not in corpus" so recall isn't unfairly capped.
- Record vector / entity-1hop-graph / hybrid numbers — **the bar PPR must beat**.
- Keep the current `retrieve()` as the legacy path for ablation.

**Stage 1 — Build the PPR graph object**
- New `graph/retrieval/ppr_graph.py` (or extend `_GraphIndex`): load `nodes.jsonl` / `edges.jsonl` and
  assemble one `scipy.sparse` adjacency over entities ∪ chunks ∪ documents, including the
  structural `part_of` edges. Cache per topic like `_index_cache`.

**Stage 2 — Seeding**
- Phrase seeds from `search_entities_scored`, weight = `idf × signal` (reuse `GRAPH_MIN_FLOOR`,
  `GRAPH_MAX_SEED_ENTITIES`).
- Toggleable dense passage seeds from the vector leg's top chunks (`PPR_PASSAGE_SEED_WEIGHT`).

**Stage 3 — Run PPR + read out**
- Power-iteration PPR (`scipy`/`igraph`), rank chunks, reuse `GRAPH_SCORE_FLOOR_RATIO`,
  `GRAPH_MAX_CHUNKS_PER_DOC`, `GRAPH_CANDIDATE_K`. Return `GraphResult`. Abstain rule unchanged.

**Stage 4 — Wire into fusion (minimal)**
- In `_search_question_fusion`, pass `vector_ids`/scores into `graph_retrieve(...)` for passage
  seeds. Add `GRAPH_RETRIEVAL_MODE = "entity_1hop" | "ppr"` to flip and compare.

**Stage 5 — New config knobs (in `data_config.py`)**
- `PPR_ALPHA`, `PPR_MAX_ITER`, `PPR_PASSAGE_SEED_WEIGHT`, `PPR_USE_STRUCTURAL_EDGES`,
  `PPR_STRUCTURAL_EDGE_WEIGHT`, `PPR_RELATED_EDGE_WEIGHT`. Conservative defaults (low α = precision).

**Stage 6 — Evaluate & ablate**
- Run `run_predictions.py` + `score_retrieval.py` for: vector-only; graph entity-1hop vs PPR;
  hybrid (vector + PPR). Add ~5 relational questions ("what's needed *before* X?", multi-doc) to
  expose where PPR earns its keep.

**Order:** 0 → 1 → 2 → 3 → 4 → 5 → 6. Stages 1–3 are the real build; 4–5 are small; 6 is measurement.

---

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| PPR over-diffuses (noise) | lower `PPR_ALPHA`; IDF seed weighting; tune structural weight |
| Structural edges pool *everything* in a doc | keep `PPR_STRUCTURAL_EDGE_WEIGHT` small; ablate on/off |
| Passage seeds make the graph "copy" vector | keep them a toggle; report entity-only vs +passage separately |
| No gain on single-hop questions | expected — add relational questions; report as a finding |

---

## 7. Out of scope (parked on purpose)

- KG²RAG MST-organization, LightRAG dual-keyword LLM routing.
- Re-ingesting the 15 missing videos (separate data task).
- A/B harness beyond the existing `score_retrieval.py` comparison.
