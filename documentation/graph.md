# Graph

*Part of AVAILABLE 2.0 · Pipeline STEP 4 (`pipeline/index.py --graph / --semantic`, `dc_available/graph/`)*

---

## Aim

Vector search matches on wording. It can miss a passage that describes the same idea in different
words, and it has no notion of how documents relate to each other. The **graph** is a second index,
built from the same content, that connects passages through the **concepts they share** and through
the **structure of the handbook** — so retrieval can reach the right passage even when its words don't
match the query.

The graph is the **second "eye"** next to the vector leg (see [hybrid](hybrid.md) for how the two are
combined).

---

## How the graph is constructed

The graph is built from the same `index.jsonl` the vector store uses, so both indexes share the same
chunk identities. It has **two layers**, built in order:

```
index.jsonl  (chunks + summaries, per training)
      │
      │  1) STRUCTURAL layer   (deterministic — no LLM)
      ▼
   one tree per Topic:
     Topic → Subtopic → Level → Training → Document → Chunk
     (linked by contains / part_of)
      │
      │  2) SEMANTIC layer    (LLM)
      ▼
   batch chunks → extract entities & relations → merge → integrate
     adds:   Chunk —mentions→ Entity        Entity ↔ Entity  (typed relations)
      ▼
   nodes.jsonl + edges.jsonl   (one folder per topic)
```

### 1. Structural layer (deterministic)
Built directly from the ingestion metadata, with **no LLM**. One **tree per top-level topic**
(e.g. Hydraulics, DigitalTransformation, GROWNFIELD). The hierarchy is
`Topic → Subtopic → Level → Training → Document → Chunk`, connected by `contains` / `part_of` edges.
This simply encodes the handbook's table of contents. Any explicit "next / previous / related" links
from ingestion become `related_to` edges between chunks.

### 2. Semantic layer (LLM)
This adds *meaning* on top of the structure, in four sub-steps:

1. **Batch** — the chunks of each document are grouped into small overlapping batches (no LLM yet).
2. **Extract** — an LLM reads each batch and returns the **entities** it mentions and the
   **relations** between them (e.g. *requires*, *precedes*, *prerequisite_for*), each with a
   confidence and the chunk it came from.
3. **Merge** — the same entity mentioned in many places is merged into **one** canonical entity;
   duplicate relations are combined. Each entity also gets an **IDF** value — a specificity weight,
   so rare/precise concepts count for more than generic ones.
4. **Integrate** — the merged entities are written back into the graph as **Entity** nodes, linked to
   the chunks that mention them (`mentions` edges) and to each other (typed `Entity ↔ Entity`
   relations).

Each topic ends up with its own `nodes.jsonl` and `edges.jsonl`.

---

## How the nodes are created (Document, Chunk, Entity)

- **Chunk nodes** come straight from ingestion. Each chunk keeps its **unique id** — the *same id
  used in the vector store* — so a chunk found by vector search and the same chunk in the graph are
  one and the same. This shared id is the bridge between the two indexes.
- **Document nodes** group the chunks that belong to the same source file; **Training / Level /
  Topic** nodes group the documents. All of this comes from the structural build.
- **Entity nodes** come from the semantic build: concepts the LLM found in the text, normalized and
  merged so that, for example, "Sales Order" written many ways becomes one entity carrying its IDF
  specificity weight.
- **Entity search index** — the entities are also **embedded into their own small vector index**, so
  that at query time the user's question can be matched to the **seed entities** it is about. (This
  is separate from the chunk vector store and is what the graph retrieval below starts from.)

---

## Graph retrieval approaches

Both approaches start the same way — match the query to a few **seed entities** — and end the same
way — hand a ranked list of chunks to [hybrid](hybrid.md) (union + reranker). They differ in **how
relevance spreads through the graph**. A mode switch selects which engine runs.

### Approach 1 — Entity hop

The simple version: hop from the query's concepts to the chunks that contain them.

1. **Seed** — match the query to seed entities; keep the confident ones; weight each by
   `signal × IDF` (specific concepts get more weight).
2. **Hop** — for each seed entity, step across `mentions` edges to every chunk that mentions it, and
   add the entity's weight to that chunk. *(Optionally repeat for k hops: chunk → its entities → more
   chunks, with a decay each hop and generic hub entities skipped.)*
3. **Normalize** — divide each chunk's score by the square root of how many entities it has, so long
   chunks don't win just by mentioning more things.
4. **Document prior** — give a small boost to chunks in a document whose best chunk already scored
   highly.
5. **Cut** — drop weak chunks below a relative floor, cap how many chunks come from one document, and
   keep the top-k.

### Approach 2 — PPR (Personalized PageRank)

The richer version (*water & pipes*): pour weight on the query's concepts and let it flow across the
**whole** graph — both layers at once — then read out where the most weight settles.

1. **Seed** — fetch candidate entities, keep the ones above a confidence threshold, **drop
   boilerplate** (entities that appear almost everywhere), and **merge** same-concept fragments into
   one seed weighted by `signal × IDF`. *(Optionally, the vector leg's top chunks can be added as
   extra seeds.)*
2. **Pour** — turn those seeds into a starting weight vector over the graph nodes.
3. **Flow** — run Personalized PageRank (`π = α·S·π + (1−α)·seed`) so weight spreads through the
   graph's edges. Each edge family carries a different strength:

   | Edge family | Meaning | Weight |
   |---|---|---|
   | `mentions` | Chunk ↔ Entity (the core signal) | 1.0 |
   | Entity ↔ Entity | typed relations (multi-hop reasoning) | 0.8 |
   | `related_to` | Chunk ↔ Chunk sequence links | 0.5 |
   | `part_of` / `contains` | the hierarchy (pools weight inside a document/module) | 0.3 |

   Because a node with many connections splits its weight thinly, generic hubs can't flood the graph,
   and the structural edges keep weight pooling in the right document.
4. **Read out** — rank the chunk nodes by how much weight they collected.
5. **Cut** — the same floor, per-document cap, and top-k as Approach 1.

Both approaches return the **same** ranked `(chunk, score)` list, so the rest of the system doesn't
change when you switch between them.

---

## How this fits into AVAILABLE 2.0

The graph is built in **STEP 4** alongside the vector store:

```
ingest → sync → pre-process → index → serve
                                 │
                                 ├─ vector store  (Qdrant)
                                 └─ graph         (structural + semantic)
```

At query time the graph leg runs next to the vector leg and their results are fused and reranked
(see [hybrid](hybrid.md)). The whole graph — a second retrieval leg and the two approaches above — is
a **new capability in AVAILABLE 2.0**; the 1.0 baseline had vector search only.

---

## How to run

The graph is built by the index step, in two phases:

```bash
python -m pipeline.index --graph      # structural graph + prepare the semantic batches
python -m pipeline.index --semantic   # LLM extraction → merge → integrate (adds entities + relations)
```

Useful `--semantic` options: `--semantic-model`, `--semantic-topics`,
`--semantic-min-confidence`, `--semantic-min-occurrences`. Add `--refresh` to rebuild from scratch.
```
