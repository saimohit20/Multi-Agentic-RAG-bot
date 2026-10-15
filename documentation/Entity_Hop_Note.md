# Update Note — Graph Retrieval, Approach 1: Entity Hop

## Aim
Improve retrieval on the process-handbook corpus by adding a **graph leg** next to the existing
vector (embedding) search, so that passages can be found through the **concepts they share**, not
only through surface-text similarity.

## The problem
Pure vector search matches on wording. It therefore (a) can miss a passage that describes the same
specific concept in different words, and (b) is easily pulled toward chunks full of **generic**
terms ("order", "billing") that appear everywhere.

## Solution — Approach 1: Entity Hop
Connect chunks through the **entities** they mention, and let a query "hop" from its concepts to the
chunks that contain them — while giving **specific** concepts far more weight than generic ones.

**How it solves the problem (in short):**
- Retrieval follows **meaning links** (shared entities), so it reaches the right chunk even when the
  words differ.
- Each concept is weighted by **IDF (specificity)**, so rare/precise terms drive the result and
  common terms are suppressed.
- It **reuses the graph already built** during ingestion (chunk→entity links), so it is a light
  add-on, not a new pipeline.
- It plugs into the existing system unchanged: its output is fused with the vector leg and passed to
  the same reranker.

## How it fits my project — the flow

```
entity extraction  →  seed weighting  →  graph lookup (hop)  →  score aggregation  →  rank
```

1. **Entity extraction** *(ingestion)* — every chunk is linked to the entities it mentions
   (`chunk —mentions→ entity`); each entity gets an **IDF** value (high = specific, low = generic).
2. **Seed weighting** *(query time)* — the query is matched to a set of **seed entities**; each seed
   gets a weight = its match confidence × its IDF.
3. **Graph lookup (hop)** — for each seed entity, take **one hop** to every chunk that mentions it,
   passing the seed's weight to that chunk.
4. **Score aggregation** — sum the weights each chunk received, normalise for chunk length, and add
   a small **document prior** (a chunk in an already-strong document is boosted).
5. **Rank** — sort chunks by score, drop weak ones, cap per document, keep the top-k → hand to
   fusion + reranker.

## The maths behind it

**Notation:** `Q` = seed entities from the query · `s_e` = match confidence of entity `e` ·
`idf_e` = its specificity · `M(e)` = chunks that mention `e` · `E_c` = entities in chunk `c`.

**1. Seed weight** — specific and confidently-matched concepts get the most "water":

```
w_e = s_e · idf_e            with   idf_e = log( N / df_e )
```

(`N` = total chunks, `df_e` = chunks mentioning `e`; so a rare entity has a large `idf_e`.)

**2. One-hop chunk score** — each chunk collects the weight of every seed it mentions:

```
raw(c) = Σ_{e ∈ Q}  w_e · 1[ c ∈ M(e) ]
```

**3. Length normalisation** — so long chunks don't win just by mentioning more entities:

```
score(c) = raw(c) / sqrt( |E_c| )
```

**4. Document prior (MaxP)** — a document is as strong as its best chunk; boost its chunks a little:

```
maxp(d) = max_{c ∈ d} score(c)
prior(d) = maxp(d) / max_d maxp(d)
score'(c) = score(c) · ( 1 + β · prior(doc(c)) )
```

**5. Ranking** — keep chunks above a relative floor, cap per document, take the top-k:

```
keep c  if  score'(c) ≥ ρ · max_c score'(c)      then per-doc cap → top-k
```

**Optional multi-hop (k-hop).** The single hop can be repeated: relevance flows one more step
(`chunk → its entities → more chunks`), with each extra hop scaled by a decay factor `γ < 1` and
generic hub entities skipped, so the walk reaches related passages without diffusing into noise.
</content>
