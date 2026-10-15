# Reducing Graph Retrieval Latency in AVAILABLE 2.0

## 1. Purpose

This document describes the investigation and optimization of query-time graph retrieval latency in AVAILABLE 2.0. The analysis shows that the graph algorithm itself was not the main bottleneck. Most of the latency came from repeated neural embedding computations performed before Personalized PageRank (PPR) began.

The optimization therefore focused on eliminating redundant BGE-M3 inference while preserving the existing retrieval behavior and ranking signals.

---

## 2. Graph Retrieval Pipeline

The graph retrieval leg uses the following sequence:

```text
User question
    |
    v
Entity matching in Qdrant
    |- hybrid entity search: dense + sparse similarity
    `- dense-only entity search: raw cosine confidence
    |
    v
Entity filtering and seed selection
    |
    v
Personalized PageRank over the knowledge graph
    |
    v
Graph chunk ranking and result assembly
```

The hybrid entity score determines the ordering of matching entities. The additional dense score is used as a confidence signal during seed selection. The selected entities, together with optional passage seeds from vector retrieval, form the personalization vector for PPR.

Relevant implementation files are:

- `dc-available/dc_available/RAG/hybrid_embeddings.py`
- `dc-available/dc_available/graph/construct/entity_index.py`
- `dc-available/dc_available/graph/retrieval/ppr_retriever.py`
- `dc-available/dc_available/RAG/qa_functions.py`

---

## 3. Initial Latency Problem

### 3.1 Repeated embedding of the same question

BGE-M3 supports both dense semantic embeddings and sparse lexical embeddings. Originally, the same user question could be encoded multiple times during one retrieval request:

1. Dense encoding for hybrid chunk retrieval.
2. Sparse encoding for hybrid chunk retrieval.
3. Dense encoding for hybrid entity retrieval.
4. Sparse encoding for hybrid entity retrieval.
5. Dense encoding for the separate dense-only entity search.

Although these calls used the same question and the same model, the generated vectors were not shared across retrieval operations. This was expensive because BGE-M3 was running on the CPU.

The computational pattern was approximately:

```text
same question -> BGE-M3 dense
same question -> BGE-M3 sparse
same question -> BGE-M3 dense
same question -> BGE-M3 sparse
same question -> BGE-M3 dense
```

The problem was therefore redundant model inference rather than graph traversal complexity.

### 3.2 Repeated Qdrant wrapper validation

A second source of latency was found after adding detailed timing and encode logs. The LangChain `QdrantVectorStore` validates the vector dimension when a wrapper is created. It does this by embedding an internal string named `"dummy_text"`.

The entity retrieval code created two wrappers:

- one hybrid wrapper;
- one dense-only wrapper.

When these wrappers were reconstructed for a query with collection validation enabled, LangChain performed two additional BGE-M3 calls:

```text
"dummy_text" -> BGE-M3 dense validation
"dummy_text" -> BGE-M3 dense validation
```

These calls did not represent useful retrieval work. They only verified that the model output dimension matched the Qdrant collection configuration. Since the collection has a fixed schema at query time, repeating this validation for every question was unnecessary.

---

## 4. Profiling Method

Fine-grained timers were added around the main graph retrieval phases using `time.perf_counter()`.

The instrumentation separates:

- hybrid entity search;
- dense-only entity search;
- seed selection;
- PPR execution;
- result assembly;
- total graph retrieval;
- vector retrieval and reranking outside the graph leg.

The embedding adapter also logs every real BGE-M3 forward pass. This made it possible to distinguish question encoding from LangChain's internal `dummy_text` validation.

Example diagnostic output:

```text
[ENCODE] real BGE-M3 pass n=1 dense=True sparse=False first48='dummy_text'
[ENCODE] real BGE-M3 pass n=1 dense=True sparse=False first48='dummy_text'
[LAT] entity_search hybrid(dense+sparse)=6.57s dense_only=4.44s
[LAT] graph_phases seeds=11.58s ppr=0.81s assemble=0.00s
[LAT] retrieval breakdown vector=2.38s graph=12.39s rerank=22.00s
```

### 4.1 Interpretation of the measurement

| Graph phase | Observed latency | Share of graph latency |
|---|---:|---:|
| Hybrid entity search | 6.57 s | 53.0% |
| Dense-only entity search | 4.44 s | 35.8% |
| PPR and remaining graph work | approximately 1.38 s | 11.1% |
| Total graph retrieval | 12.39 s | 100% |

The separately reported PPR time was only 0.81 seconds, or approximately 6.5% of total graph latency. Seed selection took 11.58 seconds because that timer included entity matching and its embedding work.

This result changed the optimization strategy. Reducing the PPR iteration limit would target a small part of the measured latency and could reduce retrieval quality. Removing redundant embedding operations offered much greater potential without changing graph ranking semantics.

> The values above are from one diagnostic run and should be treated as profiling evidence, not as a statistically complete benchmark. Median and 95th-percentile results over repeated warm queries are required for final quantitative claims.

---

## 5. Implemented Optimization 1: Combined Query Encoding

### 5.1 One model pass for dense and sparse vectors

The `HybridBGEM3Embeddings` adapter now computes dense and sparse query vectors together in a single `get_dense_and_sparse()` call, which issues one combined BGE-M3 forward pass:

```python
def get_dense_and_sparse(self, texts):
    payload = self._encode(
        texts,
        return_dense=True,
        return_sparse=True,
    )
    ...
```

BGE-M3 produces both representations in one forward pass. The vectors are then separated into the formats required by LangChain and Qdrant. On a query cache miss, `_cache_query_embeddings()` invokes this method once and stores both resulting vectors (see §5.2).

The optimized pattern is:

```text
question -> one BGE-M3 pass -> dense vector + sparse vector
```

### 5.2 Query-level LRU cache

The dense and sparse vectors are stored in two coordinated least-recently-used caches. Both caches contain at most 16 questions.

When either `embed_query()` or `get_query_sparse_embedding()` encounters a cache miss, `_cache_query_embeddings()` computes both representations and stores them together. Therefore, the result does not depend on whether LangChain requests the dense or sparse representation first.

Subsequent operations reuse the cached vectors for:

- chunk hybrid search;
- entity hybrid search;
- entity dense-only search;
- repeated identical questions within the cache window.

The optimization changes computation reuse but does not change the embedding model, vectors, similarity functions, entity scores, or PPR configuration. Retrieval behavior should therefore remain equivalent apart from normal floating-point determinism.

---

## 6. Implemented Optimization 2: Removing Repeated Validation Inference

The entity index code now constructs the hybrid and dense Qdrant wrappers with:

```python
validate_collection_config=False
```

This prevents LangChain from embedding `"dummy_text"` merely to validate the already-known collection dimension during query-time wrapper construction.

The code also maintains a `_VS_CACHE` of vector-store wrappers, keyed by client identity, embedding-model identity, retrieval mode, and collection name. This allows wrappers to be reused when the same Qdrant client and embedding instance are reused.

Including the collection name in the key is important because AVAILABLE 2.0 can use separate entity collections for different data sources. It prevents one cached wrapper from accidentally querying another collection.

The two changes serve related but distinct purposes:

- `validate_collection_config=False` removes the expensive validation embedding even if a wrapper is reconstructed.
- `_VS_CACHE` avoids reconstructing wrappers when the same client and embeddings are reused.

Collection compatibility remains an ingestion and deployment responsibility. The entity collection is created with the BGE-M3 dense dimension and the expected sparse-vector configuration before serving queries.

---

## 7. Before-and-After Computational Flow

### Before optimization

```text
Question
  |- repeated dense query encodes
  |- repeated sparse query encodes
  |- hybrid entity wrapper validation using "dummy_text"
  |- dense entity wrapper validation using "dummy_text"
  `- PPR
```

### Current implementation

```text
Question
  |- one combined dense+sparse encode
  |- cache both query vectors
  |- reuse vectors across retrieval calls
  |- skip repeated Qdrant collection validation inference
  |- reuse Qdrant wrappers when client/model identities match
  `- PPR
```

The key design principle is to perform expensive, deterministic neural computation once and reuse its result throughout the request.

---

## 8. Why PPR Was Not Simplified

PPR was initially a possible suspect because it can perform up to 100 power iterations. Profiling showed that it required only 0.81 seconds in the diagnostic run, while entity matching required approximately 11 seconds.

Changing parameters such as `PPR_MAX_ITER` or `PPR_TOL` could affect graph ranking quality and would address only a small fraction of the observed latency. For this reason, the graph algorithm was retained and optimization was applied to redundant preprocessing operations instead.

This illustrates an important engineering principle: optimize the measured bottleneck rather than the component that appears theoretically complex.

---

## 9. Expected Effect and Current Evidence

The implemented changes remove the following unnecessary work:

- separate dense and sparse forward passes for the same new question;
- repeated dense query encoding across retrieval operations;
- repeated sparse query encoding across retrieval operations;
- two `dummy_text` validation embeddings during entity retrieval.

A focused mocked validation confirmed that dense-first and sparse-first query access each cause exactly one combined embedding call and that the second representation is served from cache.

The earlier diagnostic run established that the removable validation and entity-search work dominated graph latency. However, the final implementation still requires a controlled post-optimization benchmark before reporting a measured end-to-end speedup in the thesis.

A scientifically defensible comparison should use the same questions, hardware, model versions, Qdrant collection, and process state before and after optimization.

---

## 10. Recommended Evaluation Protocol

1. Restart the application and record the first request separately as a cold-start measurement.
2. Run one warm-up question to load BGE-M3, graph indexes, sparse matrices, and the reranker.
3. Execute at least 10 representative questions.
4. Repeat the same question set for the unoptimized and optimized versions.
5. Report the median, mean, standard deviation, and 95th percentile.
6. Record these components independently:
   - hybrid entity search;
   - dense-only entity search;
   - seed selection excluding entity search, if instrumented separately;
   - PPR;
   - total graph retrieval.
7. Confirm from `[ENCODE]` logs that new questions require one combined query encode and no repeated `dummy_text` encodes.
8. Run the existing retrieval evaluation to verify that Hit@k, MRR, and source-hit metrics do not regress.

Cold and warm measurements must not be mixed because initial model loading and graph-index construction represent startup cost rather than steady-state query latency.

---

## 11. Threats to Validity

Several factors can influence the observed latency:

- BGE-M3 runs much faster on a GPU than on a CPU.
- Kubernetes CPU requests, limits, and throttling can affect inference time.
- The first request may load model weights and graph artifacts from disk.
- Operating-system file caches can make later graph loads faster.
- Repeated questions benefit from the query LRU cache, while unique questions still require one model pass.
- Qdrant network and storage conditions may vary between local and deployed environments.
- Debug logging adds small overhead, although it is minor compared with CPU model inference.

Therefore, benchmark conditions and hardware should be reported together with latency results.

---

## 12. Thesis-Ready Summary

The initial graph retrieval implementation exhibited high latency, but fine-grained profiling demonstrated that Personalized PageRank was not the primary cause. In one diagnostic run, the complete graph leg required 12.39 seconds, whereas PPR itself required only 0.81 seconds. Most time was spent in entity matching, where the BGE-M3 model repeatedly encoded the same query for dense and sparse retrieval. In addition, the LangChain Qdrant wrapper performed two query-time dimension checks by embedding an internal `dummy_text` string whenever hybrid and dense wrappers were constructed.

To reduce this overhead, the embedding adapter was modified to compute dense and sparse query representations in a single BGE-M3 forward pass. Both vectors are stored in coordinated LRU caches and reused across chunk retrieval, entity retrieval, and dense confidence scoring. The Qdrant entity wrappers were also configured to skip repeated collection validation and were cached for reuse when the same client, model, retrieval mode, and collection are used. These changes preserve the retrieval model and graph-ranking procedure while eliminating deterministic duplicate computation.

The investigation demonstrates that the dominant graph-retrieval cost originated in neural preprocessing and framework initialization rather than graph propagation. Consequently, measurement-guided removal of redundant model inference was preferred over reducing PPR iterations, which would have offered a smaller latency benefit and could have degraded retrieval quality. A controlled repeated-query benchmark is required to quantify the final reduction in median and tail latency.
