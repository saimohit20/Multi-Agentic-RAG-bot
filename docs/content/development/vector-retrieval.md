# Vector Retrieval Design

AVAILABLE's vector leg is a dense/sparse Qdrant search over processed corpus
text. "Vector-only" means excluding the graph leg, not disabling sparse lexical
retrieval. Dense/sparse fusion inside Qdrant and vector/graph fusion in the
application are two separate ranking stages.

## Representation and Storage

| Representation | Role | Current implementation |
| --- | --- | --- |
| Dense embedding | Semantic similarity across phrasing. | `BAAI/bge-m3`, normally 1024 dimensions, cosine collection distance. |
| Learned sparse vector | Lexical/token-level matching, including distinctive codes and terms. | BGE-M3 lexical weights, stored under Qdrant vector name `sparse`. |
| Stored text and metadata | Answer evidence, provenance, source navigation, and fusion identity. | LangChain document payloads in Qdrant. |
| Local document store | Original ingestion records keyed by metadata ID. | In-memory store persisted with the retriever's local cache. |

The collection has an unnamed dense vector and named sparse vector. Dense
vectors are configured with `on_disk=False`; the sparse index uses
`on_disk=True`. These are index/storage choices, not a guarantee of application
memory use or search latency.

BGE-M3 produces both representations through one model. Sparse weights are
learned token weights, not a separately maintained BM25 index. ColBERT-style
multi-vector outputs are disabled. The LangChain Qdrant wrapper is configured
for `RetrievalMode.HYBRID`, which combines dense and sparse rankings through
Qdrant's rank-fusion path. Returned hybrid scores are not raw dense cosines.

## What Gets Embedded

Ingestion records are split with `CharacterTextSplitter` using size `1000` and
overlap `200`. These are character-based targets, not token limits or a promise
that every fragment has that exact length; separator behavior matters.

Before embedding, the text is enriched with an identity header containing the
owning document's module code when recognizable, document name/code/title, and
known level. JSON-formatted summaries are parsed into readable string content
rather than embedding fence and object syntax. Non-JSON content remains prose
after applicable fence removal.

This changes retrieval representation without requiring every summary to repeat
its source title. It particularly helps short or generic passages and exact
document-code queries. Module extraction currently recognizes a particular
document-code pattern; a new corpus naming scheme may not get that part of the
header automatically.

The original records remain in the document store. The vector payload contains
the enriched child text, so the two stores deliberately need not have identical
`page_content` for the same metadata ID.

## Synthetic Document Cards

Indexing also adds one representative card per document-name group. A card uses
a file-level/meta summary when available, otherwise the first available record,
and includes identity, document name, and up to 1500 characters of cleaned
description. It has type `document_card`, page `-1`, and a fresh UUID.

Cards let overview documents compete with detailed sibling passages. They are
search candidates, not an independent source file or a user-authored passage.
They inherit source metadata but their IDs are not automatically graph Chunk
nodes or document-store entries. A retrieved card can remain vector evidence,
while its identifier contributes no PPR passage seed if absent from the graph.

Card grouping currently uses document name rather than a topic-qualified key.
Same-named documents across topics need care. IDs change across rebuilds, so do
not use card UUIDs as stable evaluation document labels or cross-build cache keys.

## The Identifier Contract

Distinguish three identities:

- **Qdrant point:** the stored vector record.
- **Metadata ID:** the original ingestion record identifier, inherited by its
  split children and used as the graph/fusion/document-store bridge.
- **Owning document:** the source document name used for recommendations and
  several grouping policies.

Several vector points can share one metadata ID after child splitting. Fusion
deduplicates by that metadata ID, not by Qdrant point. Therefore top-ten search
hits need not yield ten distinct evidence identifiers. A future chunking change
must decide whether to retain this record-level identity or introduce stable
child IDs across all stores; changing only one index breaks hydration/provenance.

## Query Path and Parent Context

The Q&A fusion path searches the vector store directly with the question and
`FUSION_VECTOR_K`, currently `10`. It uses those loaded hits as candidates,
passes their identifiers/scores to PPR when enabled, and resolves missing
graph-only identifiers through the document store.

Although the retriever abstraction derives from a parent-document retriever,
the main Q&A path does not automatically replace each vector hit with its full
parent. A separate retriever invocation path can group hits by training and
format a training summary with passages. That behavior should not be assumed
for direct Q&A search, and changing the formatter alone will not change Q&A's
candidate construction.

Document cards, child passages, and graph-hydrated original records can thus
coexist in the candidate pool. Inspect actual text and modality in traces when
evaluating context size and quality. See [Hybrid Retrieval Design](hybrid-retrieval.md)
for downstream ordering, reranking, trust, and multimodal expansion.

## Build, Load, and Compatibility

Loading expects both the Qdrant collection and the local `index_ret.pkl` cache.
If either is absent, or refresh is requested, the current builder recreates the
collection and rebuilds the matching document store. Existence checks do not
prove that two surviving artifacts are from the same corpus version.

!!! warning "Rebuild is not an incremental upsert policy"
    Refresh deletes the existing target collection before rebuilding it. Use a
    coordinated maintenance/publishing procedure; this is not an atomic
    blue-green index replacement. Only load trusted local pickle artifacts.

Treat embedding model, dense dimension, sparse vocabulary, text enrichment,
splitter policy, graph identifiers, and document-store version as one compatible
artifact set. Replacing the embedding model requires re-embedding documents as
well as questions and entity labels; matching dimensions alone is insufficient.

The embeddings and retriever are process-level singletons. Supplying another
model name after initialization does not automatically replace the existing
model. QA and RICH have separate retriever/collection targets while sharing the
embedding model. Restart or explicitly version these resources during a model
or corpus transition.

## Performance and Operational Boundaries

The model uses CUDA when available and fp16 on CUDA; otherwise it runs on CPU.
Encoding batches use size `32`. Query encoding obtains dense and sparse outputs
together and caches up to sixteen query entries per representation, so repeated
chunk/entity searches can reuse vectors. Document ingestion calls the dense and
sparse adapters separately; do not assume one shared model pass for each batch.

Separate cold model loading, query encoding, Qdrant search, entity search,
hydration, and downstream reranking in latency measurements. A warm repeated
question can benefit from an embedding cache even if no retrieval-result cache
exists. Cache behavior should be controlled in benchmark reports.

There is no topic or permission filter passed by the main Q&A vector-search
call. Topic selection in the graph leg is not access control and does not limit
the vector leg. Multi-tenant or restricted-corpus extensions must implement
authorization-aware filtering across every retrieval and hydration path.

## Extending and Evaluating

Evaluate exact-code queries, paraphrases, thin overview documents, same-named
documents across topics, duplicate child IDs, and missing paired artifacts.
Measure distinct-source recall before evaluating generation. When changing text
headers or cards, rebuild and hold the question set and graph policy fixed;
otherwise results can mix representation and fusion changes.

Ownership is split across `dc_available/RAG/hybrid_embeddings.py` for model
adapters, `retrieval.py` for representation/index population, and
`retrieval_helpers.py` for collection lifecycle. This guide describes their
contracts without prescribing a function-by-function implementation.