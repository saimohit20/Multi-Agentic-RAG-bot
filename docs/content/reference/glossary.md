# Glossary

| Term | Meaning in AVAILABLE |
| --- | --- |
| RAG | Retrieval-augmented generation: provide retrieved evidence to a language model before answering. |
| Corpus | The source material included in a particular knowledge-base build. |
| Training | A document collection grouped beneath a topic and level. |
| ETL | Extract, transform, load: prepare source content for downstream use. |
| Raw mirror | Local copy of the organized SharePoint source. |
| Artifact | Generated output such as chunks, graph files, or a parent-document store. |
| Chunk | A searchable text passage with source metadata. |
| Parent-document store | Larger context passages reconstructed from retrieved child chunks. |
| Dense embedding | Learned vector representing semantic similarity. |
| Sparse embedding | Weighted sparse representation useful for lexical matching. |
| Entity | A normalized concept extracted from training passages. |
| IDF | Inverse document frequency; a specificity signal that reduces the influence of common terms. |
| PPR | Personalized PageRank; query-seeded graph ranking. |
| Fusion | Combining candidate results from retrieval paths. |
| RRF | Reciprocal Rank Fusion; combine rankings using rank positions. |
| Reranker | A model that reorders candidate passages using the question and passage together. |
| Trust prior | Metadata and content-based expectation of source reliability. |
| TrustRank | Propagation of trust signals across document relationships. |
| Trace / span / thread | Request record / processing step / related conversation activity in Opik. |
| Hit@k | Fraction of queries with a relevant result among the first k results, under the evaluation's matching rules. |
| MRR | Mean reciprocal rank of the first relevant result. |
| Workload Identity | Federated Azure authentication for Kubernetes workloads without an application API key. |
| PVC | Kubernetes persistent volume claim. |
| RICH | Separate interview-oriented feature and corpus workflow. |