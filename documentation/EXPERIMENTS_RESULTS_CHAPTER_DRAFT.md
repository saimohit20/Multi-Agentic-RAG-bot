# Experiments and Results

The evaluation examined whether AVAILABLE 2.0 improves access to SAP S/4HANA training content and whether its trust scoring responds to changes in document quality. Retrieval was assessed across controlled and expert prepared question sets. Generation quality was assessed from the answers and their retrieved context. The trust score was examined through controlled document perturbations and through a retrieval and generation ablation with poisoned document twins. Together, these experiments provide evidence for the retrieval, answer generation, and source assessment components introduced in the preceding methodology.

## Retrieval Results

Retrieval performance was examined under three data conditions. The Grownfield questions represent expert assessment of the training corpus, with 33 questions divided between 17 single hop and 16 multi hop cases. A second expert set was formed from one German Hydraulics handbook and contains 59 questions. The synthetic evaluation extends the coverage to 1,215 questions generated from the available source documents. Five questions were associated with each source document, covering the five question types defined for the evaluation.

The retrieval results are considered at several ranks because a relevant document can support an answer without being placed first. Hit@1, Hit@3, and Hit@10 measure whether the relevant document occurs within the respective result depth, while mean reciprocal rank reflects its position across the complete question set. The Grownfield questions were also evaluated at code, submodule, and module level. Recall@10 was used alongside these levels to distinguish exact retrieval from recovery within the surrounding process structure. Since nDCG was not computed, no result is reported for that measure.

### Graph Engine Selection

The choice of graph retrieval engine was examined on the same Grownfield question set containing 23 questions. One hop entity retrieval was compared with two hop retrieval, three hop retrieval, and personalized PageRank. Their document level results are given in Table 1.

| Graph engine | Hit@1 | Hit@3 | Hit@10 | MRR |
| --- | ---: | ---: | ---: | ---: |
| One hop merge | 0.4348 | 0.6087 | 0.6522 | 0.5254 |
| Two hop merge | 0.2609 | 0.5217 | 0.5217 | 0.3841 |
| Three hop merge | 0.2609 | 0.5217 | 0.5217 | 0.3841 |
| PPR merge | 0.3043 | 0.6522 | 0.6957 | 0.4601 |

The one hop configuration placed the relevant document first in 0.4348 of the questions and achieved an MRR of 0.5254. Extending the search to two or three hops did not improve any measured value. Both configurations produced identical results, with Hit@1 of 0.2609, Hit@3 of 0.5217, Hit@10 of 0.5217, and MRR of 0.3841.

Personalized PageRank produced the highest Hit@3 and Hit@10 values, at 0.6522 and 0.6957. Its Hit@1 was 0.3043 and its MRR was 0.4601. Thus, PageRank increased the proportion of questions for which the relevant document was reached at greater rank depths, whereas one hop retrieval placed that document first more often. No significance test was stored with this comparison, so the differences are interpreted descriptively.

For this question set, extending the entity search beyond one hop therefore introduced no measured retrieval benefit. The additional hops reduced first rank accuracy and did not increase document level reach at rank ten. PageRank provided greater reach at ranks three and ten, but one hop retrieval remained preferable for first rank accuracy and reciprocal rank.

#### Suggested figures and tables

1. Table 1. Document level retrieval metrics for the four graph engines.
2. Figure 1. Comparison of Hit@1, Hit@3, and Hit@10 across graph engines.
3. Figure 2. Distribution of the correct document rank for each graph engine.
