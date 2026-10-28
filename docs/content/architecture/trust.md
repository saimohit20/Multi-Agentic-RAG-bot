# Document Trust

Trust scoring provides a query-independent document ranking signal. It combines
metadata-based expectations with graph-based support; it does not prove that a
document's statements are true.

![Document trust from metadata prior and graph support to query-time ranking](../assets/figures/trust/trust%20score.png)

*Trust is precomputed and looked up during retrieval.
[Open full size](../assets/figures/trust/trust%20score.png).*

## Scoring Stages

| Stage | Inputs and purpose |
| --- | --- |
| Metadata collection and alignment | Match SharePoint source metadata to ingested documents. |
| Prior | Combine content quality, authority, format, and freshness signals. |
| Graph support | Propagate the prior over document connections derived from shared specific entities. |
| Final trust | Blend the prior and support values into the published document table. |

The documented prior weights are content quality `0.45`, authority `0.25`,
format `0.15`, and freshness `0.15`. The final blend gives the prior `0.7` and
graph support `0.3`. These are policy and model-derived signals, not measured
probabilities of factual accuracy. Shared entities indicate topical overlap,
not necessarily agreement between documents.

## Build Scope and Commands

The current production builder is coupled to the GROWNFIELD metadata workflow.
It requires the configured source SharePoint access, aligned ingestion outputs,
Azure model access for content scoring, and graph artifacts. It is not a generic
trust builder selected solely by `--source`.

From `dc-available`, for the configured GROWNFIELD corpus:

```powershell
poetry run python -m dc_available.trust_score.collect_metadata
poetry run python -m dc_available.trust_score.align_metadata
poetry run python -m dc_available.trust_score.prior_score
poetry run python -m dc_available.trust_score.final_trust
```

The final command runs graph propagation and publishes the final table. The
automatic pipeline invokes the orchestration after indexing. Running
`python -m pipeline.trust_score` alone is not a supported CLI workflow: that
module exposes orchestration but does not invoke it as a module entry point.

The default table is
`dc_available/data/Artifacts/resources/trust/grownfield/metadata_trust.jsonl`.
`TRUST_FILE` can select an existing compatible table at serving time; it does
not redirect the whole build workflow. Restart the app after replacing a table
because lookups are cached in-process.

## Interpreting Missing Values

Unmatched documents receive the table mean. If the table is missing or empty,
the common fallback is `1.0`. A displayed `1.00` can therefore be a fallback,
not a computed perfect-trust assessment. Inspect table coverage before using
displayed values for comparisons.

`TRUST_ENABLED` and `TRUST_BETA` govern ranking influence. Test their effect on
retrieval and answers rather than judging quality from trust values alone.