"""Stage 2 (RICH) — TrustRank over graph/rich.

Thin wrapper: reuses ``trustrank.run`` (same PageRank math + tunables) but points it at RICH's
own artifacts — the ``metadata_prior.jsonl`` we built and the RICH graph topic dir
(``graph/rich/rich``) — writing ``trust/rich/metadata_trust.jsonl``. QA's grownfield run is
untouched (the paths are just parameters now).

RUN (from the dc-available directory):
    python -m dc_available.trust_score.trustrank_rich
"""
from __future__ import annotations

from dc_available.config import data_config
from dc_available.trust_score import trustrank


def run() -> None:
    arts_root = data_config.source_artifact_paths("rich")["ingestion"].parent
    trust_dir = arts_root / "trust"
    graph_topic_dir = arts_root / "graph" / "rich"  # the single RICH topic's nodes/edges
    trustrank.run(
        graph_dir=graph_topic_dir,
        prior_path=trust_dir / "metadata_prior.jsonl",
        out_path=trust_dir / "metadata_trust.jsonl",
    )


if __name__ == "__main__":
    run()
