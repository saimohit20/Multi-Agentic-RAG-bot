"""Production orchestration for the AVAILABLE trust-score build."""

from dc_available.trust_score import (
    align_metadata,
    collect_metadata,
    final_trust,
    prior_score,
)


def run() -> None:
    """Build final document trust after the graph index has been created."""
    print("\n[Auto] Building trust scores...")
    collect_metadata.collect()
    align_metadata.align()
    prior_score.run()
    final_trust.run()
    print("[Auto] Trust build complete.")