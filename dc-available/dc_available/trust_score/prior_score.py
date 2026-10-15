"""Build the query-independent trust prior.

This production stage groups content scoring, metadata/content merging, and
four-pillar prior calculation behind one entry point. The underlying modules
remain importable because trust-score evaluations exercise those exact scorers.
"""

from dc_available.trust_score.components import merge_scores, score_content, score_prior


def run() -> None:
    """Build content scores, merge them with metadata, then calculate priors."""
    score_content.run()
    merge_scores.merge()
    score_prior.run()


if __name__ == "__main__":
    run()