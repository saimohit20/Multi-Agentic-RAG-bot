"""Perturbation-sensitivity evaluation of the trust score.

A self-contained experiment: take approved documents, degrade them in one
controlled way each, re-score with the *real* trust-score code, and check that
the trust ``prior`` (and the specific pillar attacked) drops. This manufactures
ground truth in a corpus that is otherwise uniformly approved.

Scope (first pass): one strong severity per perturbation, PRIOR only (no
TrustRank), no validation LLM. Corrupt with gpt-5.2, score with the existing
gpt-4o scorer. All outputs live under
``data/Artifacts/resources/trust/grownfield/perturbation/`` — the real pipeline
artifacts are never touched.

Run order (from the dc-available directory):
    python -m evaluation.trust_score.perturbation.select_sample
    python -m evaluation.trust_score.perturbation.perturb
    python -m evaluation.trust_score.perturbation.score
    python -m evaluation.trust_score.perturbation.analyze
    python -m evaluation.trust_score.perturbation.stats
"""
