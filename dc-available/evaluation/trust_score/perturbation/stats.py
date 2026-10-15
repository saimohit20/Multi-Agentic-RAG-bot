"""Step 5 — statistical significance of the CONTENT perturbations.

Only the 3 LLM-scored content perturbations (procedural, technical, filler) are
tested. authority/freshness are deterministic formula changes, so a significance
test there is trivial and meaningless.

The design is PAIRED: every doc has a before (original) and after (perturbed)
score on the pillar the perturbation attacks. For each content perturbation we
report, over the 40 paired observations:

    * n and % of docs that dropped          -> how consistent
    * median / mean Δ                        -> how big
    * 95% bootstrap CI on the mean Δ         -> is the effect away from 0
    * Wilcoxon signed-rank, one-sided        -> is the drop real (p-value)
      (alternative "after < before")
    * Cohen's d (paired)                     -> effect size
    * Benjamini-Hochberg adjusted p          -> corrected for the 3 tests

Input : perturbation/results.jsonl
Output: console table (+ perturbation/stats.jsonl)

RUN (from the dc-available directory):
    python -m evaluation.trust_score.perturbation.stats
"""
from __future__ import annotations

import json
from statistics import mean, median, stdev

import numpy as np
from scipy.stats import wilcoxon

from evaluation.trust_score.perturbation.config import (
    CONTENT_TYPES,
    RESULTS_FILE,
    TARGET_PILLAR,
)

STATS_FILE = RESULTS_FILE.parent / "stats.jsonl"
_BOOTSTRAP = 10000
_SEED = 42


def _bootstrap_ci(diffs: list[float]) -> tuple[float, float]:
    rng = np.random.default_rng(_SEED)
    arr = np.asarray(diffs, dtype=float)
    means = rng.choice(arr, size=(_BOOTSTRAP, arr.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _cohens_d(diffs: list[float]) -> float:
    s = stdev(diffs) if len(diffs) > 1 else 0.0
    return float(mean(diffs) / s) if s else 0.0


def _benjamini_hochberg(pvals: list[float]) -> list[float]:
    """Return BH-adjusted p-values (q-values)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])   # ascending
    adj = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, pvals[i] * m / rank)
        adj[i] = min(running, 1.0)
    return adj


def run() -> None:
    rows = [json.loads(line) for line in RESULTS_FILE.open(encoding="utf-8")]
    base = {r["base"]: r for r in rows if r["perturbation"] == "none"}

    results: list[dict] = []
    for t in CONTENT_TYPES:
        tp = TARGET_PILLAR[t]
        before, after = [], []
        for r in rows:
            if r["perturbation"] != t:
                continue
            before.append(base[r["base"]][tp])
            after.append(r[tp])
        diffs = [a - b for a, b in zip(after, before)]
        n = len(diffs)
        n_drop = sum(1 for d in diffs if d < 0)
        lo, hi = _bootstrap_ci(diffs)

        # one-sided Wilcoxon signed-rank: is (after < before)?
        try:
            _, p = wilcoxon(after, before, alternative="less", zero_method="wilcox")
            p = float(p)
        except ValueError:   # e.g. all differences are zero
            p = 1.0

        results.append({
            "perturbation": t, "target": tp, "n": n,
            "n_dropped": n_drop, "pct_dropped": round(100 * n_drop / n, 1) if n else 0.0,
            "mean_delta": round(mean(diffs), 4) if diffs else 0.0,
            "median_delta": round(median(diffs), 4) if diffs else 0.0,
            "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
            "cohens_d": round(_cohens_d(diffs), 3),
            "wilcoxon_p": p,
        })

    for r, q in zip(results, _benjamini_hochberg([r["wilcoxon_p"] for r in results])):
        r["wilcoxon_p_bh"] = round(q, 4)

    print("\nStatistical significance — content perturbations (paired, target pillar):\n")
    print(f"{'perturbation':12} {'target':21} {'dropped':>8} {'medianΔ':>9} "
          f"{'95% CI':>19} {'d':>6} {'p(BH)':>9}")
    print("-" * 90)
    for r in results:
        ci = f"[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]"
        dropped = f"{r['n_dropped']}/{r['n']}"
        sig = " *" if r["wilcoxon_p_bh"] < 0.05 else ""
        print(f"{r['perturbation']:12} {r['target']:21} {dropped:>8} "
              f"{r['median_delta']:>+9.3f} {ci:>19} {r['cohens_d']:>+6.2f} "
              f"{r['wilcoxon_p_bh']:>9.4f}{sig}")
    print("\n* = statistically significant drop (BH-adjusted p < 0.05).")
    print("Read: p(BH) = is the drop real?  |  dropped = how consistent?  |  d = how big?")

    with STATS_FILE.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nsaved -> {STATS_FILE.name}")


if __name__ == "__main__":
    run()
