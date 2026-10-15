"""Step 4 — summarise the perturbation effect.

For every perturbation we average, across the 40 docs, the change of each pillar
relative to *that doc's own* original ("none") row. The headline is the drop of
the TARGET pillar; ``prior`` is reported as a (diluted) secondary number. For
the content perturbations the full 4-sub-score delta vector doubles as a
specificity check — the targeted sub-score should drop most.

Input : perturbation/results.jsonl
Output: console table (+ optional bar chart under perturbation/figures/)

RUN (from the dc-available directory):
    python -m evaluation.trust_score.perturbation.analyze
"""
from __future__ import annotations

import json
from collections import defaultdict
from statistics import mean

from evaluation.trust_score.perturbation.config import (
    ALL_TYPES,
    FIGURES_DIR,
    RESULTS_FILE,
    TARGET_PILLAR,
)

CONTENT_SUBS = ["technical_depth", "process_completeness", "training_usefulness", "focus"]
PILLARS = CONTENT_SUBS + ["content_score", "authority_score", "freshness_score", "prior"]


def _deltas(rows: list[dict]) -> dict[str, dict[str, float]]:
    base = {r["base"]: r for r in rows if r["perturbation"] == "none"}
    out: dict[str, dict[str, float]] = {}
    for t in ALL_TYPES:
        acc: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            if r["perturbation"] != t:
                continue
            b = base[r["base"]]
            for p in PILLARS:
                acc[p].append(r[p] - b[p])
        out[t] = {p: mean(v) for p, v in acc.items()} if acc else {}
    return out


def _before_after(rows: list[dict]) -> dict[str, dict[str, tuple[float, float]]]:
    """Per perturbation, the average of the target pillar and the prior BEFORE
    (each doc's original) and AFTER the damage."""
    base = {r["base"]: r for r in rows if r["perturbation"] == "none"}
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for t in ALL_TYPES:
        acc: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
        for r in rows:
            if r["perturbation"] != t:
                continue
            b = base[r["base"]]
            for p in (TARGET_PILLAR[t], "prior"):
                acc[p][0].append(b[p])
                acc[p][1].append(r[p])
        out[t] = {p: (mean(o), mean(a)) for p, (o, a) in acc.items()}
    return out


def _print_before_after(ba: dict[str, dict[str, tuple[float, float]]]) -> None:
    # Table 1 — the pillar each perturbation attacked
    print("\nTable 1 — target pillar (did the attacked pillar drop?):\n")
    print(f"{'perturbation':12} {'target pillar':22} {'original':>9} {'after':>8} "
          f"{'change':>9} {'%drop':>8}")
    print("-" * 72)
    for t in ALL_TYPES:
        tp = TARGET_PILLAR[t]
        o, a = ba[t][tp]
        pct = (a - o) / o * 100 if o else 0.0
        print(f"{t:12} {tp:22} {o:9.3f} {a:8.3f} {a - o:+9.3f} {pct:+7.1f}%")

    # Table 2 — the overall trust prior
    print("\nTable 2 — overall trust prior (did the overall score drop?):\n")
    print(f"{'perturbation':12} {'original':>9} {'after':>8} {'change':>9} {'%drop':>8}")
    print("-" * 50)
    for t in ALL_TYPES:
        o, a = ba[t]["prior"]
        pct = (a - o) / o * 100 if o else 0.0
        print(f"{t:12} {o:9.3f} {a:8.3f} {a - o:+9.3f} {pct:+7.1f}%")


def _print_specificity(table: dict[str, dict[str, float]]) -> None:
    print("\nTable 3 — specificity (Δ of all 4 content sub-scores; target should drop most):\n")
    print(f"{'perturbation':12} " + " ".join(f"{s[:12]:>13}" for s in CONTENT_SUBS))
    print("-" * 68)
    for t in ("procedural", "technical", "filler"):
        d = table[t]
        print(f"{t:12} " + " ".join(f"{d.get(s, 0):+13.3f}" for s in CONTENT_SUBS))


def _plot(table: dict[str, dict[str, float]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available — skipping plot)")
        return
    labels = list(ALL_TYPES)
    dtarget = [table[t].get(TARGET_PILLAR[t], 0) for t in labels]
    dprior = [table[t].get("prior", 0) for t in labels]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([i - 0.2 for i in x], dtarget, 0.4, label="Δ target pillar")
    ax.bar([i + 0.2 for i in x], dprior, 0.4, label="Δ prior")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("mean change vs original")
    ax.set_title("Trust-score sensitivity to controlled perturbations")
    ax.legend()
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "perturbation_deltas.png"
    fig.savefig(path, dpi=150)
    print(f"\nplot -> {path}")


def run(plot: bool = True) -> None:
    rows = [json.loads(line) for line in RESULTS_FILE.open(encoding="utf-8")]
    n_base = sum(1 for r in rows if r["perturbation"] == "none")
    print(f"loaded {len(rows)} variants across {n_base} base docs")
    table = _deltas(rows)
    _print_before_after(_before_after(rows))
    _print_specificity(table)
    if plot:
        _plot(table)


if __name__ == "__main__":
    run()
