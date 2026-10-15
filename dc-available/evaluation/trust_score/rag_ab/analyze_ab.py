"""Step 8 - summarise the 3-arm A/B ablation.

Layers:
  RETRIEVAL   - % good_doc (good ranked ABOVE its twin), per arm + lever; McNemar.
  GENERATION  - PRIMARY: exact-match %correct (gold value present), per arm + lever;
                McNemar. Plus %twin (followed the wrong value).
  VALIDATION  - own gpt-4o LLM judge %correct per arm (confirms exact match).
  METRIC CHECK- agreement of each grader vs the exact-match ground truth
                (LLM judge should agree; RAGAS should not → it is unsuited to short
                factual answers).

Inputs : rag_ab/ab_off.jsonl, ab_prior.jsonl, ab_full.jsonl   (retrieval: good_doc)
         rag_ab/ab_gen.jsonl                                   (exact, twin, llm_judge)
         rag_ab/ab_scores.jsonl                                (RAGAS answer_correctness)
Output : console tables (+ bar chart under rag_ab/figures/)

RUN (from the dc-available directory):
    python -m evaluation.trust_score.rag_ab.analyze_ab
"""
from __future__ import annotations

import json
from collections import defaultdict

from scipy.stats import binomtest, chi2

from evaluation.trust_score.rag_ab.config import (
    AB_ARM_FILES,
    AB_GEN_FILE,
    FIGURES_DIR,
)

ARMS = ["off", "prior", "full"]
LEVERS = ["freshness", "authority"]
PAIRS = [("off", "prior"), ("prior", "full"), ("off", "full")]


def _pct(num, den) -> float:
    return 100.0 * num / den if den else 0.0


def _mcnemar(cx: dict, cy: dict) -> dict:
    """Full paired 2x2 for two arms X, Y (over shared questions):
       a = both success, b = X only, c = Y only, d = both fail.
    Returns counts + McNemar chi-square (continuity-corrected) and exact-binomial p."""
    common = set(cx) & set(cy)
    a = sum(1 for q in common if cx[q] and cy[q])
    b = sum(1 for q in common if cx[q] and not cy[q])
    c = sum(1 for q in common if not cx[q] and cy[q])
    d = sum(1 for q in common if not cx[q] and not cy[q])
    n = b + c
    chi2_stat = ((abs(b - c) - 1) ** 2) / n if n else 0.0
    p_chi2 = float(chi2.sf(chi2_stat, 1)) if n else 1.0
    p_exact = binomtest(min(b, c), n, 0.5).pvalue if n else 1.0
    return {"a": a, "b": b, "c": c, "d": d,
            "chi2": chi2_stat, "p_chi2": p_chi2, "p_exact": p_exact}


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def _load_arms() -> dict[str, list]:
    out = {}
    for arm in ARMS:
        path = AB_ARM_FILES[arm]
        out[arm] = [json.loads(l) for l in path.open(encoding="utf-8")] if path.is_file() else []
    return out


def _load_gen() -> dict[tuple, dict]:
    """(question_id, arm) -> unified grader row {exact, twin, llm_judge, answer_correctness, ragas}."""
    out = {}
    if AB_GEN_FILE.is_file():
        for line in AB_GEN_FILE.open(encoding="utf-8"):
            r = json.loads(line)
            out[(r["question_id"], r["arm"])] = r
    return out


# --------------------------------------------------------------------------- #
# retrieval
# --------------------------------------------------------------------------- #
def _retrieval(by_arm) -> dict[str, float]:
    print("\n=== RETRIEVAL - which source ranks first (of n) ===\n")
    print(f"{'arm':6} {'n':>4} {'original':>9} {'poisoned':>9} {'neither':>8} {'%original':>10}")
    print("-" * 52)
    good_by: dict[str, dict] = defaultdict(dict)   # original (good above twin), for McNemar
    overall = {}
    for arm in ARMS:
        rs = by_arm.get(arm, [])
        if not rs:
            continue
        for r in rs:
            good_by[arm][r["question_id"]] = bool(r["good_doc"])
        n = len(rs)
        original = sum(1 for r in rs if r["good_doc"])                    # good above twin
        poisoned = sum(1 for r in rs if r["twin_rank"] is not None
                       and (r["good_rank"] is None or r["twin_rank"] < r["good_rank"]))
        neither = sum(1 for r in rs if r["good_rank"] is None and r["twin_rank"] is None)
        overall[arm] = _pct(original, n)
        print(f"{arm:6} {n:>4} {original:>9} {poisoned:>9} {neither:>8} {overall[arm]:>9.1f}%")
    print("\n(original = good doc ranked above twin; poisoned = twin ranked above good; "
          "neither = neither retrieved; sums to n.  %original = original/n)")
    _pairwise_mcnemar(good_by, "original above poisoned (retrieval)", overview=True)
    return overall


# --------------------------------------------------------------------------- #
# generation (exact match = primary)
# --------------------------------------------------------------------------- #
def _grader_comparison(gen) -> None:
    """Compare the 3 graders, then justify using exact for the outcome table."""
    graders = ["exact", "llm_judge", "ragas"]
    # Table A - %correct per arm, per grader
    print("\n\n=== GENERATION - grader comparison ===\n")
    print("Table A - %correct per arm:\n")
    print(f"{'grader':10} " + " ".join(f"{a:>8}" for a in ARMS))
    print("-" * 36)
    for key in graders:
        cells = []
        for arm in ARMS:
            rows = [v for (q, a), v in gen.items() if a == arm and v.get(key) is not None]
            cells.append(f"{_pct(sum(bool(v[key]) for v in rows), len(rows)):>7.1f}%"
                         if rows else f"{'-':>8}")
        print(f"{key:10} " + " ".join(cells))

    # Table B - agreement with exact-match ground truth (all arms pooled)
    print("\nTable B - agreement with exact-match ground truth (all arms):\n")
    print(f"{'grader':12} {'n':>5} {'agreement':>10}")
    print("-" * 30)
    keys = [k for k, v in gen.items() if v.get("exact") is not None]
    for key in ("llm_judge", "ragas"):
        ks = [k for k in keys if gen[k].get(key) is not None]
        agree = sum(1 for k in ks if bool(gen[k][key]) == bool(gen[k]["exact"]))
        print(f"{key:12} {len(ks):>5} {_pct(agree, len(ks)):>9.1f}%")
    print("\n(exact = ground truth; high llm_judge + low RAGAS => RAGAS unsuited to short answers,\n"
          " so the outcome table below uses exact match)")


def _generation(gen) -> dict[str, dict]:
    """Exact-match outcome table (4-way, mutually exclusive) + McNemar. Returns
    the per-arm breakdown for the stacked-bar plot."""
    print("\n\n=== GENERATION - outcomes (exact match) [PRIMARY] ===\n")
    print(f"{'arm':6} {'n':>4} {'correct':>8} {'poisoned':>9} {'both':>6} {'missing':>8}")
    print("-" * 46)
    breakdown: dict[str, dict] = {}
    exact_by: dict[str, dict] = defaultdict(dict)   # success = gold value present
    for arm in ARMS:
        rows = [v for (q, a), v in gen.items() if a == arm]
        if not rows:
            continue
        for v in rows:
            exact_by[arm][v["question_id"]] = bool(v["exact"])
        n = len(rows)
        correct = sum(1 for v in rows if v["exact"] and not v["twin"])   # gold only
        poisoned = sum(1 for v in rows if v["twin"] and not v["exact"])  # wrong only
        both = sum(1 for v in rows if v["exact"] and v["twin"])          # both values
        missing = sum(1 for v in rows if not v["exact"] and not v["twin"])
        breakdown[arm] = {"n": n, "correct": correct, "poisoned": poisoned,
                          "both": both, "missing": missing}
        print(f"{arm:6} {n:>4} {correct:>8} {poisoned:>9} {both:>6} {missing:>8}")
    print("\n(correct=gold only, poisoned=wrong only, both=both values present, "
          "missing=neither; sums to n)")
    _pairwise_mcnemar(exact_by, "exact match (correct = gold value present)", overview=True)
    return breakdown


def _pairwise_mcnemar(binary_by: dict[str, dict], label: str, overview: bool = False) -> None:
    print(f"\nMcNemar - {label} (paired 2x2 per arm pair):")
    results = []
    for x, y in PAIRS:
        cx, cy = binary_by.get(x), binary_by.get(y)
        if not (cx and cy):
            continue
        results.append((x, y, _mcnemar(cx, cy)))

    if overview:
        print("\n  OVERVIEW:")
        print(f"    {'comparison':14}{'b':>5}{'c':>5}{'chi2':>9}{'p_exact':>10}   better")
        for x, y, m in results:
            b, c = m["b"], m["c"]
            better = y if c > b else (x if b > c else "tie")
            print(f"    {x+' vs '+y:14}{b:>5}{c:>5}{m['chi2']:>9.3f}{m['p_exact']:>10.4f}   {better}")
        print("\n  DETAIL:")

    for x, y, m in results:
        b, c = m["b"], m["c"]
        better = y if c > b else (x if b > c else "tie")
        sig = "significant" if m["p_exact"] < 0.05 else "n.s."
        print(f"\n  {x} (X) vs {y} (Y):")
        print(f"    {'':14}{'Y success':>12}{'Y failure':>12}")
        print(f"    {'X success':14}{m['a']:>12}{m['b']:>12}")
        print(f"    {'X failure':14}{m['c']:>12}{m['d']:>12}")
        print(f"    discordant: b={b} (X only), c={c} (Y only)  ->  better: {better}")
        print(f"    chi2(cc)={m['chi2']:.3f}  p_chi2={m['p_chi2']:.4f}  "
              f"p_exact={m['p_exact']:.4f}  [{sig}]")


def run(plot: bool = True) -> None:
    by_arm = _load_arms()
    gen = _load_gen()
    if not any(by_arm.values()):
        print("No arm files found - run run_ab.py first.")
        return

    _retrieval(by_arm)
    breakdown = {}
    if gen:
        _grader_comparison(gen)
        breakdown = _generation(gen)
    else:
        print("\n(no ab_gen.jsonl yet - run score_generation.py for the generation layer)")

    if plot and breakdown:
        _plot(breakdown)


def _plot(breakdown) -> None:
    """4-way stacked bar per arm: correct / poisoned / both / missing (out of n)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\n(matplotlib not available - skipping plot)")
        return

    arms = [a for a in ARMS if a in breakdown]
    x = np.arange(len(arms))
    segments = [("correct", "#2ca02c"), ("poisoned", "#d62728"),
                ("both", "#ff7f0e"), ("missing", "#999999")]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bottom = np.zeros(len(arms))
    for key, color in segments:
        vals = np.array([breakdown[a][key] for a in arms], dtype=float)
        bars = ax.bar(x, vals, bottom=bottom, color=color, label=key)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + v / 2,
                        f"{int(v)}", ha="center", va="center", fontsize=8, color="white")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylabel("questions (of n)")
    ax.set_title("Generation outcomes by arm (exact match)")
    ax.legend(loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "ab_generation.png"
    fig.savefig(path, dpi=150)
    print(f"\nplot -> {path}")


if __name__ == "__main__":
    run()
