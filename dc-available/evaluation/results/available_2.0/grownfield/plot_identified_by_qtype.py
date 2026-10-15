"""
Grouped + stacked bar chart:
  x-axis  : question_type
  y-axis  : number of questions
  groups  : retrieval method (vector / graph / hybrid)
  stack   : identified (golden_rank is not None) vs not identified (golden_rank is None)
"""
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

HERE = __import__("os").path.dirname(__file__)
DATA = __import__("os").path.join(HERE, "grownfield.json")

with open(DATA, encoding="utf-8") as f:
    rows = json.load(f)

QTYPES = ["factual", "conceptual", "procedural", "application"]
METHODS = ["vector", "graph", "hybrid"]

# counts[qtype][method] = [identified, not_identified]
counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for r in rows:
    qt, m = r["question_type"], r["retrieval_type"]
    if r.get("golden_rank") is not None:
        counts[qt][m][0] += 1   # identified
    else:
        counts[qt][m][1] += 1   # not identified

x = np.arange(len(QTYPES))
n_methods = len(METHODS)
bar_w = 0.8 / n_methods

C_ID = "#2c7fb8"     # identified
C_MISS = "#d95f5f"   # not identified

fig, ax = plt.subplots(figsize=(10, 6))

for j, method in enumerate(METHODS):
    offset = (j - (n_methods - 1) / 2) * bar_w
    ident = [counts[qt][method][0] for qt in QTYPES]
    miss = [counts[qt][method][1] for qt in QTYPES]
    ax.bar(x + offset, ident, bar_w, color=C_ID,
           edgecolor="white", linewidth=0.5)
    ax.bar(x + offset, miss, bar_w, bottom=ident, color=C_MISS,
           edgecolor="white", linewidth=0.5)
    # method label under each bar group
    for xi in x:
        ax.text(xi + offset, -0.3, method, ha="center", va="top",
                fontsize=8, rotation=90, color="#444")

# per-type totals annotated on top
for xi, qt in zip(x, QTYPES):
    total = sum(counts[qt][METHODS[0]])  # same total across methods
    ax.text(xi, max(sum(counts[qt][m]) for m in METHODS) + 0.15,
            f"n={total}", ha="center", va="bottom", fontsize=9, color="#666")

ax.set_xticks(x)
ax.set_xticklabels([qt.capitalize() for qt in QTYPES], fontsize=12, fontweight="bold")
ax.tick_params(axis="x", pad=48, length=0)
ax.set_ylabel("Number of questions")
ax.set_title("Golden-source retrieval by question type and method (Grownfield)")

# legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C_ID, label="Identified"),
                   Patch(color=C_MISS, label="Not identified")],
          loc="upper right", frameon=False)

ax.set_ylim(0, max(sum(counts[qt][m]) for qt in QTYPES for m in METHODS) + 1)
ax.margins(x=0.02)
ax.spines[["top", "right"]].set_visible(False)
plt.subplots_adjust(bottom=0.18)

out = __import__("os").path.join(HERE, "identified_by_qtype.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
print("saved:", out)
