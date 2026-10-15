"""Step 1 — the 20 'good' documents for the A/B poisoning test (LOCKED list).

Hand-curated after reading the summaries: the 20 most fact-dense docs (Quick
Reference Cards + detailed GUI/User-Guide docs — full of transaction codes,
values, IDs to swap). Prose .pptx overview decks and German duplicates were
excluded. Order = richness tiers; levers alternate so both groups (freshness /
authority) get a mix.

Input : metadata_trust.jsonl  (+ the __summary.txt files)
Output: rag_ab/docs.jsonl     (20 rows in g01..g20 order)

RUN (from the dc-available directory):
    python -m evaluation.trust_score.rag_ab.select_docs
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from dc_available.trust_score.components.score_content import _read_text, _summary_index
from evaluation.trust_score.rag_ab.config import AB_DIR, BASELINE_FILE, DOCS_FILE

# The locked 20, by exact summary filename, in richness order (g01..g20).
LOCKED = [
    "Q2E-02.031 Quick Reference Card Q2E - Inspection Plan.pdf__summary.txt",
    "I2P-01.011_IM DC User Guide.pdf__summary.txt",
    "B2P-02.020 PCP Toolset.html__summary.txt",
    "B2P-06.020 Execute cost center planning.html__summary.txt",
    "B2P-06.040 Execute Internal Order Planning.html__summary.txt",
    "B2P-01.030 GUI PPC serial production - Preparation steps 2.html__summary.txt",
    "P2P-02.022 Quick Reference Card P2P - Logistic and Purchase Elements.pdf__summary.txt",
    "P2P-04.014 Quick Reference Card P2P - Goods Receipt Inspection.pdf__summary.txt",
    "O2C-02.000 Quick Reference Card O2C- O2S.pdf__summary.txt",
    "O2C-03.000 Quick Reference Card O2C - S2B.pdf__summary.txt",
    "O2C-04.010 Manage Customer Billing - Initial Invoice.pptx__summary.txt",
    "P2P-02.012 Quick Reference Card P2P - Material Planning.pdf__summary.txt",
    "STI-00.001 Quick Reference Card STI - Inbound EN.pdf__summary.txt",
    "STI-00.004 Quick Reference Card STI - Physical Inventory EN.pdf__summary.txt",
    "B2P-01.020 GUI Subs. Plan. - Simulation costing variants.html__summary.txt",
    "B2P-05.010 Analyzing and Posting of Inventory Differences.html__summary.txt",
    "I2O-02.010 Create Quotation.pptx__summary.txt",
    "O2C-04.001 Quick Reference Card O2C - B2C.pdf__summary.txt",
    "B2P-11.020 DC CO-P&L with fix-var.html__summary.txt",
    "STI-00.003 Quick Reference Card STI - Production Supply EN.pdf__summary.txt",
]

_KEEP = ("match_key", "topic", "file_type", "created_by", "modified_by",
         "modified", "content_score", "prior", "final_trust")
_FACT_RE = re.compile(r"\b[A-Z]{2,}[A-Z0-9_/]*\b|\b\d[\d.,]*\b")


def run() -> None:
    rows = [json.loads(line) for line in BASELINE_FILE.open(encoding="utf-8")]
    summaries = _summary_index()

    # index every llm-scored doc by its summary filename
    by_summary_name: dict[str, dict] = {}
    for r in rows:
        if r.get("content_source") != "llm":
            continue
        sp = summaries.get(r["match_key"])
        if sp is None:
            continue
        by_summary_name[Path(sp).name] = (r, sp)

    picked, missing = [], []
    for n, name in enumerate(LOCKED, 1):
        hit = by_summary_name.get(name)
        if hit is None:
            missing.append(name)
            continue
        r, sp = hit
        keep = {k: r.get(k) for k in _KEEP}
        keep["doc_id"] = f"g{n:02d}"
        keep["lever"] = "freshness" if n % 2 == 1 else "authority"   # alternate 10/10
        keep["summary_path"] = str(sp)
        keep["richness"] = len({t for t in _FACT_RE.findall(_read_text(sp)) if len(t) >= 2})
        picked.append(keep)

    AB_DIR.mkdir(parents=True, exist_ok=True)
    with DOCS_FILE.open("w", encoding="utf-8") as fh:
        for r in picked:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{'id':4} {'rich':>4} {'lever':10} {'topic':5} match_key")
    print("-" * 78)
    for r in picked:
        print(f"{r['doc_id']:4} {r['richness']:>4} {r['lever']:10} "
              f"{str(r.get('topic')):5} {r['match_key'][:42]}")
    n_fresh = sum(1 for r in picked if r["lever"] == "freshness")
    print(f"\nlocked {len(picked)}/20 -> {DOCS_FILE.name} "
          f"({n_fresh} freshness / {len(picked) - n_fresh} authority)")
    if missing:
        print("MISSING (filename not matched):")
        for m in missing:
            print(f"  - {m}")


if __name__ == "__main__":
    run()
