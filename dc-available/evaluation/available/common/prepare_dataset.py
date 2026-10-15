"""Prepare an evaluation dataset: fill Level / Type from the corpus, validate sources.

Reads an .xlsx/.csv with columns (Question, Answer, Source) and:
  - resolves each Source module code -> level (Basic/Advanced/Expert) + media type
    (pptx/video/pdf/docx/...) by scanning the corpus folders,
  - drops source codes not found in the corpus (and drops a row only if NONE of its
    sources are found),
  - fills Level, Type, question_id, and (for grownfield) Question Type from GF_LABELS,
  - writes <input>_prepared.xlsx.

Usage:
  python evaluation/prepare_dataset.py <input.xlsx> [output.xlsx]
"""
from __future__ import annotations

import os
import re
import sys

import pandas as pd

CORPUS_ROOTS = [
    "dc_available/data/local_sharepoint/GROWNFIELD",
    "dc_available/data/grownfield/organized",
]
EXT_TYPE = {".pptx": "pptx", ".mp4": "video", ".pdf": "pdf", ".docx": "docx", ".txt": "text"}
LEVELS = {"Basic", "Advanced", "Expert"}
CODE_RE = re.compile(r"[A-Z0-9]{3}-\d{2}\.\d{3}")

# Manual question_type labels for the grownfield set, keyed by the '#' column.
GF_LABELS = {
    1: "factual", 2: "factual", 3: "conceptual", 4: "procedural", 5: "conceptual",
    6: "factual", 7: "procedural", 8: "procedural", 9: "procedural", 10: "conceptual",
    11: "procedural", 12: "procedural", 13: "procedural", 14: "factual", 15: "procedural",
    16: "procedural", 17: "procedural",
}


def build_corpus_map() -> dict[str, dict]:
    cmap: dict[str, dict] = {}
    for root in CORPUS_ROOTS:
        if not os.path.isdir(root):
            continue
        for dp, _, files in os.walk(root):
            level = next((p for p in dp.split(os.sep) if p in LEVELS), None)
            for f in files:
                m = CODE_RE.search(f)
                t = EXT_TYPE.get(os.path.splitext(f)[1].lower())
                if not m or t is None:
                    continue
                e = cmap.setdefault(m.group(0), {"level": None, "types": set()})
                e["types"].add(t)
                if level and not e["level"]:
                    e["level"] = level
    return cmap


def split_codes(raw) -> list[str]:
    return CODE_RE.findall(str(raw or ""))


def main() -> None:
    inp = sys.argv[1] if len(sys.argv) > 1 else (
        r"evaluation/datasets/Grownfield/Dataset.xlsx"
    )
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(inp)[0] + "_prepared.xlsx"

    df = pd.read_excel(inp) if inp.lower().endswith(".xlsx") else pd.read_csv(inp)
    cmap = build_corpus_map()

    rows = []
    dropped_rows, dropped_codes = [], []
    for _, r in df.iterrows():
        num = r.get("#")
        codes = split_codes(r.get("Source"))
        found = [c for c in codes if c in cmap]
        missing = [c for c in codes if c not in cmap]
        dropped_codes += [(num, c) for c in missing]
        if not found:
            dropped_rows.append((num, r.get("Question")))
            continue
        levels = [cmap[c]["level"] for c in found if cmap[c]["level"]]
        types = sorted({t for c in found for t in cmap[c]["types"]})
        rows.append({
            "question_id": f"gf_{int(num):03d}" if pd.notna(num) else "",
            "question": r.get("Question"),
            "golden_answer": r.get("Answer"),
            "golden_source": "\n".join(found),
            "level": levels[0] if levels else "",
            "type": ", ".join(types),
            "question_type": GF_LABELS.get(int(num)) if pd.notna(num) else "",
        })

    out_df = pd.DataFrame(rows)
    out_df.to_excel(out, index=False)

    print(f"Rows in: {len(df)} | rows out: {len(out_df)}")
    if dropped_codes:
        print("Dropped source codes (not found in corpus):")
        for num, c in dropped_codes:
            print(f"  row {num}: {c}")
    if dropped_rows:
        print("Dropped whole rows (no source found):")
        for num, q in dropped_rows:
            print(f"  row {num}: {str(q)[:60]}")
    print(f"Wrote -> {out}")


if __name__ == "__main__":
    main()
