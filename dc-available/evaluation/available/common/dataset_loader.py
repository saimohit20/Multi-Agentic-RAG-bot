from __future__ import annotations

from pathlib import Path
import logging

import pandas as pd

logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = {"question", "ground_truth"}


def load_qa_dataset(path: str | Path) -> list[dict]:
    """Load QA evaluation dataset from an Excel file.

    Expected columns: 'question', 'ground_truth'. Optional: 'module' (the golden
    source, matched as a substring against retrieved source paths during evaluation).
    Returns a list of dicts with keys: question, ground_truth, golden_source
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [c.strip().lower() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in dataset: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    has_module = "module" in df.columns
    records = []
    for _, row in df.iterrows():
        question = str(row["question"]).strip()
        ground_truth = str(row["ground_truth"]).strip()

        if not question or question.lower() == "nan":
            continue

        golden_source = str(row["module"]).strip() if has_module else ""
        if golden_source.lower() == "nan":
            golden_source = ""

        records.append({
            "question": question,
            "ground_truth": ground_truth if ground_truth.lower() != "nan" else "",
            "golden_source": golden_source,
        })

    print(f"Loaded {len(records)} questions from {path.name}")
    return records
