"""Publish and retrieve final production document trust scores."""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path

def run() -> None:
    """Calculate and publish ``metadata_trust.jsonl``."""
    from dc_available.trust_score import trustrank

    trustrank.run()


_DEFAULT_TRUST_FILE = (
    Path(__file__).resolve().parent.parent
    / "data" / "Artifacts" / "resources" / "trust" / "grownfield" / "metadata_trust.jsonl"
)


def _trust_file() -> Path:
    env = os.environ.get("TRUST_FILE")
    return Path(env) if env else _DEFAULT_TRUST_FILE


@functools.lru_cache(maxsize=8)
def _trust_table(path_str: str) -> tuple[dict[str, float], float]:
    """Load and cache a final trust table per file path."""
    table: dict[str, float] = {}
    path = Path(path_str)
    if path.is_file():
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            final_trust = row.get("final_trust")
            if final_trust is not None:
                table[str(row["match_key"]).strip().lower()] = float(final_trust)
    mean = sum(table.values()) / len(table) if table else 1.0
    return table, mean


def trust_for(
    part_of_document: str | None = None,
    storage_path: str | None = None,
    trust_file: str | os.PathLike | None = None,
) -> float:
    """Return final document trust; use the table mean if no key matches.

    ``trust_file`` selects which ``metadata_trust.jsonl`` to read (defaults to the QA
    grownfield file / the ``TRUST_FILE`` env). A separate source such as RICH passes its own
    file so QA and RICH can each resolve trust in the same process. Tables are cached per path.
    """
    table, mean = _trust_table(str(trust_file) if trust_file else str(_trust_file()))
    if part_of_document:
        value = table.get(str(part_of_document).strip().lower())
        if value is not None:
            return value
    if storage_path:
        value = table.get(Path(str(storage_path)).stem.strip().lower())
        if value is not None:
            return value
    return mean


if __name__ == "__main__":
    run()