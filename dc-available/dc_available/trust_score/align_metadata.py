"""Align raw metadata to the actual documents (dedup + join to local storage).

The raw walk over-collects: many .lnk shortcuts point to the SAME target (dupes),
and the approved folder holds extras the pipeline never ingested. This step turns
the raw metadata.jsonl into ONE clean row per real document:

    raw metadata.jsonl (636)
       -> dedup by match_key         (collapse repeated shortcuts)
       -> inner-join to local stems  (drop source items with no local document)
       -> attach local_path + topic  (ready for the Stage-B scorer + graph join)
       -> metadata_aligned.jsonl (== number of local files)

Dedup rule: when a match_key repeats, prefer a resolved row over an
'lnk_unresolved' one, then keep the most recently modified.

RUN (from the dc-available directory):
    python -m dc_available.trust_score.align_metadata
"""
from __future__ import annotations

import json
from pathlib import Path

_DATA = Path(__file__).resolve().parent.parent / "data"
_RAW = _DATA / "Artifacts" / "resources" / "trust" / "grownfield" / "metadata.jsonl"
_OUT = _DATA / "Artifacts" / "resources" / "trust" / "grownfield" / "metadata_aligned.jsonl"
_LOCAL = _DATA / "local_sharepoint" / "GROWNFIELD"


def _local_index() -> dict[str, Path]:
    """Map filename stem (lower) -> local file path. Stems are unique in local."""
    idx: dict[str, Path] = {}
    for p in _LOCAL.rglob("*"):
        if p.is_file():
            idx.setdefault(p.stem.strip().lower(), p)
    return idx


def _pick(rows: list[dict]) -> dict:
    """Choose the winning row for one match_key."""
    # Prefer resolved rows; fall back to whatever we have.
    resolved = [r for r in rows if r.get("source_type") != "lnk_unresolved"] or rows
    # Most recently modified wins (ISO strings sort chronologically).
    return max(resolved, key=lambda r: r.get("modified", "") or "")


def align() -> None:
    raw = [json.loads(line) for line in _RAW.open(encoding="utf-8")]

    by_key: dict[str, list[dict]] = {}
    for r in raw:
        by_key.setdefault(r["match_key"], []).append(r)

    local = _local_index()

    out: list[dict] = []
    for key, rows in by_key.items():
        if key not in local:
            continue  # source extra with no local document -> drop
        row = _pick(rows)
        path = local[key]
        row["local_path"] = str(path)
        row["topic"] = path.relative_to(_LOCAL).parts[0]
        out.append(row)

    with _OUT.open("w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    matched_keys = {r["match_key"] for r in out}
    missing_local = len(set(local) - matched_keys)
    print(f"raw rows              : {len(raw)}")
    print(f"unique match_keys     : {len(by_key)}")
    print(f"local documents       : {len(local)}")
    print(f"aligned rows written  : {len(out)}  -> {_OUT.name}")
    print(f"dropped (no local doc): {len(by_key) - len(out)}")
    print(f"local files unmatched : {missing_local}")


if __name__ == "__main__":
    align()
