"""Stage A (RICH) — collect RAW SharePoint metadata for every RICH interview PDF.

The RICH analogue of ``collect_metadata.py`` (which is GROWNFIELD-only and also
resolves .lnk/.url shortcuts). RICH is simpler: the source is a flat set of
interview files on RICH's own SharePoint folder, so we just walk it, keep the
interview documents (``.pdf``/``.docx``), and write one raw-metadata row each.

No scoring here — Stage B (prior) reads this later. Reuses
``common/sharepoint_connector.py`` (the same Graph auth/walk QA uses); nothing
talks to Graph directly except through that module.

Source resolution comes from the ``rich`` entry in data_config:
    site   = env named by rich["site_env"]        (e.g. AVAILABLE_SITE_ID)
    folder = env named by rich["remote_folder_env"] (e.g. RICH_FOLDER_PATH)

Output: Artifacts/rich/trust/metadata.jsonl
Join key = filename STEM, which matches RICH's ``part_of_document``.

RUN (from the dc-available directory, with the RICH Graph envs + creds set):
    python -m dc_available.trust_score.collect_rich_metadata
"""
from __future__ import annotations

import json
from pathlib import Path

from dc_available.common import sharepoint_connector as sp
from dc_available.config import data_config


def _rich_trust_dir() -> Path:
    """Artifacts/rich/trust (sibling of ingestion/graph)."""
    return data_config.source_artifact_paths("rich")["ingestion"].parent / "trust"


def _stem(name: str) -> str:
    """Filename without its extension, lower-cased (the join key)."""
    return name.rsplit(".", 1)[0].strip().lower() if "." in name else name.strip().lower()


def _ext(name: str) -> str:
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def _person(node: dict | None) -> str:
    if not node:
        return ""
    user = node.get("user") or {}
    return user.get("displayName") or user.get("email") or ""


# Interview source formats we collect (interviews 1-5 are PDFs, 6-7 are DOCX).
_WANTED_EXTS = {"pdf", "docx"}


def _row(name: str, rel: str, item: dict) -> dict:
    """Build one raw-metadata row from an interview DriveItem (pdf/docx)."""
    file_info = item.get("file") or {}
    return {
        "match_key": _stem(name),          # join to local by stem (== part_of_document)
        "source_name": name,               # file name in the RICH folder
        "source_rel": rel,                 # path within the RICH folder
        "file_type": _ext(name),           # "pdf"
        "mime_type": file_info.get("mimeType", ""),
        "created": item.get("createdDateTime", ""),
        "modified": item.get("lastModifiedDateTime", ""),
        "created_by": _person(item.get("createdBy")),
        "modified_by": _person(item.get("lastModifiedBy")),
        "size": item.get("size", 0),
        "web_url": item.get("webUrl", ""),
        "graph_item_id": item.get("id", ""),
    }


def collect() -> None:
    rich_cfg = data_config._data_config["rich"]
    site_env = rich_cfg.get("site_env", "AVAILABLE_SITE_ID")
    folder_env = rich_cfg.get("remote_folder_env", "RICH_FOLDER_PATH")

    sid = sp.cfg(site_env)
    if not sid:
        raise RuntimeError(
            f"RICH site id env {site_env!r} is not set; cannot reach SharePoint."
        )
    folder = sp.cfg(folder_env)
    did = sp.drive_id(sid)
    print(f"Walking RICH source: site={site_env} folder={folder or '<root>'}")

    rows: list[dict] = []
    skipped_other = 0
    for item, rel in sp.list_files(did, folder):
        if _ext(item["name"]) not in _WANTED_EXTS:
            skipped_other += 1
            continue
        rows.append(_row(item["name"], rel, item))

    out_dir = _rich_trust_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "metadata.jsonl"
    with out_file.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    with_created_by = sum(1 for r in rows if r["created_by"])
    with_modified = sum(1 for r in rows if r["modified"])
    print(f"Done. {len(rows)} interview file(s) -> {out_file}  (skipped {skipped_other} other)")
    print(
        "  pillar signal: "
        f"created_by present on {with_created_by}/{len(rows)}, "
        f"modified present on {with_modified}/{len(rows)}"
    )


if __name__ == "__main__":
    collect()
