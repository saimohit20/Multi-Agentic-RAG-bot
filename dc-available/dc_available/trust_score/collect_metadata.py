"""Stage A — collect RAW SharePoint metadata for every GROWNFIELD document.

No scoring here. We walk the GROWNFIELD *source* folder once, resolve shortcuts,
and write one raw-metadata row per document to a single metadata.jsonl. The
scorer (Stage B) reads that file later; keeping collection dumb means we can
re-tune scoring without re-walking SharePoint.

Resolution rules (decided with the user):
    * regular file (.pptx/.mp4/.html/...) -> its own DriveItem metadata
    * .lnk (Verknüpfung)  -> resolve to the REAL target file, take ITS metadata
    * .url (Docupedia)    -> use the shortcut's OWN metadata (do NOT resolve)

Join key = filename STEM (extension stripped), because the pipeline restructures
files on the way to local storage, so folder paths don't line up but the name /
training-code does.

Reuses dc_available/common/sharepoint_connector.py (walk, auth, .lnk resolution) —
nothing here talks to Graph directly except through that module.

RUN (from the dc-available directory):
    python -m dc_available.trust_score.collect_metadata
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# All Graph/auth/walk/.lnk logic already lives here; importing it also loads
# pipeline/.env (the GROWNFIELD source config) at import time.
from dc_available.common import sharepoint_connector as sp

# Output lands next to the other GROWNFIELD artifacts.
_OUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "data" / "Artifacts" / "resources" / "trust" / "grownfield"
)
_METADATA_FILE = _OUT_DIR / "metadata.jsonl"
_UNRESOLVED_FILE = _OUT_DIR / "unresolved.jsonl"

# Training code at the start of a name, e.g. "O2C-03.001", "B2P-01.020".
_CODE_RE = re.compile(r"^([A-Za-z0-9]+-\d+\.\d+)")


# --------------------------------------------------------------------------- #
# Field extraction (raw only, no scoring)
# --------------------------------------------------------------------------- #
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


def _training_code(name: str) -> str:
    m = _CODE_RE.match(name.strip())
    return m.group(1) if m else ""


def _row(source_name: str, source_rel: str, source_type: str,
         meta_item: dict, resolved_name: str) -> dict:
    """Build one raw-metadata row from the DriveItem we chose to trust."""
    file_info = meta_item.get("file") or {}
    return {
        "match_key": _stem(resolved_name),        # join to local by stem
        "training_code": _training_code(resolved_name),
        "source_type": source_type,               # file | lnk | url
        "source_name": source_name,               # name in the approved folder
        "source_rel": source_rel,                 # path within source folder
        "resolved_name": resolved_name,           # real target name (== source for file/url)
        "file_type": _ext(resolved_name),
        "mime_type": file_info.get("mimeType", ""),
        "created": meta_item.get("createdDateTime", ""),
        "modified": meta_item.get("lastModifiedDateTime", ""),
        "created_by": _person(meta_item.get("createdBy")),
        "modified_by": _person(meta_item.get("lastModifiedBy")),
        "size": meta_item.get("size", 0),
        "web_url": meta_item.get("webUrl", ""),
        "graph_item_id": meta_item.get("id", ""),
    }


# --------------------------------------------------------------------------- #
# Main walk
# --------------------------------------------------------------------------- #
def collect() -> None:
    did = sp.drive_id(sp.site_id("SOURCE"))
    folder = sp.cfg("SOURCE_FOLDER_PATH")
    print(f"Walking GROWNFIELD source: {folder or '<root>'}")

    rows: list[dict] = []
    unresolved: list[dict] = []
    counts = {"file": 0, "lnk": 0, "url": 0, "lnk_unresolved": 0}

    for item, rel in sp.list_files(did, folder):
        name = item["name"]
        ext = _ext(name)

        if ext == "lnk":
            # Resolve to the real target file and take ITS metadata.
            target, real_name = sp.lnk_to_item(did, sp.get_bytes(did, item["id"]))
            if target is None:
                # Target not found in library — keep shortcut meta, but flag it.
                counts["lnk_unresolved"] += 1
                row = _row(name, rel, "lnk_unresolved", item, real_name or name)
                unresolved.append({"source_rel": rel, "wanted": real_name})
            else:
                counts["lnk"] += 1
                row = _row(name, rel, "lnk", target, real_name)

        elif ext == "url":
            # Use the shortcut's OWN metadata (no Confluence resolve).
            counts["url"] += 1
            row = _row(name, rel, "url", item, name)

        else:
            counts["file"] += 1
            row = _row(name, rel, "file", item, name)

        rows.append(row)

    _write(rows, unresolved)
    total = sum(counts.values())
    print(f"Done. {total} documents -> {_METADATA_FILE}")
    print(f"  files={counts['file']}  lnk={counts['lnk']}  url={counts['url']}"
          f"  lnk_unresolved={counts['lnk_unresolved']}")
    if unresolved:
        print(f"  {len(unresolved)} unresolved .lnk logged -> {_UNRESOLVED_FILE}")


def _write(rows: list[dict], unresolved: list[dict]) -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    with _METADATA_FILE.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if unresolved:
        with _UNRESOLVED_FILE.open("w", encoding="utf-8") as fh:
            for u in unresolved:
                fh.write(json.dumps(u, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    collect()
