"""
pipeline/sync.py — STEP 2 of the pipeline.

Delta-pull a folder of the AVAILABLE SharePoint into a local mirror, so Step 3
(preprocess) can read it. Only newly added files are downloaded; files deleted in
the cloud are removed locally. Existing files are not re-downloaded even when their
eTag changes. A small manifest (relative path -> eTag) decides what changed.

Which folder is mirrored (and to where) is chosen by --source, resolved from
dc_available/config/data_config.py:
    available (default)  AVAILABLE_FOLDER_PATH  -> data/local_sharepoint
    rich                 RICH_FOLDER_PATH       -> data/local_sharepoint/rich
Both use the same site (AVAILABLE_SITE_ID) and credentials; each mirror keeps its
own .manifest.json so their delta-tracking never collides.

Quiet logging: only downloads / deletes / failures are printed; unchanged files
are counted, not listed.

Run:
    python -m pipeline.sync                 # source: available
    python -m pipeline.sync --source rich   # source: rich

Config (pipeline/.env, via dc_available/common/sharepoint_connector.py):
    AVAILABLE_SITE_ID      the AVAILABLE SharePoint site id
    AVAILABLE_FOLDER_PATH  the folder to mirror for `available` (…/DC_DATA)
    RICH_FOLDER_PATH       the folder to mirror for `rich`     (…/RICH)
"""

import argparse
import json
import shutil
from pathlib import Path

from dc_available.common import sharepoint_connector as sp
from dc_available.config import data_config

# --- locations --------------------------------------------------------------
# The `available` DC_DATA mirror. Kept as a module constant because seed() (called
# by ingest.py's available upload flow) targets it directly.
MIRROR = Path(__file__).resolve().parent.parent / "dc_available" / "data" / "local_sharepoint"
MANIFEST = MIRROR / ".manifest.json"


def _source_paths(source: str):
    """Resolve (mirror_dir, manifest_path, site_id, remote_folder) for a data source.

    The mirror dir comes from data_config (so it matches what preprocess reads), and
    the SharePoint site/folder come from the env-var names declared in the source's
    config entry (loaded from pipeline/.env by sharepoint_connector).
    """
    data_config.update_data_config(source)
    cfg = data_config._data_config[source]
    mirror = Path(data_config.DEFAULT_DATA_PATH)
    manifest = mirror / ".manifest.json"
    site_id = sp.cfg(cfg.get("site_env", "AVAILABLE_SITE_ID"))
    remote_folder = sp.cfg(cfg.get("remote_folder_env", "AVAILABLE_FOLDER_PATH"))
    # RICH lives in a sibling folder of DC_DATA in the SAME site. If its own env var
    # isn't set, derive it from AVAILABLE_FOLDER_PATH by swapping the last path segment
    # to `RICH` (…/DC_DATA -> …/RICH), so no extra configuration is needed.
    if not remote_folder and source == "rich":
        avail = sp.cfg("AVAILABLE_FOLDER_PATH").strip("/")
        remote_folder = (avail.rsplit("/", 1)[0] + "/RICH") if "/" in avail else "RICH"
    return mirror, manifest, site_id, remote_folder


def _load_manifest(manifest: Path = MANIFEST) -> dict:
    if Path(manifest).is_file():
        return json.loads(Path(manifest).read_text(encoding="utf-8"))
    return {}


def _save_manifest(m: dict, manifest: Path = MANIFEST) -> None:
    Path(manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest).write_text(json.dumps(m, indent=2), encoding="utf-8")


def seed(entries: dict) -> int:
    """Record already-uploaded files so run() won't re-download them.

    entries maps  <path relative to the mirrored DC_DATA folder>  ->  (eTag, local_file).
    Each file is copied straight into the mirror and its eTag stored in the manifest,
    so the next run() sees a matching eTag + present file and skips the download.
    Called by ingest.py right after it uploads to the AVAILABLE SharePoint (the
    `available` source only).
    """
    if not entries:
        return 0
    manifest = _load_manifest(MANIFEST)
    for rel, (etag, local_file) in entries.items():
        dest = MIRROR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_file, dest)
        manifest[rel] = etag
    _save_manifest(manifest, MANIFEST)
    return len(entries)


# --- quiet logging: only print what actually happened -----------------------
def _ok(rel, tag):
    print(f"  ✓ {rel}   {tag}")


def _del(rel):
    print(f"  ⌫ {rel}   deleted")


def _fail(rel, why):
    print(f"  ✗ {rel}   failed ({why})")


def run(source: str = "available"):
    mirror, manifest_path, sid, folder = _source_paths(source)
    did = sp.drive_id(sid)

    print("=" * 52)
    print(f"SYNC · AVAILABLE → local mirror   (source: {source})")
    print("=" * 52)
    print(f"  from : {Path(folder).name}  (AVAILABLE SharePoint)")
    print(f"  to   : {mirror}")
    print("-" * 52)

    manifest = _load_manifest(manifest_path)
    before_total = len(manifest)
    seen = set()
    new = skipped = deleted = failed = 0
    remote_updates = 0
    fails = []

    # new -> download ; existing -> skip (even if eTag changed)
    for item, rel in sp.list_files(did, folder):
        seen.add(rel)
        etag = item.get("eTag", "")

        # Existing path in manifest means "already known" - do not re-download
        # even if local files were cleaned after preprocess.
        if rel in manifest:
            if manifest.get(rel) != etag:
                remote_updates += 1
                manifest[rel] = etag
            skipped += 1
            continue

        dest = mirror / rel
        try:
            sp.download(did, item["id"], dest)
            manifest[rel] = etag
            _ok(rel, "new")
            new += 1
        except Exception as exc:
            _fail(rel, str(exc)); failed += 1; fails.append((rel, str(exc)))

    # gone from the cloud -> remove locally
    for rel in [r for r in manifest if r not in seen]:
        dest = mirror / rel
        try:
            if dest.exists():
                dest.unlink()
            del manifest[rel]
            _del(rel); deleted += 1
        except Exception as exc:
            _fail(rel, str(exc)); failed += 1; fails.append((rel, str(exc)))

    _save_manifest(manifest, manifest_path)
    after_total = len(manifest)
    has_changes = (new + deleted) > 0

    print("-" * 52)
    print("  SYNC REPORT")
    print(f"    Before (total files) : {before_total}")
    print(f"    Unchanged            : {skipped}")
    print(f"    Changed              : {new + deleted}")
    print(f"       new               : {new}")
    print(f"       updated(remote)   : {remote_updates}   (metadata only, no download)")
    print(f"       deleted           : {deleted}")
    print(f"    Failed               : {failed}")
    print(f"    After (total files)  : {after_total}")
    for rel, why in fails:
        print(f"       ✗ {rel} — {why}")
    print("=" * 52)
    return has_changes


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Step 2 — mirror an AVAILABLE SharePoint folder locally")
    ap.add_argument(
        "--source",
        choices=["available", "rich"],
        default="available",
        help="data source to mirror (default: available)",
    )
    args = ap.parse_args()
    run(args.source)
