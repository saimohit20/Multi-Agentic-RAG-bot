"""
pipeline/ingest.py — STEP 1 of the pipeline.

Bring training data from the grownfield SharePoint into the AVAILABLE SharePoint:

    fetch     grownfield SharePoint -> staging/    (.pptx/.mp4 direct,
                                                    .url -> HTML,
                                                    .lnk -> real file resolved by name)
    classify  staging/            -> organized/<Topic>/<Level>/training/   (MOVE)
    upload    organized/          -> AVAILABLE SharePoint

Data lives under  dc_available/data/grownfield/{staging,organized}.

Run:
    python -m pipeline.ingest

Config + Graph/.lnk/.url/classify helpers come from  dc_available/common/sharepoint_connector.py
(which loads pipeline/.env).
"""

import argparse
import contextlib
import io
import shutil
from pathlib import Path

from dc_available.common import sharepoint_connector as sp
from pipeline import sync

# --- data locations ---------------------------------------------------------
_GROWNFIELD = Path(__file__).resolve().parent.parent / "dc_available" / "data" / "grownfield"
STAGING = _GROWNFIELD / "staging"
ORGANIZED = _GROWNFIELD / "organized"


# ---------------------------------------------------------------------------
# clean logging — one line per file, one summary per phase
# ---------------------------------------------------------------------------
def _header(title):
    print(f"\n{title}")
    print("-" * 52)


def _ok(name):
    print(f"  ✓ {name}")


def _skip(name, why="already done"):
    print(f"  ⊘ {name}   skipped ({why})")


def _fail(name, why):
    print(f"  ✗ {name}   failed ({why})")


def _summary(counts, failed):
    print("  " + "-" * 50)
    total = counts["ok"] + counts["skip"] + counts["fail"]
    print(f"  Total:    {total}")
    print(f"  Success:  {counts['ok']}")
    print(f"  Skipped:  {counts['skip']}")
    print(f"  Failed:   {counts['fail']}")
    for name, why in failed:
        print(f"     ✗ {name} — {why}")


def _classified_dest(name):
    """organized/<Topic>/<Level>/training/<name>  (same rule preprocess uses to discover)."""
    return ORGANIZED / sp.topic_of(name) / sp.level_of(name) / sp.training_of(name) / name


# ---------------------------------------------------------------------------
# phase 1 — fetch grownfield -> staging  (skip if already in organized)
# ---------------------------------------------------------------------------
def fetch():
    _header("FETCH · grownfield → staging")
    did = sp.drive_id(sp.site_id("SOURCE"))
    counts = {"ok": 0, "skip": 0, "fail": 0}
    failed = []

    for item, rel in sp.list_files(did, sp.cfg("SOURCE_FOLDER_PATH")):
        name = Path(rel).name
        ext = Path(rel).suffix.lower()
        src_item = item

        # decide the OUTPUT file name per type
        if ext in (".pptx", ".mp4"):
            out = name
        elif ext == ".url":
            out = Path(name).with_suffix(".html").name
        elif ext == ".lnk":
            try:
                src_item, real = sp.lnk_to_item(did, sp.get_bytes(did, item["id"]))
            except ImportError:
                _fail(name, "pip install pylnk3"); counts["fail"] += 1
                failed.append((name, "pylnk3 missing")); continue
            if not src_item:
                _fail(name, f"target not found: {real or 'unparsable'}"); counts["fail"] += 1
                failed.append((name, "target not found")); continue
            out = real
        else:
            continue  # .srt etc. — ignored silently

        # move-not-copy: the final artifact lives in organized/, so skip on THAT
        if _classified_dest(out).exists():
            _skip(out); counts["skip"] += 1; continue
        dest = STAGING / out
        if dest.exists():
            _skip(out, "already staged"); counts["skip"] += 1; continue

        try:
            if ext == ".url":
                # url_to_html is quiet now, but keep output isolated just in case
                with contextlib.redirect_stdout(io.StringIO()):
                    sp.url_to_html(sp.get_bytes(did, item["id"]), dest, rel)
            else:
                sp.download(did, src_item["id"], dest)
            _ok(out); counts["ok"] += 1
        except Exception as exc:
            _fail(out, str(exc)); counts["fail"] += 1; failed.append((out, str(exc)))

    _summary(counts, failed)


# ---------------------------------------------------------------------------
# phase 2 — classify staging -> organized  (MOVE, not copy)
# ---------------------------------------------------------------------------
def classify():
    _header("CLASSIFY · staging → organized (move)")
    counts = {"ok": 0, "skip": 0, "fail": 0}
    failed = []

    for src in sorted(STAGING.rglob("*")):
        if not src.is_file():
            continue
        dest = _classified_dest(src.name)
        if dest.exists():
            _skip(src.name); counts["skip"] += 1; continue
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            _ok(f"{src.name}  →  {dest.parent.relative_to(ORGANIZED).as_posix()}")
            counts["ok"] += 1
        except Exception as exc:
            _fail(src.name, str(exc)); counts["fail"] += 1; failed.append((src.name, str(exc)))

    _summary(counts, failed)


# ---------------------------------------------------------------------------
# phase 3 — upload organized -> AVAILABLE SharePoint
# ---------------------------------------------------------------------------
def upload():
    _header("UPLOAD · organized → AVAILABLE")
    did = sp.drive_id(sp.site_id("DEST"))
    base = sp.cfg("DEST_FOLDER_PATH").strip("/")
    avail = sp.cfg("AVAILABLE_FOLDER_PATH").strip("/")   # what sync.py mirrors (…/DC_DATA)
    counts = {"ok": 0, "skip": 0, "fail": 0}
    failed = []
    seed_entries = {}   # mirror-relative path -> (eTag, local file)

    for src in sorted(ORGANIZED.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(ORGANIZED).as_posix()
        dest_path = f"{base}/{rel}" if base else rel
        try:
            etag = sp.upload(did, dest_path, src)
            # our upload target sits under the folder sync.py mirrors, so record it
            # (mirror copy + eTag) → sync.py won't re-download our own file.
            if etag and avail and dest_path.startswith(avail + "/"):
                seed_entries[dest_path[len(avail) + 1:]] = (etag, src)
            _ok(rel); counts["ok"] += 1
        except Exception as exc:
            _fail(rel, str(exc)); counts["fail"] += 1; failed.append((rel, str(exc)))

    _summary(counts, failed)
    seeded = sync.seed(seed_entries)
    if seeded:
        print(f"  seeded {seeded} file(s) into the local mirror — sync will skip re-downloading them")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Step 1 — grownfield → AVAILABLE SharePoint")
    ap.add_argument("--phase", choices=["fetch", "classify", "upload", "all"],
                    default="all", help="run one phase only (default: all)")
    args = ap.parse_args()

    print("=" * 52)
    print(f"INGEST · grownfield → AVAILABLE   (phase: {args.phase})")
    print("=" * 52)
    STAGING.mkdir(parents=True, exist_ok=True)
    ORGANIZED.mkdir(parents=True, exist_ok=True)

    if args.phase in ("fetch", "all"):
        fetch()
    if args.phase in ("classify", "all"):
        classify()
    if args.phase in ("upload", "all"):
        upload()

    print("\nDone.")


if __name__ == "__main__":
    main()
