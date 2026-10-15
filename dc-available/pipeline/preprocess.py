import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
from pathlib import Path


# Domain imports 
from dc_available.config import data_config
from dc_available.create_handbook import create_handbook
from dc_available.common.artifact_paths import (
    ARTIFACT_KIND_INGESTION,
    mirror_training_path_to_artifacts,
)
from dc_available.summarization.trainings import (
    ACCEPTED_FORMATS,
    summarize_training_for_retrieval,
)


def _has_premade_artifacts(path: Path) -> bool:
    """True if a raw folder already ships finished artifacts (index.jsonl + summary.txt)."""
    return (path / "index.jsonl").is_file() and (path / "summary.txt").is_file()


def _discover_trainings(data_path: Path, include_premade: bool = False) -> list[Path]:
    """Find every "training" folder anywhere under data_path.

    A training is any folder that directly contains at least one supported file.
    Its parent is treated as the level and its grandparent as the topic, so any
    wrapper folders above the topic (e.g. a dataset name like 'GROWNFIELD') are
    simply ignored. This lets 'topic/level/training/files' sit at any depth under
    the root, and different branches may have different depths.

    When include_premade is set (RICH), folders that ship finished artifacts
    (index.jsonl + summary.txt) but no raw source files are also discovered, so
    pre-processed interviews are picked up and copied without re-summarization.
    """
    trainings = []
    for path in sorted(data_path.rglob("*")):
        if not path.is_dir() or path.name.startswith("."):
            continue
        has_supported_file = any(
            child.is_file() and child.suffix.lower() in ACCEPTED_FORMATS
            for child in path.iterdir()
        )
        if has_supported_file or (include_premade and _has_premade_artifacts(path)):
            trainings.append(path)
    return trainings


def _copy_premade_artifacts(training_path: Path) -> bool:
    """Copy a raw folder's finished artifacts into the artifacts tree, mirroring its path.

    Returns False if index.jsonl + summary.txt are not both present (caller then
    falls back to normal summarization).
    """
    if not _has_premade_artifacts(training_path):
        return False
    dest_dir = mirror_training_path_to_artifacts(
        training_path, artifact_kind=ARTIFACT_KIND_INGESTION
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(training_path / "index.jsonl", dest_dir / "index.jsonl")
    shutil.copy2(training_path / "summary.txt", dest_dir / "summary.txt")
    return True


def _has_cached_training_outputs(training_path: Path) -> bool:

    artifact_training_path = mirror_training_path_to_artifacts(training_path)
    return (
        (artifact_training_path / "index.jsonl").is_file()
        and (artifact_training_path / "summary.txt").is_file()
    )


def _training_signature(training_path: Path) -> str:
    """Fingerprint of a training's source files (name + size + mtime).

    Changes whenever a file is added, removed, or modified — so preprocess can
    rebuild exactly the trainings that actually changed, not just the ones with
    missing artifacts. (A file re-downloaded by sync.py gets a new mtime.)
    """
    parts = []
    for f in sorted(training_path.iterdir()):
        if f.is_file() and f.suffix.lower() in ACCEPTED_FORMATS:
            st = f.stat()
            parts.append([f.name, st.st_size, int(st.st_mtime)])
    return hashlib.sha1(json.dumps(parts).encode("utf-8")).hexdigest()


def _signature_file(training_path: Path) -> Path:
    return mirror_training_path_to_artifacts(training_path) / ".source_sig"


def _rel(training_path: Path) -> str:
    """Short display name: the training folder relative to the data root."""
    try:
        return training_path.relative_to(data_config.DEFAULT_DATA_PATH).as_posix()
    except ValueError:
        return training_path.name


def _cleanup_mirror_downloads(data_path: Path) -> int:
    """Delete mirrored raw files after preprocessing, keep manifest."""
    if not data_path.exists():
        return 0

    deleted = 0
    for child in data_path.iterdir():
        if child.name == ".manifest.json":
            continue
        if child.is_dir():
            shutil.rmtree(child)
            deleted += 1
        elif child.is_file():
            child.unlink()
            deleted += 1
    return deleted


def run_ingestion(
    training_path: Path | None = None,
    overwrite: bool = False,
    verbose: bool = False,
    cleanup_downloads: bool = False,
    copy_existing: bool = False,
    build_handbook: bool = True,
) -> list[Path]:

    if training_path is not None:
        training_path = Path(training_path)
        if not training_path.is_dir():
            print(f"  ERROR: --training path does not exist: {training_path}")
            sys.exit(1)
        trainings = [training_path]
    else:
        trainings = _discover_trainings(
            data_config.DEFAULT_DATA_PATH, include_premade=copy_existing
        )

    if not trainings:
        print(f"  ERROR: no training folders found under {data_config.DEFAULT_DATA_PATH}")
        sys.exit(1)

    print(f"  found : {len(trainings)} trainings")
    print("-" * 52)

    # Quiet by default: hide the underlying summarization chatter unless --verbose.
    quiet = contextlib.redirect_stdout(io.StringIO()) if not verbose else contextlib.nullcontext()

    skipped, failed, built = [], [], 0
    for training in trainings:
        rel = _rel(training)

        # Simple mode: once artifacts exist, treat this training as already processed.
        # We intentionally do not detect source-file updates in this mode.
        if not overwrite and _has_cached_training_outputs(training):
            print(f"  ⊘ {rel}   already processed")
            skipped.append(training)
            continue

        # RICH: if the raw folder already ships finished artifacts, copy them into
        # the artifacts tree and skip the (expensive) summarization entirely.
        if copy_existing and _copy_premade_artifacts(training):
            print(f"  ⇪ {rel}   copied pre-made artifacts (no summarization)")
            built += 1
            continue

        print(f"  ⏳ {rel}   building…")
        try:
            with quiet:
                docs = summarize_training_for_retrieval(training, overwrite=overwrite)
            if docs:
                print(f"  ✓ {rel}   built ({len(docs)} docs)")
                built += 1
            else:
                print(f"  ✗ {rel}   failed (no documents produced)")
                failed.append((rel, "no documents produced"))
        except Exception as exc:
            print(f"  ✗ {rel}   failed ({exc})")
            failed.append((rel, str(exc)))

    # Handbook — only when something was (re)built (skipped entirely for RICH).
    if build_handbook and built > 0:
        all_trainings = [
            mirror_training_path_to_artifacts(t)
            for t in _discover_trainings(data_config.DEFAULT_DATA_PATH)
        ]
        print(f"  handbook → {data_config.DEFAULT_HANDBOOK_PATH}")
        with (contextlib.redirect_stdout(io.StringIO()) if not verbose else contextlib.nullcontext()):
            create_handbook(all_trainings, data_config.DEFAULT_HANDBOOK_PATH)

    print("-" * 52)
    print(f"  Total {len(trainings)} · Built {built} · Unchanged {len(skipped)} · Failed {len(failed)}")
    for rel, reason in failed:
        print(f"     ✗ {rel} — {reason}")

    # Optional cleanup mode for SharePoint mirror workflows.
    if cleanup_downloads and not failed:
        if training_path is None:
            removed = _cleanup_mirror_downloads(data_config.DEFAULT_DATA_PATH)
            print(f"  cleanup: removed {removed} mirrored item(s) from {data_config.DEFAULT_DATA_PATH}")
        else:
            print("  cleanup: skipped (single-training mode)")

    print("=" * 52)

    if failed:
        sys.exit(1)
    return trainings

                                                                                                                    
def main():
    parser = argparse.ArgumentParser(
        description="AVAILABLE — Preprocess stage: summarize documents and build handbook."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help=(
            "Data source key from _data_config in data_config.py "
            "(e.g. 'dc_available', 'test'). "
            "Defaults to the value of AVAILABLE_APP_SOURCE env var or DEFAULT_DATA_SOURCE ('available')."
        ),
    )
    parser.add_argument(
        "--training",
        type=str,
        default=None,
        help=(
            "Path to a single training folder to process. "
            "If omitted, all trainings under the data path are processed."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-summarize every training, including already processed ones.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show the underlying summarization logs (PDF/PPTX/video/LLM). Default: quiet.",
    )
    parser.add_argument(
        "--cleanup-downloads",
        action="store_true",
        default=False,
        help="After successful run, delete mirrored raw files from the data path (keeps .manifest.json).",
    )
    args = parser.parse_args()

    # Apply data source selection before anything else
    if args.source:
        data_config.update_data_config(args.source)
    else:
        data_config.update_data_config()  # reads AVAILABLE_APP_SOURCE env var or uses default

    # RICH: fold in finished artifacts shipped with the raw folder and skip the handbook.
    resolved_source = args.source or os.environ.get(
        data_config.APP_ID_ENV_VAR, data_config.DEFAULT_DATA_SOURCE
    )
    is_rich = resolved_source == "rich"

    print("=" * 52)
    print("PREPROCESS · local_sharepoint → artifacts")
    print("=" * 52)
    print(f"  data  : {data_config.DEFAULT_DATA_PATH}")
    print(f"  out   : {data_config.DEFAULT_INGESTION_ARTIFACTS_PATH}")
    if args.overwrite:
        print("  mode  : overwrite (rebuild all)")
    if is_rich:
        print("  source: rich (copy pre-made artifacts, no handbook)")
    if args.training:
        print(f"  target: {args.training}")

    run_ingestion(
        training_path=args.training,
        overwrite=args.overwrite,
        verbose=args.verbose,
        cleanup_downloads=args.cleanup_downloads,
        copy_existing=is_rich,
        build_handbook=not is_rich,
    )


if __name__ == "__main__":
    main()
