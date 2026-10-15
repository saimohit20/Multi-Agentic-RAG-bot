import argparse
import os
import subprocess
import sys
from pathlib import Path

from dc_available.config import data_config

def _check_index_ready(cache_path: Path) -> bool:
    # The index now has two parts that must BOTH exist (see get_retriever):
    #   1. the docstore pickle on disk (parent documents)
    #   2. the Qdrant collection (child-chunk vectors)
    pickle_file = cache_path / "index_ret.pkl"
    if not pickle_file.is_file():
        print(f"[Serve] ERROR: docstore not found: {pickle_file}")
        print("[Serve] Run  poetry run python -m pipeline.index  first.")
        return False

    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=data_config.QDRANT_URL)
        if not client.collection_exists(data_config.QDRANT_COLLECTION):
            print(
                f"[Serve] ERROR: Qdrant collection '{data_config.QDRANT_COLLECTION}' "
                f"not found at {data_config.QDRANT_URL}."
            )
            print("[Serve] Run  poetry run python -m pipeline.index --vectordb  first.")
            return False
    except Exception as exc:
        print(f"[Serve] ERROR: cannot reach Qdrant at {data_config.QDRANT_URL}: {exc}")
        print("[Serve] Start Qdrant (e.g. docker start qdrant) and retry.")
        return False
    return True


def run_serve(mode: str = "assistant", source: str | None = None) -> None:

    # Ensure the vector index (Qdrant + docstore) is ready before starting the app.
    if not _check_index_ready(data_config.DEFAULT_INDEX_ARTIFACTS_PATH):
        sys.exit(1)

    print(
        f"[Serve] Vector index ready: collection '{data_config.QDRANT_COLLECTION}' "
        f"+ docstore at {data_config.DEFAULT_INDEX_ARTIFACTS_PATH}"
    )

    page_map = {
        "assistant": Path("dc_available") / "frontend" / "Assistant.py",
        "qa": Path("dc_available") / "frontend" / "pages" / "2_QA.py",
    }

    if mode not in page_map:
        print(f"[Serve] ERROR: Unknown mode '{mode}'. Choose 'assistant' or 'qa'.")
        sys.exit(1)

    target = page_map[mode]
    print(f"[Serve] Starting Streamlit → {target}")
    print("[Serve] ══════════════════════════════════════\n")

    # Streamlit runs as a fresh process — pass the source through the env var so the
    # app selects the SAME data source we just validated (not the default).
    env = dict(os.environ)
    if source:
        env[data_config.APP_ID_ENV_VAR] = source

    cmd = [sys.executable, "-m", "streamlit", "run", str(target)]
    try:
        subprocess.run(cmd, check=True, env=env)
    except KeyboardInterrupt:
        print("\n[Serve] Stopped by user.")
    except subprocess.CalledProcessError as exc:
        print(f"[Serve] Streamlit exited with code {exc.returncode}.")
        sys.exit(exc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AVAILABLE — Serve stage: start the Streamlit app."
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
        "--mode",
        type=str,
        default="assistant",
        choices=["assistant", "qa"],
        help=(
            "'assistant' starts the full AI learning assistant (default). "
            "'qa' starts the Q&A-only page."
        ),
    )
    args = parser.parse_args()

    if args.source:
        data_config.update_data_config(args.source)
    else:
        data_config.update_data_config()

    print("[Serve] ══════════════════════════════════════")
    print(f"[Serve] Ingestion artifacts : {data_config.DEFAULT_INGESTION_ARTIFACTS_PATH}")
    print(f"[Serve] Index artifacts     : {data_config.DEFAULT_INDEX_ARTIFACTS_PATH}")
    print(f"[Serve] Mode         : {args.mode}")
    print("[Serve] ══════════════════════════════════════\n")

    run_serve(mode=args.mode, source=args.source)


if __name__ == "__main__":
    main()
