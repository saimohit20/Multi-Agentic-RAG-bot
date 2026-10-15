from __future__ import annotations

from pathlib import Path

from dc_available import DATA_DIR
from dc_available.config import data_config


ARTIFACT_KIND_INGESTION = "ingestion"
ARTIFACT_KIND_INDEX = "index"
ARTIFACT_KIND_HANDBOOK = "handbook"


def get_artifacts_root() -> Path:
    """Return the root folder for all generated artifacts for the active source."""
    return data_config.DEFAULT_ARTIFACTS_ROOT


def get_ingestion_artifacts_root() -> Path:
    """Return the ingestion artifact root for the active source."""
    return data_config.DEFAULT_INGESTION_ARTIFACTS_PATH


def get_index_artifacts_root() -> Path:
    """Return the index artifact root for the active source."""
    return data_config.DEFAULT_INDEX_ARTIFACTS_PATH


def get_handbook_artifacts_root() -> Path:
    """Return the handbook artifact root for the active source."""
    return data_config.DEFAULT_HANDBOOK_ARTIFACTS_PATH


def mirror_training_path_to_artifacts(
    training_path: str | Path,
    artifact_kind: str = ARTIFACT_KIND_INGESTION,
) -> Path:
    """Mirror a raw training folder into the configured artifacts tree.

    Example:
        raw:      resources/DC_data/DigitalTransformation/Basic/Training1
        result:   resources/artifacts/DC_data/ingestion/DigitalTransformation/Basic/Training1
    """
    training_path = Path(training_path)
    if not training_path.is_absolute():
        training_path = (Path.cwd() / training_path).resolve()

    try:
        relative_training_path = training_path.relative_to(data_config.DEFAULT_DATA_PATH)
    except ValueError:
        # Fallback for callers that pass a path outside the configured data root.
        # In that case, keep only the folder name to avoid leaking the full source path.
        relative_training_path = Path(training_path.name)

    if artifact_kind == ARTIFACT_KIND_INGESTION:
        base_root = get_ingestion_artifacts_root()
    elif artifact_kind == ARTIFACT_KIND_INDEX:
        base_root = get_index_artifacts_root()
    elif artifact_kind == ARTIFACT_KIND_HANDBOOK:
        base_root = get_handbook_artifacts_root()
    else:
        raise ValueError(f"Unknown artifact kind: {artifact_kind}")

    return base_root / relative_training_path


def mirror_file_path_to_artifacts(
    file_path: str | Path,
    artifact_kind: str = ARTIFACT_KIND_INGESTION,
) -> Path:
    """Mirror a generated file path from the raw data tree into artifacts.

    Example:
        raw:      resources/DC_data/.../Training1/file.pdf__summary.txt
        result:   resources/artifacts/DC_data/ingestion/.../Training1/file.pdf__summary.txt
    """
    file_path = Path(file_path)
    if not file_path.is_absolute():
        file_path = (Path.cwd() / file_path).resolve()

    if file_path.is_dir():
        return mirror_training_path_to_artifacts(file_path, artifact_kind=artifact_kind)

    try:
        relative_file_path = file_path.relative_to(data_config.DEFAULT_DATA_PATH)
    except ValueError:
        relative_file_path = Path(file_path.name)

    return mirror_training_path_to_artifacts(
        data_config.DEFAULT_DATA_PATH / relative_file_path.parent,
        artifact_kind=artifact_kind,
    ) / relative_file_path.name


def get_source_artifacts_root(source_key: str | None = None) -> Path:
    """Return the artifact root for a source key without mutating global config.

    This is useful for planning or migration tasks.
    """
    if source_key is None:
        return get_artifacts_root()

    source_config = data_config._data_config[source_key]
    return DATA_DIR / "artifacts" / source_config["main_folder"]
