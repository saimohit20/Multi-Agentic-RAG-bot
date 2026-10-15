from pathlib import Path

from dc_available.RAG.retrieval_helpers import prepare_image_cache
from dc_available.common.artifact_paths import (
    ARTIFACT_KIND_INGESTION,
    mirror_training_path_to_artifacts,
)
from dc_available.common.helpers import delete_directory
from dc_available.config.data_config import (
    DEFAULT_DATA_PATH,
    DEFAULT_HANDBOOK_PATH, update_data_config,
)
from dc_available.summarization.markdown_utils import (
    INDEX_TEMPLATE,
    MK_YAML_TEMPLATE,
)
from dc_available.summarization.trainings import (
    summarize_training_for_retrieval,
)


def read_text_file_safe(filename: Path, default_encoding='utf-8') -> str:
    """
    Try to read a text file with multiple encodings
    """
    encodings = [
        default_encoding,
        'utf-8-sig',  # UTF-8 with BOM
        'latin1',  # Common for German characters
        'cp1252',  # Windows Western Europe
        'iso-8859-1'  # Another common encoding for special characters
    ]

    for encoding in encodings:
        try:
            return filename.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    # If all encodings fail, try binary read and decode with error handling
    try:
        return filename.read_bytes().decode(default_encoding, errors='replace')
    except Exception as e:
        print(f"Warning: Could not read file {filename}: {str(e)}")
        return f"[Error reading file: {filename.name}]"


def create_handbook(
        list_of_trainings: list[str | Path],
        output_directory: str | Path,
        reset: bool = True,
        overwrite: bool = False,
        encoding: str = 'utf-8'
):
    list_of_trainings = [Path(training) for training in list_of_trainings]
    output_directory = Path(output_directory)
    if reset:
        delete_directory(output_directory)

    print("Will read the documents from:", list_of_trainings)
    print("Will create the handbook in:", output_directory)

    if not output_directory.exists():
        output_directory.mkdir(parents=True)
        print(f"Created {output_directory}")

    filename_mkdocs_yml = output_directory / "mkdocs.yml"
    if not filename_mkdocs_yml.exists() or overwrite:
        with open(filename_mkdocs_yml, "w", encoding=encoding) as fout:
            fout.write(MK_YAML_TEMPLATE)
            print(f"Created {filename_mkdocs_yml}")

    output_documents_directory = output_directory / "docs"
    if not output_documents_directory.exists():
        output_documents_directory.mkdir()
        print(f"Created {output_documents_directory}")

    filename_handbook_index = output_documents_directory / "index.md"
    if not filename_handbook_index.exists() or overwrite:
        with open(filename_handbook_index, "w", encoding=encoding) as fout:
            fout.write(INDEX_TEMPLATE)
            print(f"Created {filename_handbook_index}")

    for training in sorted(list_of_trainings):
        training = Path(training)
        print("Check training:", training)
        assert training.is_dir()

        overview_summary = None
        other_summaries = []

        for filename in sorted(training.iterdir()):
            if filename.name == "summary.txt":
                overview_summary = read_text_file_safe(filename)
                overview_summary = overview_summary.replace("#", "# Overview\n\n##", 1)
                if "## Meta-Data" in overview_summary:
                    overview_summary = overview_summary.split("## Meta-Data")[0]
            elif filename.name.endswith("__summary.txt"):
                summary = read_text_file_safe(filename)
                filename_to_display = filename.name.replace("__summary.txt", "")
                summary = summary.replace("#", f"# {filename_to_display}\n\n##", 1)
                if "## Meta-Data" in summary:
                    summary = summary.split("## Meta-Data")[0]
                other_summaries.append((filename.stem + ".md", summary))
            else:
                continue

        if overview_summary is None or len(other_summaries) == 0:
            print(
                f"Skip training {training}: missing generated summaries "
                f"(requires summary.txt and at least one *__summary.txt)."
            )
            continue

        name = f"{training.parent.parent.stem}_{training.parent.stem}_{training.stem}"
        output_training_directory = output_documents_directory / name
        if not output_training_directory.exists():
            output_training_directory.mkdir()
            print(f"Created {output_training_directory}")

        # Write files with error handling
        try:
            with open(
                    output_training_directory / "index.md", "w", encoding=encoding
            ) as fout:
                fout.write(overview_summary)
                print(f"Created {output_training_directory / 'index.md'}")

            for fname, summary in other_summaries:
                with open(
                        output_training_directory / fname, "w", encoding=encoding
                ) as fout:
                    fout.write(summary)
                    print(f"Created {output_training_directory / fname}")
        except Exception as e:
            print(f"Error writing files for {name}: {str(e)}")
            continue

    print("\nHandbook was successfully created.")
    print("Now open a terminal and type the following commands:")
    print(f"\ncd {output_directory.resolve()}")
    print("poetry run mkdocs build")
    print("poetry run mkdocs serve")
    print(
        "\nThen, you can access the handbook files via the browser at the given url (default is http://127.0.0.1:8000/).")

if __name__ == "__main__":
    trainings = []
    topics = list(sorted(DEFAULT_DATA_PATH.iterdir()))
    for topic in DEFAULT_DATA_PATH.iterdir():
        if topic.is_dir():
            for knowledge_level in topic.iterdir():
                if knowledge_level.is_dir():
                    for training in knowledge_level.iterdir():
                        if training.is_dir() and not training.name.startswith("."):
                            trainings.append(training)
    trainings = list(sorted(trainings))

    for t in trainings:
        summarize_training_for_retrieval(t)
    update_data_config()
    prepare_image_cache()
    artifact_trainings = [
        mirror_training_path_to_artifacts(t, artifact_kind=ARTIFACT_KIND_INGESTION)
        for t in trainings
    ]
    create_handbook(artifact_trainings, DEFAULT_HANDBOOK_PATH)
