import shutil
import tempfile
import uuid
from pathlib import Path

from langchain_core.documents import Document

from dc_available.config import data_config
from dc_available.common.data_conversion import video_to_audio
from dc_available.common.data_loading import (
    load_docs_from_jsonl,
    save_docs_to_jsonl,
    load_msg_file,
)
from dc_available.common.artifact_paths import (
    ARTIFACT_KIND_INGESTION,
    mirror_file_path_to_artifacts,
    mirror_training_path_to_artifacts,
)
from dc_available.summarization.markdown_utils import (
    doc_to_markdown,
)
from dc_available.RAG.data import AvailableMetaData
from dc_available.summarization import prompts
from dc_available.summarization.slides import (
    generate_powerpoint_summary_for_retrieval,
    generate_pdf_or_pptx_summary_for_retrieval,
)
from dc_available.summarization.texts import (
    generate_meta_summary,
    generate_msg_summary_for_retrieval,
    generate_html_summary_for_retrieval,
    generate_meta_summary_as_document,
)
from dc_available.summarization.trust_score import (
    get_trust_score,
    get_meta_trust_score,
)
from dc_available.summarization.videos import (
    generate_video_summaries_for_retrieval,
)

SKIP_SUFFIX = [
    "__mp4.mp3",
    "__pptx.pdf",
    "__mp3_transcript.txt",
    "__summary.txt",
    "__summary_docs.jsonl",
    "index.jsonl",
    "summary.txt",
]
SKIP_PREFIX = ["~$"]
SKIP_FOLDERS = ["__pdf_images", "__pptx_images", "__mp4_slides", "__mp4_understanding"]
ACCEPTED_FORMATS = [".mp4", ".pdf", ".pptx", ".pptm", ".msg", ".xlsx", ".docx", ".html"]


DEFAULT_MAX_PAGES = 30
DEFAULT_SAMPLING_RATE = 60
DEFAULT_MODE_FAST = "fast"
DEFAULT_MODE_PRECISE = "precise"


def _work_copy(document: Path, dest_root: Path) -> Path:
    """Copy a source file into dest_root, mirroring its relative path, and return the
    copy. Extraction derives ALL its outputs (pdf, __*_images, __mp4_slides, audio,
    ...) from the file's own path, so processing the copy keeps every intermediate
    next to it — never inside the raw data mirror. dest_root is a private scratch dir
    (see summarize_single_file), so the raw file never lands in Extracted_resources.
    """
    document = Path(document)
    try:
        rel = document.resolve().relative_to(data_config.DEFAULT_DATA_PATH.resolve())
    except ValueError:
        rel = Path(document.name)
    work = dest_root / rel
    work.parent.mkdir(parents=True, exist_ok=True)
    if not work.exists() or work.stat().st_size != document.stat().st_size:
        shutil.copy2(document, work)
    return work


def _promote_derived_to_extracted(
    scratch_root: Path,
    extracted_root: Path,
    work_file: Path,
    output_documents: list[Document],
) -> None:
    """Move the DERIVED artifacts produced in the scratch dir into Extracted_resources,
    then rewrite the document metadata paths to point at their new home.

    Extraction runs on a copy of the raw file inside a private scratch dir, so every
    derived output (__pdf_images / __pptx_images / __pptx.pdf / __mp4_slides / ...)
    is created there. Here we move ONLY those derived outputs into Extracted_resources;
    the raw source file — and, for video, the extracted __mp4.mp3 — is deliberately
    left behind and discarded when the scratch dir is deleted, so the raw file never
    lands in Extracted_resources at all. The converted __<suffix>.pdf (e.g. __pptx.pdf)
    IS a derived artifact and IS kept.
    """
    work_file = Path(work_file)
    stem = work_file.stem
    src_dir = work_file.parent
    rel_dir = src_dir.resolve().relative_to(scratch_root.resolve())
    dest_dir = extracted_root / rel_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Never move the raw source; for video also skip the extracted audio.
    drop = {work_file.name}
    if work_file.suffix.lower() == ".mp4":
        drop.add(f"{stem}__mp4.mp3")

    for entry in src_dir.iterdir():
        if entry.name in drop or not entry.name.startswith(f"{stem}__"):
            continue  # raw file, dropped intermediate, or another file's output
        target = dest_dir / entry.name
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.move(str(entry), str(target))

    # The extractors baked scratch paths into the metadata (IMAGE_PATH, VIDEO_PATH,
    # STORAGE_PATH, ...). Rewrite them to the Extracted_resources location we just
    # moved the files to; scratch and Extracted_resources share the same relative
    # tail, so a prefix swap is exact.
    old_prefix = scratch_root.resolve().as_posix()
    new_prefix = extracted_root.resolve().as_posix()
    for doc in output_documents:
        for key, value in list(doc.metadata.items()):
            if isinstance(value, str) and old_prefix in value:
                doc.metadata[key] = value.replace(old_prefix, new_prefix)


def summarize_single_file(
        document: Path,
        summary_file: Path = None,
        summary_docs_file: Path = None,
    artifact_summary_file: Path = None,
    artifact_summary_docs_file: Path = None,
        max_pages: int = DEFAULT_MAX_PAGES,
        slide_sampling_rate: int = DEFAULT_SAMPLING_RATE,
        mode_pdf: str = DEFAULT_MODE_PRECISE,
        mode_pptx: str = DEFAULT_MODE_PRECISE,
) -> tuple[list[Document], list[Path]]:
    new_file_paths = []
    output_documents = None
    # Raw media is processed in a private scratch dir so the raw file NEVER lands in
    # Extracted_resources. After extraction we move ONLY the derived artifacts out.
    scratch_root: Path | None = None
    work_copy: Path | None = None       # the raw file's copy inside the scratch dir
    extracted_root: Path | None = None  # final Extracted_resources root for this format

    try:
        if document.suffix == ".pdf":
            # Standard PDF (could be converted PowerPoint). Process a WORK COPY in a
            # scratch dir so the derived __pdf_images are produced there; only those
            # images are later moved to Extracted_resources, never the raw .pdf.
            scratch_root = Path(tempfile.mkdtemp(prefix="dcav_extract_"))
            extracted_root = data_config.DEFAULT_PDF_PATH
            work_doc = _work_copy(document, scratch_root)
            work_copy = work_doc
            try:
                pdf_texts, pdf_text_summaries = generate_pdf_or_pptx_summary_for_retrieval(
                    work_doc, max_pages=max_pages, mode=mode_pdf
                )
            except Exception as e:
                if "PyCryptodome is required for AES algorithm" in str(e):
                    print("PDF is password protected. Skip it.")
                    return [], []
                else:
                    raise e
            if len(pdf_text_summaries) > len(pdf_texts):
                # All pdf pages were processed and an overall meta-summary was created.
                meta_summary = pdf_text_summaries[0]
            else:
                # (full) Meta summary was not created earlier. Create it now with all information available.
                # We have generated fewer summaries than pages, so we need to combine them.
                meta_summary = generate_meta_summary_as_document(
                    pdf_text_summaries, pdf_texts
                )
                pdf_text_summaries = [meta_summary] + pdf_text_summaries
            output_documents = pdf_texts + pdf_text_summaries

        elif document.suffix in [".pptx", ".pptm", ".xlsx", ".docx"]:
            # Quick fix to convert Excel files to PDF using the powerpoint method
            # TODO: This is not a good solution as larger sheets are put on multiple pdf pages
            if document.suffix in [".xlsx", ".docx"]:
                max_pages = min(10, max_pages)  # want to save some cost

            # PowerPoint / Office — process a WORK COPY in a scratch dir so the derived
            # <name>__<suffix>.pdf and __<suffix>_images are produced there. The raw
            # source is discarded; only the converted pdf + images are promoted.
            scratch_root = Path(tempfile.mkdtemp(prefix="dcav_extract_"))
            extracted_root = data_config.DEFAULT_PPTX_PATH
            work_doc = _work_copy(document, scratch_root)
            work_copy = work_doc
            ppt_texts, ppt_slide_summaries = generate_powerpoint_summary_for_retrieval(
                work_doc, max_pages=max_pages, mode=mode_pptx
            )
            if len(ppt_slide_summaries) > len(ppt_texts):
                # All pptx slides were processed and an overall meta-summary was created.
                meta_summary = ppt_slide_summaries[0]
            else:
                # (full) Meta summary was not created earlier. Create it now with all information available.
                # We have generated fewer summaries than pages, so we need to combine them.
                meta_summary = generate_meta_summary_as_document(
                    ppt_slide_summaries, ppt_texts
                )
                ppt_slide_summaries = [meta_summary] + ppt_slide_summaries
            output_documents = ppt_slide_summaries + ppt_texts

        elif document.suffix == ".mp4":
            # Video — process a WORK COPY in a scratch dir so every extraction
            # intermediate (__mp4.mp3, __mp4_slides, __mp4_understanding) is produced
            # there. Only the slide images are moved to Extracted_resources; the raw
            # .mp4 and the __mp4.mp3 audio are discarded with the scratch dir.
            scratch_root = Path(tempfile.mkdtemp(prefix="dcav_extract_"))
            extracted_root = data_config.DEFAULT_VIDEO_PATH
            work_video = _work_copy(document, scratch_root)
            work_copy = work_video
            audio_file = video_to_audio(
                work_video, work_video.as_posix().replace(".mp4", "__mp4.mp3")
            )

            if not audio_file.exists():
                print(f"Failed to extract audio from video {document}")
                return [], []

            try:
                print(f"Processing video: {document}")
                (
                    meta_summary,
                    video_summaries,
                    image_summaries,
                    audio_summaries,
                ) = generate_video_summaries_for_retrieval(
                    work_video, audio_file, slide_sampling_rate=slide_sampling_rate
                )

                # Validate that we have some content
                total_content = len(video_summaries) + len(image_summaries) + len(audio_summaries)
                if total_content == 0:
                    print(f"No content extracted from video {document} - all summary lists are empty")
                    return [], []

                if meta_summary is None:
                    print(f"No meta summary generated for video {document}")
                    return [], []

                output_documents = (
                        [meta_summary] + video_summaries + image_summaries + audio_summaries
                )

            except (ValueError, AssertionError) as e:
                print(f"Failed to process video {document}: {e}")
                print("Skipping this video file.")
                return [], []
            except Exception as e:
                print(f"Unexpected error processing video {document}: {e}")
                print("Skipping this video file.")
                return [], []

        elif document.suffix == ".msg":
            # Email (maybe with attachments)
            msg_text, meta_summary, attachment_paths = generate_msg_summary_for_retrieval(
                document
            )
            new_file_paths.extend(attachment_paths)
            output_documents = [msg_text, meta_summary]

        elif document.suffix == ".html":
            # Confluence/Docupedia page exported as HTML -> treated as text
            html_text, meta_summary = generate_html_summary_for_retrieval(document)
            output_documents = [html_text, meta_summary]

        else:
            raise ValueError("Unknown document format")

        # Validate output_documents
        if not output_documents:
            print(f"No documents generated from {document}")
            return [], []

        # Extraction succeeded. Move ONLY the derived artifacts from the scratch dir
        # into Extracted_resources and rewrite the metadata paths to their new home.
        # The raw file (and, for video, the __mp4.mp3) stays behind and is discarded.
        if scratch_root is not None and work_copy is not None:
            _promote_derived_to_extracted(
                scratch_root, extracted_root, work_copy, output_documents
            )
    finally:
        if scratch_root is not None:
            shutil.rmtree(scratch_root, ignore_errors=True)

    # The meta summary (summary of the file) should be the first document in the list
    if 'meta_summary' in locals() and meta_summary in output_documents:
        if meta_summary != output_documents[0]:
            output_documents.remove(meta_summary)
            output_documents.insert(0, meta_summary)

    for doc in output_documents:
        doc.metadata["source"] = document.as_posix()
        doc.metadata[AvailableMetaData.PART_OF_DOCUMENT] = document.stem
        doc.metadata[AvailableMetaData.PART_OF_TRAINING] = "-1"

    if summary_file is None:
        summary_file = Path(document.as_posix() + "__summary.txt")
    if artifact_summary_file is None:
        artifact_summary_file = (
            mirror_file_path_to_artifacts(document, artifact_kind=ARTIFACT_KIND_INGESTION)
            .with_name(Path(summary_file).name)
        )
    print(f"Save summary file: {artifact_summary_file}")
    artifact_summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(artifact_summary_file, "w", encoding="utf-8") as fout:
        fout.write(doc_to_markdown(output_documents[0], method="general"))

    if summary_docs_file is None:
        summary_docs_file = Path(document.as_posix() + "__summary_docs.jsonl")
    if artifact_summary_docs_file is None:
        artifact_summary_docs_file = (
            mirror_file_path_to_artifacts(document, artifact_kind=ARTIFACT_KIND_INGESTION)
            .with_name(Path(summary_docs_file).name)
        )
    print(f"Save summary docs file: {artifact_summary_docs_file}")

    # Compute trust score
    trust_score, _ = get_trust_score(output_documents[0], output_documents)
    output_documents[0].metadata[AvailableMetaData.TRUST_SCORE] = trust_score
    for doc in output_documents:
        doc.metadata[AvailableMetaData.TRUST_SCORE] = trust_score
    print(f"Trust score for file: {trust_score}")

    artifact_summary_docs_file.parent.mkdir(parents=True, exist_ok=True)
    save_docs_to_jsonl(output_documents, artifact_summary_docs_file)

    return output_documents, new_file_paths


def check_for_new_files(document: Path) -> list[Path]:
    new_files = []
    if document.suffix == ".msg":
        _, new_files = load_msg_file(document)
    return new_files


def summarize_training_for_retrieval(
        document_folder: str | Path,
        output_file: str | Path = None,
    artifact_output_file: str | Path = None,
        level: str = None,
        overwrite: bool = False,
) -> list[Document]:
    document_folder = Path(document_folder)

    if level is None:
        # Case-insensitive so lowercase folder names (e.g. RICH's `expert/`) resolve.
        folder_lower = document_folder.as_posix().lower()
        if "advanced" in folder_lower:
            level = "Advanced"
        elif "expert" in folder_lower:
            level = "Expert"
        elif "basic" in folder_lower:
            level = "Basic"
        else:
            level = "Unknown"

    index: list[Document] = []
    file_summaries: list[Document] = []
    remaining_files = sorted(document_folder.iterdir())

    artifact_document_folder = mirror_training_path_to_artifacts(
        document_folder,
        artifact_kind=ARTIFACT_KIND_INGESTION,
    )

    while remaining_files:
        document = remaining_files.pop(0)

        is_intermediate_file = False
        for suffix in SKIP_SUFFIX:
            if document.name.endswith(suffix):
                is_intermediate_file = True
                break
        for prefix in SKIP_PREFIX:
            if document.name.startswith(prefix):
                is_intermediate_file = True
                break
        if is_intermediate_file:
            print(f"Skip intermediate file {document}.")
            continue

        if document.suffix in ACCEPTED_FORMATS:
            summary_file = Path(document.as_posix() + "__summary.txt")
            summary_docs_file = Path(document.as_posix() + "__summary_docs.jsonl")
            artifact_summary_file = mirror_file_path_to_artifacts(
                document, artifact_kind=ARTIFACT_KIND_INGESTION
            ).with_name(summary_file.name)
            artifact_summary_docs_file = mirror_file_path_to_artifacts(
                document, artifact_kind=ARTIFACT_KIND_INGESTION
            ).with_name(summary_docs_file.name)

            if (
                artifact_summary_file.exists()
                and artifact_summary_docs_file.exists()
                and not overwrite
            ):
                print(
                    f"Load existing artifact summary for file {document}. Call with overwrite=True to overwrite."
                )
                documents = load_docs_from_jsonl(artifact_summary_docs_file)
                new_files = check_for_new_files(document)
                remaining_files.extend(new_files)

            else:
                print(f"Summarize file: {document}")
                documents, new_files = summarize_single_file(
                    document,
                    summary_file,
                    summary_docs_file,
                    artifact_summary_file,
                    artifact_summary_docs_file,
                )
                remaining_files.extend(new_files)

            if len(documents) == 0:
                print(f"No content extracted from {document}, skipping...")
                continue

            meta_summary = documents[0]
            index.extend(documents)
            file_summaries.append(meta_summary)
        else:
            print(f"Skip file {document} with unsupported file extension for now.")
            continue

    # Check if we have any valid summaries before proceeding
    if len(file_summaries) == 0:
        print(f"Warning: No valid summaries found in {document_folder}")
        print("This could be due to:")
        print("1. All video files failed to process")
        print("2. No supported file formats found")
        print("3. All files were skipped due to processing errors")
        return []

    if output_file is None:
        output_file = artifact_document_folder / "index.jsonl"
    if artifact_output_file is None:
        artifact_output_file = artifact_document_folder / "index.jsonl"
    summary_file = artifact_document_folder / "summary.txt"
    artifact_summary_file = artifact_document_folder / "summary.txt"

    print(f"Generate folder meta summary for {document_folder}")
    try:
        training_summary = generate_meta_summary(
            [doc.page_content for doc in file_summaries],
            prompt=prompts.META_TRAINING_SUMMARY_PROMPT,
        )
    except AssertionError as e:
        print(f"Failed to generate meta summary for {document_folder}: {e}")
        return []

    training_doc = Document(
        page_content=training_summary,
        metadata={
            AvailableMetaData.ID: str(uuid.uuid4()),
            AvailableMetaData.PAGE_ID: -1,
            AvailableMetaData.FILE_TYPE: "folder",
            AvailableMetaData.DOCUMENT_TYPE: "training",
            AvailableMetaData.STORAGE_PATH: document_folder.resolve().as_posix(),
            AvailableMetaData.IS_META_SUMMARY: True,
            AvailableMetaData.PART_OF_TRAINING: document_folder.stem,
            AvailableMetaData.LINK_TO_CHILDREN: [
                doc.metadata[AvailableMetaData.ID] for doc in file_summaries
            ],
        },
    )

    artifact_summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(artifact_summary_file, "w", encoding="utf-8") as fout:
        fout.write(doc_to_markdown(training_doc, method="training"))

    for docs in index:
        docs.metadata[AvailableMetaData.LEVEL] = level
        docs.metadata[AvailableMetaData.PART_OF_TRAINING] = training_doc.metadata[
            AvailableMetaData.ID
        ]

    # Compute trust score
    trust_score, explanation = get_meta_trust_score(training_doc, index)
    training_doc.metadata[AvailableMetaData.TRUST_SCORE] = trust_score
    for doc in index:
        if AvailableMetaData.TRUST_SCORE not in doc.metadata:
            doc.metadata[AvailableMetaData.TRUST_SCORE] = trust_score
    print(f"Trust score for training: {trust_score}")

    print(f"Store index to disc -> {output_file}")
    index.insert(0, training_doc)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    save_docs_to_jsonl(index, output_file)

    if Path(artifact_output_file) != Path(output_file):
        print(f"Store index to disc -> {artifact_output_file}")
        Path(artifact_output_file).parent.mkdir(parents=True, exist_ok=True)
        save_docs_to_jsonl(index, artifact_output_file)
    print(f"Done with {document_folder}")
    return index