import copy
import os
import re
import uuid
from pathlib import Path

from langchain_core.documents import Document

from dc_available.common.data_conversion import video_to_audio
from dc_available.common.VideoIE.VideoIE import (
    training_video_information_extraction,
)
from dc_available.RAG.data import AvailableMetaData
from dc_available.summarization.texts import (
    generate_meta_summary,
)
from dc_available.summarization.utils import (
    connect_parent_and_children_documents,
    connect_related_documents,
)


def _load_video_parts(slides_dir: Path, understanding_dir: Path) -> list[dict]:
    assert slides_dir.exists() and understanding_dir.exists()

    video_parts = []
    for slide_file in understanding_dir.iterdir():
        if not slide_file.suffix == ".txt":
            continue

        with open(slide_file, "r", encoding="utf-8") as fin:
            content = fin.read()

        slide = {}
        slide["filename"] = re.search("slide name: (.*?)\n", content).group(1)
        slide["start_time"] = float(
            re.search("start time \(seconds\): (.*?)\n", content).group(1)
        )
        slide["end_time"] = float(
            re.search("end time \(seconds\): (.*?)\n", content).group(1)
        )

        slide["sentences"] = (
            re.search(
                "----\nPart_1. The text description of slide:\n\n((?:.*|\n)*?)----",
                content,
                re.MULTILINE,
            )
            .group(1)
            .strip()
        )
        slide["slide_image_understanding"] = (
            re.search(
                "----\nPart_2. The speech narrative of the presentation:\n\n((?:.*|\n)*?)----",
                content,
                re.MULTILINE,
            )
            .group(1)
            .strip()
        )
        slide["consolidated_understanding"] = (
            re.search(
                "----\nInfo-Consolidation Output:\n\n((?:.*|\n)*)",
                content,
                re.MULTILINE,
            )
            .group(1)
            .strip()
        )
        video_parts.append(slide)
    return video_parts


def _split_and_summarize_video_parts(
    video_path: Path,
    audio_path: Path,
    slide_sampling_rate: int = 60,
) -> tuple[list[Document], list[Document], list[Document]]:

    slides_dir = Path(video_path.as_posix().replace(".mp4", "__mp4_slides"))
    understanding_dir = Path(
        video_path.as_posix().replace(".mp4", "__mp4_understanding")
    )

    if slides_dir.exists() and understanding_dir.exists():
        video_parts = _load_video_parts(slides_dir, understanding_dir)
        print(f"Loading cached video parts from {understanding_dir}")
    else:
        print(f"Extracting video information from {video_path}")
        video_parts = training_video_information_extraction(
            video_path.as_posix(),
            audio_path.as_posix(),
            slide_sampling_rate=slide_sampling_rate,
        )

    if not video_parts:
        print(f"No video parts extracted from {video_path}")
        return [], [], []

    video_outputs, image_outputs, audio_outputs = [], [], []
    for pid, part in enumerate(video_parts):
        image_path = os.path.join(
            video_path.as_posix().replace(".mp4", "__mp4_slides/"), part["filename"]
        )

        general_metadata = {
            AvailableMetaData.PAGE_ID: pid,
            AvailableMetaData.FILE_TYPE: video_path.suffix,
            AvailableMetaData.STORAGE_PATH: video_path.resolve().as_posix(),
            AvailableMetaData.VIDEO_PATH: video_path.resolve().as_posix(),
            AvailableMetaData.AUDIO_PATH: audio_path.resolve().as_posix(),
            AvailableMetaData.IMAGE_PATH: Path(image_path).resolve().as_posix(),
            AvailableMetaData.START_TIME: part["start_time"],
            AvailableMetaData.END_TIME: part["end_time"],
            AvailableMetaData.TEXT: part["sentences"],
        }

        video_document = Document(
            page_content=part["consolidated_understanding"],
            metadata=copy.deepcopy(general_metadata),
        )
        video_document.metadata[AvailableMetaData.ID] = str(uuid.uuid4())
        video_document.metadata[AvailableMetaData.DOCUMENT_TYPE] = "video"

        image_document = Document(
            page_content=part["slide_image_understanding"],
            metadata=copy.deepcopy(general_metadata),
        )
        image_document.metadata[AvailableMetaData.ID] = str(uuid.uuid4())
        image_document.metadata[AvailableMetaData.DOCUMENT_TYPE] = "image"

        audio_document = Document(
            page_content=part["sentences"], metadata=copy.deepcopy(general_metadata)
        )
        audio_document.metadata[AvailableMetaData.ID] = str(uuid.uuid4())
        audio_document.metadata[AvailableMetaData.DOCUMENT_TYPE] = "audio"

        # Interlink documents
        connect_related_documents([video_document, image_document, audio_document])

        video_outputs.append(video_document)
        image_outputs.append(image_document)
        audio_outputs.append(audio_document)

    return video_outputs, image_outputs, audio_outputs


def generate_video_summaries_for_retrieval(
    video_path: str | Path,
    audio_path: str | Path = None,
    slide_sampling_rate: int = 60,
    mode: str = "precise",
) -> tuple[Document, list[Document], list[Document], list[Document]]:
    """
    Generate summaries for a video and its parts (currently split after 60 seconds).

    :param video_path: Path to the .mp4 file
    :param audio_path: Path to the .mp3 file. If none, the file will be created
    :param slide_sampling_rate: Sampling rate for the slides in seconds
    :return: Lists of langchain documents. The first document is the overall video summary. The rest are the video parts.
    """
    video_path = Path(video_path)
    if audio_path is not None:
        audio_path = Path(audio_path)
    else:
        audio_path = video_to_audio(video_path)

    (
        video_documents,
        image_documents,
        audio_documents,
    ) = _split_and_summarize_video_parts(
        video_path, audio_path, slide_sampling_rate=slide_sampling_rate
    )

    # Add check for empty documents
    if not video_documents or not image_documents or not audio_documents:
        print(f"Warning: No content extracted from video {video_path}")
        print(
            f"Video docs: {len(video_documents)}, Image docs: {len(image_documents)}, Audio docs: {len(audio_documents)}")

        # Return minimal documents or raise a more descriptive error
        raise ValueError(f"Failed to extract content from video: {video_path}")

    # Create summaries over the whole video for different modalities
    video_summary = generate_meta_summary([doc.page_content for doc in video_documents])
    image_summary = generate_meta_summary([doc.page_content for doc in image_documents])
    audio_summary = generate_meta_summary([doc.page_content for doc in audio_documents])

    general_metadata = {
        AvailableMetaData.PAGE_ID: -1,
        AvailableMetaData.FILE_TYPE: video_path.suffix,
        AvailableMetaData.STORAGE_PATH: video_path.resolve().as_posix(),
        AvailableMetaData.VIDEO_PATH: video_path.resolve().as_posix(),
        AvailableMetaData.IMAGE_PATH: video_documents[0].metadata[
            AvailableMetaData.IMAGE_PATH
        ],
        AvailableMetaData.AUDIO_PATH: audio_path.resolve().as_posix(),
        AvailableMetaData.START_TIME: video_documents[0].metadata[
            AvailableMetaData.START_TIME
        ],
        AvailableMetaData.END_TIME: video_documents[-1].metadata[
            AvailableMetaData.END_TIME
        ],
        AvailableMetaData.IS_META_SUMMARY: False,
    }

    video_meta_document = Document(
        page_content=video_summary, metadata=copy.deepcopy(general_metadata)
    )
    video_meta_document.metadata[AvailableMetaData.ID] = str(uuid.uuid4())
    video_meta_document.metadata[
        AvailableMetaData.DOCUMENT_TYPE
    ] = "video_understanding"
    video_meta_document.metadata[AvailableMetaData.TEXT] = "\n\n".join(
        [doc.page_content for doc in video_documents]
    )
    video_meta_document.metadata[AvailableMetaData.IS_META_SUMMARY] = True

    image_meta_document = Document(
        page_content=image_summary, metadata=copy.deepcopy(general_metadata)
    )
    image_meta_document.metadata[AvailableMetaData.ID] = str(uuid.uuid4())
    image_meta_document.metadata[AvailableMetaData.DOCUMENT_TYPE] = "video_image"
    image_meta_document.metadata[AvailableMetaData.TEXT] = "\n\n".join(
        [doc.page_content for doc in image_documents]
    )
    image_meta_document.metadata[AvailableMetaData.IS_META_SUMMARY] = True

    audio_meta_document = Document(
        page_content=audio_summary, metadata=copy.deepcopy(general_metadata)
    )
    audio_meta_document.metadata[AvailableMetaData.ID] = str(uuid.uuid4())
    audio_meta_document.metadata[AvailableMetaData.DOCUMENT_TYPE] = "video_audio"
    audio_meta_document.metadata[AvailableMetaData.TEXT] = "\n\n".join(
        [doc.page_content for doc in audio_documents]
    )
    audio_meta_document.metadata[AvailableMetaData.IS_META_SUMMARY] = True

    # Add links
    connect_related_documents(
        [video_meta_document, image_meta_document, audio_meta_document]
    )
    video_documents = connect_parent_and_children_documents(
        video_meta_document, video_documents
    )
    image_documents = connect_parent_and_children_documents(
        image_meta_document, image_documents
    )
    audio_documents = connect_parent_and_children_documents(
        audio_meta_document, audio_documents
    )

    # Generate one summary for document across all modalities
    documents_for_meta_summary = []
    for vid, img, aud in zip(video_documents, image_documents, audio_documents):
        documents_for_meta_summary.extend([vid, img, aud])
    meta_summary = generate_meta_summary(
        [doc.page_content for doc in documents_for_meta_summary]
    )
    meta_summary = Document(
        page_content=meta_summary, metadata=copy.deepcopy(general_metadata)
    )
    meta_summary.metadata[AvailableMetaData.DOCUMENT_TYPE] = "video_summary"
    meta_summary.metadata[AvailableMetaData.ID] = str(uuid.uuid4())
    meta_summary.metadata[AvailableMetaData.IS_META_SUMMARY] = True
    assert len(video_documents) == len(image_documents) == len(audio_documents)

    return meta_summary, video_documents, image_documents, audio_documents