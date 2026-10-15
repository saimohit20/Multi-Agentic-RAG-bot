from pathlib import Path

from pptx import Presentation

from langchain_core.documents import Document
from openai import OpenAI

from dc_available.common.data_conversion import ppt_to_pdf
from dc_available.common.helpers import identify_pdf_type
from dc_available.config.llm_config import DEFAULT_VISION_MODEL
from dc_available.summarization.images import (
    generate_sequential_image_summary_for_retrieval,
)
from dc_available.summarization.texts import (
    load_pdf_pages_for_available,
    generate_pdf_summaries_for_retrieval,
)


def extract_comments_from_pptx(
    file_path: str | Path, add_hidden_slides: bool = False
) -> list[str]:
    ppt = Presentation(file_path)
    comments = []
    for sid, slide in enumerate(ppt.slides):
        text = slide.notes_slide.notes_text_frame.text
        # TODO: this is not a reliable method to detect hidden slides
        is_hidden = slide._element.get("show") == 0
        if is_hidden and not add_hidden_slides:
            pass
        else:
            comments.append(text)
    return comments


def generate_powerpoint_summary_for_retrieval(
    document: Path,
    client: OpenAI = None,
    model: str = DEFAULT_VISION_MODEL,
    max_tokens: int = 800,
    max_pages: int = 20,
    mode: str = "precise",
) -> tuple[list[Document], list[Document]]:
    print(f"Extract pdf and images from pptx: {document}")
    if document.suffix == ".pdf":
        text_documents, image_paths = load_pdf_pages_for_available(
            document, document.as_posix().replace(".pdf", "__pdf_images")
        )
    else:
        pdf_document = ppt_to_pdf(
            document,
            document.as_posix().replace(
                document.suffix, f"__{document.suffix[1:]}.pdf"
            ),
        )
        assert pdf_document.exists()
        text_documents, image_paths = load_pdf_pages_for_available(
            pdf_document,
            document.as_posix().replace(
                document.suffix, f"__{document.suffix[1:]}_images"
            ),
        )

    if mode == "fast":
        return text_documents, []

    # comments = extract_comments_from_pptx(document)
    assert 0 < len(image_paths)
    assert len(text_documents) == len(image_paths)
    # assert len(text_documents) == len(comments)

    image_documents = generate_sequential_image_summary_for_retrieval(
        image_paths,
        text_documents,
        client=client,
        model=model,
        max_tokens=max_tokens,
        document_type="pptx",
        max_pages=max_pages,
    )
    return text_documents, image_documents


def generate_pdf_or_pptx_summary_for_retrieval(
    document: Path, max_pages: int = 20, mode: str = "precise"
) -> tuple[list[Document], list[Document]]:
    if identify_pdf_type(document) == "pptx":
        print(f"Extract information from pdf (former pptx): {document}")
        return generate_powerpoint_summary_for_retrieval(
            document, mode=mode, max_pages=max_pages
        )
    else:
        pdf_texts, pdf_text_summaries = generate_pdf_summaries_for_retrieval(
            document, mode=mode, max_pages=max_pages
        )
        if len(pdf_texts) == 1:
            (
                image_understanding,
                image_understanding_summaries,
            ) = generate_powerpoint_summary_for_retrieval(
                document, mode=mode, max_pages=1
            )
            pdf_texts = pdf_texts + image_understanding
            pdf_text_summaries = pdf_text_summaries + image_understanding_summaries
        return pdf_texts, pdf_text_summaries



