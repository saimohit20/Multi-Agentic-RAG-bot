import copy
import os
import uuid
from pathlib import Path
from typing import List

from tqdm import tqdm

import fitz
from markdownify import markdownify
from langchain.chains import LLMChain, RefineDocumentsChain
from langchain_core.documents import Document
from langchain_core.language_models import LLM

import dc_available.summarization.prompts as prompts
from dc_available.common.data_loading import load_msg_file
from dc_available.config.llm_config import get_langchain_llm
from dc_available.RAG.data import AvailableMetaData
from dc_available.summarization.utils import (
    connect_parent_and_children_documents,
    connect_related_documents,
)


def generate_text_summary(
    document: Document, previous_context: str = None, llm: LLM = None
) -> str:
    if llm is None:
        llm = get_langchain_llm()

    if previous_context is not None and previous_context:
        cross_document_summary_llm_chain = LLMChain(
            llm=llm, prompt=prompts.CROSS_DOCUMENT_SUMMARY_PROMPT
        )

        response = cross_document_summary_llm_chain.invoke(
            {
                prompts.DOCUMENT_VARIABLE_NAME: document.page_content,
                prompts.DOCUMENT_CONTEXT_VARIABLE_NAME: previous_context,
            }
        )

    else:
        single_document_summary_llm_chain = LLMChain(
            llm=llm, prompt=prompts.SINGLE_DOCUMENT_SUMMARY_PROMPT
        )
        response = single_document_summary_llm_chain.invoke(
            {
                prompts.DOCUMENT_VARIABLE_NAME: document.page_content,
            }
        )
    return response["text"]


def generate_meta_summary(
        summaries: list[str],
        llm: LLM = None,
        split_size: int = 50,
        prompt: str = prompts.META_DOCUMENT_SUMMARY_PROMPT,
) -> str:
    if llm is None:
        llm = get_langchain_llm()

    # Better error message with context
    if len(summaries) == 0:
        raise AssertionError(
            "Cannot create meta summaries without input documents. "
            "This usually happens when:\n"
            "1. Video processing failed to extract content\n"
            "2. No valid documents were found in the folder\n"
            "3. All documents failed to process due to errors\n"
            "Check the logs above for specific processing errors."
        )

    meta_document_summary_llm_chain = LLMChain(llm=llm, prompt=prompt)

    meta_summaries = []
    for i in tqdm(range(0, len(summaries), split_size)):
        content_to_summarize = "\n\n".join(summaries[i: i + split_size])
        meta_response = meta_document_summary_llm_chain(content_to_summarize)
        meta_summaries.append(meta_response["text"])

    if len(meta_summaries) > 1:
        meta_response = meta_document_summary_llm_chain("\n\n".join(meta_summaries))
        return meta_response["text"]
    else:
        return meta_summaries[0]


def generate_meta_summary_as_document(
    documents: list[Document], other_documents: list[Document] = None
) -> Document:
    if other_documents is None:
        other_documents = []

    all_texts = []
    for i in range(max(len(documents), len(other_documents))):
        if i < len(documents):
            all_texts.append(documents[i].page_content)
        if i < len(other_documents):
            all_texts.append(other_documents[i].page_content)
    meta_summary = generate_meta_summary(all_texts)

    meta_document = Document(
        page_content=meta_summary, metadata=copy.deepcopy(documents[0].metadata)
    )
    meta_document.metadata[AvailableMetaData.ID] = str(uuid.uuid4())
    meta_document.metadata[AvailableMetaData.PAGE_ID] = -1
    meta_document.metadata[AvailableMetaData.FILE_TYPE] = "text_summary"
    meta_document.metadata[AvailableMetaData.TEXT] = (
        "\n\n".join([doc.page_content for doc in documents]),
    )
    meta_document.metadata[AvailableMetaData.IS_META_SUMMARY] = True

    _ = connect_parent_and_children_documents(meta_document, documents)
    _ = connect_parent_and_children_documents(meta_document, other_documents)

    return meta_document


def generate_sequential_documents_summary(
    documents: list[Document], llm: LLM = None, max_pages: int = 20
) -> tuple[list[str], str]:
    if llm is None:
        llm = get_langchain_llm()
    if len(documents) == 1:
        return [], generate_text_summary(documents[0], llm=llm)

    single_document_summary_llm_chain = LLMChain(
        llm=llm, prompt=prompts.SINGLE_DOCUMENT_SUMMARY_PROMPT
    )
    cross_document_summary_llm_chain = LLMChain(
        llm=llm, prompt=prompts.CROSS_DOCUMENT_SUMMARY_PROMPT
    )
    refine_document_summary_llm_chain = RefineDocumentsChain(
        initial_llm_chain=single_document_summary_llm_chain,
        refine_llm_chain=cross_document_summary_llm_chain,
        document_prompt=prompts.DOCUMENT_LANGCHAIN_PROMPT,
        document_variable_name=prompts.DOCUMENT_VARIABLE_NAME,
        initial_response_name=prompts.DOCUMENT_CONTEXT_VARIABLE_NAME,
        return_intermediate_steps=True,
    )

    if max_pages > 0:
        documents = documents[:max_pages]
    response = refine_document_summary_llm_chain(documents)

    meta_document_summary_llm_chain = LLMChain(
        llm=llm, prompt=prompts.META_DOCUMENT_SUMMARY_PROMPT
    )
    meta_response = meta_document_summary_llm_chain(
        "\n\n".join(response["intermediate_steps"])
    )
    return response["intermediate_steps"], meta_response["text"]


def generate_sequential_document_summary_for_retrieval(
    documents: list[Document], llm: LLM = None, max_pages: int = 20
) -> List[Document]:
    all_summaries, meta_summary = generate_sequential_documents_summary(
        documents, llm=llm, max_pages=max_pages
    )

    meta_document = Document(
        page_content=meta_summary, metadata=copy.deepcopy(documents[0].metadata)
    )
    meta_document.metadata[AvailableMetaData.ID] = str(uuid.uuid4())
    meta_document.metadata[AvailableMetaData.PAGE_ID] = -1
    meta_document.metadata[AvailableMetaData.FILE_TYPE] = "text_summary"
    meta_document.metadata[AvailableMetaData.TEXT] = (
        "\n\n".join([doc.page_content for doc in documents]),
    )
    meta_document.metadata[AvailableMetaData.IS_META_SUMMARY] = True

    summary_documents = []
    for sid, summary in enumerate(all_summaries):
        doc = Document(
            page_content=summary, metadata=copy.deepcopy(documents[sid].metadata)
        )
        doc.metadata[AvailableMetaData.ID] = str(uuid.uuid4())
        doc.metadata[AvailableMetaData.PAGE_ID] = sid
        doc.metadata[AvailableMetaData.FILE_TYPE] = "text_summary"
        doc.metadata[AvailableMetaData.TEXT] = documents[sid].page_content
        doc.metadata[AvailableMetaData.IS_META_SUMMARY] = False
        connect_related_documents([doc, documents[sid]])
        summary_documents.append(doc)

    summary_documents = connect_parent_and_children_documents(
        meta_document, summary_documents
    )
    return summary_documents


def load_pdf_pages_for_available(
    pdf_path: str | Path, output_img_path: str | Path = None
) -> tuple[list[Document], list[Path]]:
    if output_img_path is not None:
        os.makedirs(output_img_path, exist_ok=True)
        output_img_path = Path(output_img_path)
    pdf_path = Path(pdf_path)

    image_paths, documents = [], []

    doc = fitz.open(pdf_path)
    image_paths = []
    for i in range(len(doc)):
        page = doc.load_page(i)

        if output_img_path is not None:
            pix = page.get_pixmap()
            image_path = output_img_path / f"page_{i + 1}.png"
            pix.save(image_path)
            image_paths.append(image_path.resolve())

        text = page.get_text()
        documents.append(
            Document(
                page_content=text,
                metadata={
                    AvailableMetaData.ID: str(uuid.uuid4()),
                    AvailableMetaData.PAGE_ID: i,
                    AvailableMetaData.DOCUMENT_TYPE: "text",
                    AvailableMetaData.FILE_TYPE: pdf_path.suffix,
                    AvailableMetaData.STORAGE_PATH: pdf_path.resolve().as_posix(),
                },
            )
        )
    return documents, image_paths


def generate_pdf_summaries_for_retrieval(
    document: Path, max_pages: int = 20, mode: str = "fast"
) -> tuple[list[Document], list[Document]]:
    print(f"Extract information from pdf: {document}")
    # TODO use images later
    image_base_path = document.as_posix().replace(".pdf", "__pdf_images")
    text_documents, image_paths = load_pdf_pages_for_available(
        document, image_base_path
    )
    assert 0 < len(text_documents), len(text_documents)
    # assert len(image_paths) == len(text_documents), (len(image_paths), len(text_documents))
    if mode == "precise":
        summaries = generate_sequential_document_summary_for_retrieval(
            text_documents, max_pages=max_pages
        )
    else:
        summaries = text_documents
    return text_documents, summaries


def generate_msg_summary_for_retrieval(
    document: Path,
) -> tuple[Document, Document, list[Path]]:
    print(f"Extract information from msg: {document}")
    text_document, attachment_paths = load_msg_file(document)
    text_document.metadata[AvailableMetaData.ID] = str(uuid.uuid4())
    text_document.metadata[AvailableMetaData.PAGE_ID] = 0
    text_document.metadata[AvailableMetaData.DOCUMENT_TYPE] = "text"
    text_document.metadata[AvailableMetaData.FILE_TYPE] = document.suffix
    text_document.metadata[
        AvailableMetaData.STORAGE_PATH
    ] = document.resolve().as_posix()

    summary = generate_text_summary(text_document)
    summary_document = Document(summary, metadata=copy.deepcopy(text_document.metadata))
    summary_document.metadata[AvailableMetaData.ID] = str(uuid.uuid4())
    summary_document.metadata[AvailableMetaData.PAGE_ID] = -1
    summary_document.metadata[AvailableMetaData.FILE_TYPE] = "text_summary"
    summary_document.metadata[AvailableMetaData.TEXT] = text_document.page_content
    summary_document.metadata[AvailableMetaData.IS_META_SUMMARY] = True

    return text_document, summary_document, attachment_paths


def generate_html_summary_for_retrieval(
    document: Path,
) -> tuple[Document, Document]:
    print(f"Extract information from html: {document}")
    # HTML (e.g. an exported Confluence page) -> Markdown, so tables, headings and
    # lists survive as text. Then it is summarized exactly like any other text file.
    html = Path(document).read_text(encoding="utf-8", errors="ignore")
    markdown = markdownify(html)

    text_document = Document(
        page_content=markdown,
        metadata={
            AvailableMetaData.ID: str(uuid.uuid4()),
            AvailableMetaData.PAGE_ID: 0,
            AvailableMetaData.DOCUMENT_TYPE: "text",
            AvailableMetaData.FILE_TYPE: document.suffix,
            AvailableMetaData.STORAGE_PATH: document.resolve().as_posix(),
        },
    )

    summary = generate_text_summary(text_document)
    summary_document = Document(summary, metadata=copy.deepcopy(text_document.metadata))
    summary_document.metadata[AvailableMetaData.ID] = str(uuid.uuid4())
    summary_document.metadata[AvailableMetaData.PAGE_ID] = -1
    summary_document.metadata[AvailableMetaData.FILE_TYPE] = "text_summary"
    summary_document.metadata[AvailableMetaData.TEXT] = text_document.page_content
    summary_document.metadata[AvailableMetaData.IS_META_SUMMARY] = True

    return text_document, summary_document