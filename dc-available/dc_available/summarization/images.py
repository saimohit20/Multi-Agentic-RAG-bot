import uuid
from pathlib import Path

from langchain_core.documents import Document
from openai import OpenAI

from tqdm import tqdm
import openai
import dc_available.summarization.prompts as prompts
from dc_available.common.data_conversion import img_to_bytes
from dc_available.common.helpers import langchain_doc_to_string
from dc_available.config.llm_config import (
    get_openai_client,
    DEFAULT_VISION_MODEL,
)
from dc_available.RAG.data import AvailableMetaData
from dc_available.summarization.texts import (
    generate_meta_summary,
)
from dc_available.summarization.utils import (
    connect_parent_and_children_documents,
    connect_related_documents,
)


def _create_prompt(
    base64_image: str,
    text_content: str | Document = None,
    previous_context: str | Document = None,
    prompt: str = None,
) -> list[dict]:
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an AI assistant that helps people find information.",
                }
            ],
        }
    ]
    if text_content is None:
        if prompt is None:
            prompt = prompts.IMAGE_SUMMARY_PROMPT
        user_message = prompt
    else:
        if isinstance(text_content, Document):
            text_content = text_content.page_content
        if prompt is None:
            prompt = prompts.IMAGE_AND_TEXT_SUMMARY_PROMPT
        user_message = prompt + text_content
    if previous_context is not None:
        if isinstance(previous_context, Document):
            previous_context = previous_context.page_content
        user_message += "\n" + prompts.PREVIOUS_CONTEXT_PROMPT + previous_context

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_message},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    )
    return messages


def _create_prompt_from_file(
    image_path: str | Path,
    text_content: str | Document = None,
    previous_context: str = None,
    prompt: str = None,
) -> list[dict]:
    return _create_prompt(
        img_to_bytes(image_path),
        text_content=text_content,
        previous_context=previous_context,
        prompt=prompt,
    )


def generate_image_summary(
    image_path: str | Path,
    text_content: str | Document = None,
    previous_context: str = None,
    prompt: str = None,
    client: OpenAI = None,
    model: str = DEFAULT_VISION_MODEL,
    max_tokens: int = 800,
) -> str:
    messages = _create_prompt_from_file(
        image_path,
        text_content=text_content,
        previous_context=previous_context,
        prompt=prompt,
    )
    if client is None:
        client = get_openai_client()
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens
        )
        output = response.choices[0].message.content
    except openai.BadRequestError as e:
        print("!!!")
        print("Problem with image summary generation:", e)
        print("Returning empty string.")
        print("Messages:", messages)
        print(image_path)
        print("!!!")
        return ""
    return output


def generate_image_summary_for_retrieval(
    image_path: str | Path,
    text_content: str | Document = None,
    previous_context: str = None,
    client: OpenAI = None,
    model: str = DEFAULT_VISION_MODEL,
    max_tokens: int = 800,
) -> Document:
    image_path = Path(image_path)
    summary = generate_image_summary(
        image_path,
        text_content,
        previous_context,
        client=client,
        model=model,
        max_tokens=max_tokens,
    )

    metadata = {
        AvailableMetaData.ID: str(uuid.uuid4()),
        AvailableMetaData.PAGE_ID: 0,
        AvailableMetaData.DOCUMENT_TYPE: "image",
        AvailableMetaData.FILE_TYPE: image_path.suffix,
        AvailableMetaData.TEXT: langchain_doc_to_string(text_content),
        AvailableMetaData.STORAGE_PATH: image_path.resolve().as_posix(),
        AvailableMetaData.IMAGE_PATH: image_path.resolve().as_posix(),
        AvailableMetaData.IS_META_SUMMARY: False,
    }

    return Document(page_content=summary, metadata=metadata)


def generate_sequential_image_summary_for_retrieval(
    image_paths: list[str | Path],
    text_contents: list[str | Document] = None,
    client: OpenAI = None,
    model: str = DEFAULT_VISION_MODEL,
    max_tokens: int = 800,
    document_type: str = "image_sequence",
    max_pages: int = 20,
) -> list[Document]:

    if text_contents is None:
        text_contents = [None] * len(image_paths)
    if max_pages > 0:
        image_paths = image_paths[:max_pages]
        text_contents = text_contents[:max_pages]

    # Generate individual summaries
    part_documents = []
    previous_context = None
    for img_id, (image_path, text_content) in enumerate(
        tqdm(zip(image_paths, text_contents), total=len(image_paths))
    ):
        document = generate_image_summary_for_retrieval(
            image_path,
            text_content,
            previous_context,
            client=client,
            model=model,
            max_tokens=max_tokens,
        )
        document.metadata[AvailableMetaData.PAGE_ID] = img_id
        part_documents.append(document)

    # Generate meta summary
    meta_summary = generate_meta_summary([doc.page_content for doc in part_documents])
    metadata = {
        AvailableMetaData.ID: str(uuid.uuid4()),
        AvailableMetaData.PAGE_ID: -1,
        AvailableMetaData.DOCUMENT_TYPE: document_type,
        AvailableMetaData.FILE_TYPE: part_documents[0].metadata[
            AvailableMetaData.FILE_TYPE
        ],
        AvailableMetaData.TEXT: "\n\n".join(
            [doc.metadata[AvailableMetaData.TEXT] for doc in part_documents]
        ),
        AvailableMetaData.STORAGE_PATH: image_path.resolve().as_posix(),
        AvailableMetaData.IMAGE_PATH: image_path.resolve().as_posix(),
        AvailableMetaData.IS_META_SUMMARY: True,
    }
    main_document = Document(page_content=meta_summary, metadata=metadata)

    # Add links
    image_documents = connect_parent_and_children_documents(
        main_document, part_documents
    )
    if text_contents[0] is not None:
        for doc, txt in zip(part_documents, text_contents):
            if isinstance(txt, Document):
                connect_related_documents([doc, txt])
    return image_documents
