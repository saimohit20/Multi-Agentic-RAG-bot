import json
import uuid
from pathlib import Path
from typing import Iterable

import fitz

from extract_msg import Message

from langchain_core.documents import Document


def load_pdf_pages(pdf_path: str | Path) -> list[Document]:
    pdf_path = Path(pdf_path)

    documents = []
    for page in fitz.open(pdf_path.as_posix()):
        text = page.get_text()
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": pdf_path.name,
                    "page": page.number,
                    "type": pdf_path.suffix,
                },
            )
        )
    return documents


def load_msg_file(msg_path: str | Path) -> tuple[Document, list[Path]]:
    msg_path = Path(msg_path)
    msg = Message(msg_path)

    # Extract attachments
    attachments = []
    attachment_paths = []
    attachment_output_path = Path(
        str(msg_path.absolute()).replace(".msg", "__msg_attachments")
    )
    for attachment in msg.attachments:
        attachment_output_path.mkdir(exist_ok=True)
        with open(attachment_output_path / attachment.longFilename, "wb") as f:
            f.write(attachment.data)
        attachment_paths.append(attachment_output_path / attachment.longFilename)
        attachment_data = {
            "filename": attachment.longFilename,
            "source": attachment_output_path / attachment.longFilename,
        }
        attachments.append(attachment_data)

    text = f"Subject: {msg.subject}\n"
    text += f"Date: {msg.date}\n"
    text += f"Sender: {msg.sender}\n"
    text += f"Recipients: {msg.to}\n"
    text += f"CC: {msg.cc}\n"
    text += f"BCC: {msg.cc}\n"
    text += f"\n{msg.body.strip()}"

    doc = Document(
        page_content=text,
        metadata={
            "source": msg.path,
            "subject": msg.subject,
            "date": msg.date,
            "sender": msg.sender,
            "recipients": msg.to,
            "recipients_cc": msg.cc,
            "recipients_bcc": msg.bcc,
            "attachments": attachments,
        },
    )
    return doc, attachment_paths


def save_docs_to_jsonl(array: Iterable[Document], file_path: str | Path) -> None:
    with open(file_path, "w", encoding="utf-8") as jsonl_file:
        for doc in array:
            jsonl_file.write(doc.json() + "\n")


def load_docs_from_jsonl(file_path: str | Path) -> list[Document]:
    array = []
    with open(file_path, "r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array
