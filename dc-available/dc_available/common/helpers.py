import os
import base64
import shutil
import tempfile
from pathlib import Path
import re
from typing import List

from langchain_core.documents import Document


import platform


def get_operating_system() -> str:
    system = platform.system()
    if system == "Darwin":
        return "Mac"
    elif system == "Linux":
        return "Linux"
    elif system == "Windows":
        return "Windows"
    else:
        return "Unknown"


def on_docker() -> bool:
    """Checks if the code is running in a Docker environment."""
    return os.path.exists("/.dockerenv") or os.environ.get("IN_DOCKER", "0") == "1"


def on_azure():
    """Checks if the code is running in an Azure environment."""
    # Look for some common Azure environment variables
    azure_env_vars = ["WEBSITE_INSTANCE_ID", "MSI_ENDPOINT", "AZURE_HTTP_USER_AGENT"]
    return any(var in os.environ for var in azure_env_vars)


def delete_directory(directory_path: str | Path):
    path = Path(directory_path)
    if path.exists() and path.is_dir():
        shutil.rmtree(path)


def langchain_doc_to_string(doc: str | Document) -> str:
    if doc is None:
        text_content = ""
    elif isinstance(doc, Document):
        text_content = doc.page_content
    else:
        text_content = doc
    return text_content


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


### Helper to structure and beautify rag chain with sources answer
def beatufiy_rag_chain_answer(full_response: dict) -> str:
    """Function to beautify the answer from rag chain with sources.

    Args:
        full_response: Response from rag chain with sources.

    Returns:
        template: Beautified answer with sources if available.
    """
    extracted_sources = [
        doc.dict()["metadata"]["source"] for doc in full_response["context"]
    ]
    nat_pattern = r".*?(f\d{2})"
    extracted_sources = [re.sub(nat_pattern, r"\1", path) for path in extracted_sources]
    if extracted_sources == [] or (
        full_response["question"] in ["Hello", "hello", "Hello?", "hello?"]
    ):
        template = "{answer}".format(answer=full_response["answer"])
    else:
        template = "{answer}  <br><br> You find more information in following document(s):<br> ▪ {context}".format(
            answer=full_response["answer"], context="<br> ▪ ".join(extracted_sources)
        )
    return template


def _get_temporary_folder():
    temp_dir = tempfile.mkdtemp()
    return temp_dir


def _check_path(filename: Path) -> Path:
    if not filename.exists():
        print(f"Expected file not found: {filename}")
        raise SystemExit("Conversion must have failed or file is missing.")
    return filename


def join_paths(
    given_path: str | Path, local_path: str | Path, joint_folder: str
) -> Path:
    given_path = Path(given_path)
    local_path = Path(local_path)
    g_index = given_path.parts.index(joint_folder)
    l_index = local_path.parts.index(joint_folder)
    root = os.sep.join(local_path.parts[:l_index])
    if root.startswith("//"):
        root = root[1:]
    target = os.sep.join(given_path.parts[g_index:])
    return Path(root + os.sep + target)


ACCEPTED_CONTENT_EXTS = {".pdf", ".pptx", ".pptm", ".mp4", ".msg", ".xlsx", ".docx", ".html"}


def topic_from_storage_path(storage_path: str | Path) -> str | None:
    """Topic from a path that ends in  .../<topic>/<level>/<training>/<file>.

    Anchors from the bottom, so any wrapper folders above the topic (e.g. a
    dataset name like 'GROWNFIELD') are ignored. Works for both file paths
    (chunks) and folder paths (a training folder).
    """
    parts = [p for p in str(storage_path).replace("\\", "/").split("/") if p]
    if parts and Path(parts[-1]).suffix.lower() in ACCEPTED_CONTENT_EXTS:
        parts = parts[:-1]   # drop the filename -> path ends at the training folder
    return parts[-3] if len(parts) >= 3 else None


def path_display_name(path: str | Path, base: str | Path) -> str:
    """Readable label for `path` relative to `base`.

    Robust when `path` is not under `base` — e.g. the 'available' data source
    stores files under dc_available/data/... while DATA_DIR points at resources/.
    Falls back to the path relative to its 'data' root, else the file name.
    """
    path = Path(path)
    try:
        return str(path.relative_to(base))
    except ValueError:
        parts = path.parts
        if "data" in parts:
            return "/".join(parts[parts.index("data") + 1 :])
        return path.name


def identify_pdf_type(file_path: str | Path):
    import PyPDF2

    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        page = reader.pages[0]
        width = page.mediabox.width
        height = page.mediabox.height
        metadata = reader.metadata

    creator = metadata.get("/Creator", "").lower()
    producer = metadata.get("/Producer", "").lower()

    if width > height:  # and (round(width/height) in [round(16/9), round(4/3)]):
        return "pptx"
    elif "powerpoint" in creator or "powerpoint" in producer:
        return "pptx"
    return "pdf"
