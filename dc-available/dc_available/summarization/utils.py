from typing import List, Optional

from langchain.output_parsers import PydanticOutputParser
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from dc_available.RAG.data import AvailableMetaData


def connect_parent_and_children_documents(parent: Document, children: list[Document]):
    if AvailableMetaData.LINK_TO_CHILDREN not in parent.metadata:
        parent.metadata[AvailableMetaData.LINK_TO_CHILDREN] = []

    prev_child = None
    for child in children:
        parent.metadata[AvailableMetaData.LINK_TO_CHILDREN].append(
            child.metadata[AvailableMetaData.ID]
        )
        child.metadata[AvailableMetaData.LINK_TO_PARENT] = parent.metadata[
            AvailableMetaData.ID
        ]
        if prev_child is not None:
            child.metadata[AvailableMetaData.LINK_TO_PREVIOUS] = prev_child.metadata[
                AvailableMetaData.ID
            ]
            prev_child.metadata[AvailableMetaData.LINK_TO_NEXT] = child.metadata[
                AvailableMetaData.ID
            ]
            prev_child = child

    return [parent] + children


def connect_related_documents(documents: list[Document]):
    for doc in documents:
        if AvailableMetaData.LINK_TO_RELATED not in doc.metadata:
            doc.metadata[AvailableMetaData.LINK_TO_RELATED] = []

        other_ids = [d.metadata[AvailableMetaData.ID] for d in documents if d != doc]
        other_ids = [
            idx
            for idx in other_ids
            if idx not in doc.metadata[AvailableMetaData.LINK_TO_RELATED]
        ]
        doc.metadata[AvailableMetaData.LINK_TO_RELATED].extend(other_ids)


class Summarizer(BaseModel):
    title: Optional[str] = Field("Title")
    short_description: Optional[str] = Field("Short Description")
    content: List[str] = Field("Bullet Points containing content")
    tags: List[str] = Field("Short keywords describing the problem")


parser = PydanticOutputParser(pydantic_object=Summarizer)
