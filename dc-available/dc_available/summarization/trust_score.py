import json
from enum import Enum
from pathlib import Path
from typing import List, Tuple

from langchain.chains import LLMChain
from langchain_core.documents import Document
from langchain_core.language_models import LLM

from dc_available.config.llm_config import get_langchain_llm
from dc_available.RAG.data import AvailableMetaData
from dc_available.summarization import prompts


class AvailableTrustScore(str, Enum):
    """Enumerator of the different trust scores for documents"""

    TRUSTED = "TRUSTED"
    OUTDATED = "OUTDATED"
    WRONG = "WRONG"
    UNCHECKED = "UNCHECKED"


def check_for_trust_file(document: Document) -> Tuple[AvailableTrustScore, str]:
    if AvailableMetaData.STORAGE_PATH not in document.metadata:
        print("No STORAGE_PATH in metadata")
        return AvailableTrustScore.UNCHECKED, None

    trust_file = (
        Path(document.metadata[AvailableMetaData.STORAGE_PATH]).parent / "trust.json"
    )
    print(f"Looking for trust file at: {trust_file}")

    if not trust_file.exists():
        print("Trust file does not exist")
        return AvailableTrustScore.UNCHECKED, None


    with open(trust_file, "r", encoding="utf-8") as fin:
        trust_data = json.load(fin)

    print(f"Trust data loaded: {trust_data}")
    trust_score = AvailableTrustScore(trust_data["rating"].upper())
    print(f"Trust score: {trust_score}")
    explanation = trust_data["description"]
    return trust_score, explanation


def get_trust_score(
    document: Document, additional_pages: List[Document] = None, llm: LLM = None
) -> Tuple[AvailableTrustScore, str]:
    """Calculate the trust score of a document and an explanation with a simple LLM call."""

    trust_score, data_source_information = check_for_trust_file(document)
    if trust_score != AvailableTrustScore.UNCHECKED:
        return trust_score, data_source_information

    if data_source_information is None:
        data_source_information = "we do not have any information about the source."

    if not additional_pages:
        additional_pages = []
        # TODO: adapt / not used currently
    text = document.page_content

    if llm is None:
        llm = get_langchain_llm()
    trust_score_chain = LLMChain(llm=llm, prompt=prompts.TRUST_SCORE_PROMPT)
    response = trust_score_chain.invoke(
        {
            prompts.DOCUMENT_VARIABLE_NAME: text,
            prompts.DOCUMENT_CONTEXT_VARIABLE_NAME: data_source_information,
        }
    )

    output = response["text"]
    trust_score, explanation = output.split("\n\n", 1)
    trust_score = trust_score.split(":")[1].strip().upper()
    trust_score = AvailableTrustScore(trust_score)
    return trust_score, explanation


def get_meta_trust_score(
    meta_document: Document, all_documents: List[Document]
) -> Tuple[AvailableTrustScore, str]:
    """Calculate the trust score of a collection of documents based on their trust scores.
    - If more than 10% of the documents are wrong --> wrong
    - If more than 25% of the documents are outdated --> outdated
    - If more than 33% of the documents are unchecked --> unchecked
    - Otherwise --> trusted
    """

    if len(all_documents) == 0:
        return get_trust_score(meta_document)

    trust_scores = [
        doc.metadata.get(AvailableMetaData.TRUST_SCORE, None) for doc in all_documents
    ]
    trust_scores = [score for score in trust_scores if score is not None]

    wrong_document_count = trust_scores.count(AvailableTrustScore.WRONG) / len(
        trust_scores
    )
    outdated_document_count = trust_scores.count(AvailableTrustScore.OUTDATED) / len(
        trust_scores
    )
    trusted_document_count = trust_scores.count(AvailableTrustScore.TRUSTED) / len(
        trust_scores
    )
    unchecked_document_count = trust_scores.count(AvailableTrustScore.UNCHECKED) / len(
        trust_scores
    )

    # if >10% of documents are wrong --> wrong
    if wrong_document_count > 0.1:
        return (
            AvailableTrustScore.WRONG,
            f"{round(wrong_document_count,1)}% of pages are wrong.",
        )

    # if >25% of documents are outdated --> outdated
    if outdated_document_count > 0.25:
        return (
            AvailableTrustScore.OUTDATED,
            f"{round(outdated_document_count,1)}% of pages are outdated.",
        )

    # if >33% of documents are unchecked --> unchecked
    if unchecked_document_count > 0.33:
        return (
            AvailableTrustScore.UNCHECKED,
            f"{round(unchecked_document_count,1)}% of pages are unchecked.",
        )

    # Otherwise --> trusted
    return (
        AvailableTrustScore.TRUSTED,
        f"{round(trusted_document_count,1)}% of pages are trusted.",
    )


if __name__ == "__main__":
    document = Document("The wheel was never invented.", metadata={"source": "nobody"})
    print(get_trust_score(document))
