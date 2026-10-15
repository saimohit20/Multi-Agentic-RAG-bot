import re
import urllib.parse
from typing import Any, List, Optional

from langchain.output_parsers import PydanticOutputParser
from langchain_core.documents import Document
from pydantic import BaseModel, Field


# %%
class GeneralSummarizer(BaseModel):
    title: Optional[str] = Field("Title")
    description: Optional[str] = Field("Description")


class TrainingSummarizer(BaseModel):
    title: Optional[str] = Field("Title")
    course_description: Optional[str] = Field("Course Description")
    content: List[str] = Field("Bullet Points containing content")
    tags: List[str] = Field("Short keywords describing the problem")
    learning_objectives: List[str] = Field("Learning Objectives")
    target_audience: List[str] = Field("Target Audience")
    requirements: List[str] = Field("Participation Requirements")


general_parser = PydanticOutputParser(pydantic_object=GeneralSummarizer)
training_parser = PydanticOutputParser(pydantic_object=TrainingSummarizer)

MARKDOWN_TEMPLATE_TRAINING = """
# Overview
## $TITLE

$COURSE_DESCRIPTION

## Content

$CONTENT_AS_LIST

## Tags

$TAGS_AS_LIST

## Learning Objectives

$LEARNING_OBJECTIVES

## Target Audience

$TARGET_AUDIENCE

## Requirements

$REQUIREMENTS
"""


MARKDOWN_TEMPLATE_DOCUMENT = """
# $TITLE

$DESCRIPTION
"""

MARKDOWN_TEMPLATE_RETRIEVAL = """
# $TITLE

$DESCRIPTION
"""

INDEX_TEMPLATE = """
# Overview

## AVAILABLE Handbook
Created by CR/AIR within AIM-205
"""

MK_YAML_TEMPLATE = """site_name: AVAILABLE Handbook

theme:
  name: material
"""


def escape_urls(text: Any, for_markdown: bool = False):
    """Escape urls in a given text."""
    text = str(text)
    url_pattern = re.compile(r"(https?://\S+)")
    urls = re.findall(url_pattern, text)

    for url in urls:
        escaped_url = urllib.parse.quote(url, safe=":/")
        if for_markdown:
            escaped_url = f"[{url}]({escaped_url})"
        text = text.replace(url, escaped_url)
    return text


def to_markdown(
    parsed_response,
    template: str = MARKDOWN_TEMPLATE_DOCUMENT,
    meta_data: dict[str, str] = None,
) -> str:
    """Convert the parsed response to markdown."""
    md = template.replace("$TITLE", parsed_response.title)

    if template == MARKDOWN_TEMPLATE_TRAINING:
        md = md.replace("$COURSE_DESCRIPTION", parsed_response.course_description)
        md = md.replace(
            "$CONTENT_AS_LIST",
            "\n".join(f"- {content}" for content in parsed_response.content),
        )
        md = md.replace(
            "$TAGS_AS_LIST", "\n".join(f"- {tag}" for tag in parsed_response.tags)
        )
        md = md.replace(
            "$LEARNING_OBJECTIVES",
            "\n".join(f"- {tag}" for tag in parsed_response.learning_objectives),
        )
        md = md.replace(
            "$TARGET_AUDIENCE",
            "\n".join(f"- {tag}" for tag in parsed_response.target_audience),
        )
        md = md.replace(
            "$REQUIREMENTS",
            "\n".join(f"- {tag}" for tag in parsed_response.requirements),
        )
    else:
        md = md.replace("$DESCRIPTION", parsed_response.description)

    if template != MARKDOWN_TEMPLATE_RETRIEVAL:
        if meta_data is not None and len(meta_data) > 0:
            md += "\n\n## Meta-Data\n\n"
            for key, value in meta_data.items():
                md += f"- {key}: {escape_urls(value, for_markdown=True)}\n"
    return md


def doc_to_markdown(doc: Document, method: str = "general") -> str:
    try:
        if method == "general":
            parsed_response = general_parser.parse(doc.page_content)
            return to_markdown(
                parsed_response,
                template=MARKDOWN_TEMPLATE_DOCUMENT,
                meta_data=doc.metadata,
            )
        elif method == "retrieval":
            parsed_response = general_parser.parse(doc.page_content)
            return to_markdown(
                parsed_response,
                template=MARKDOWN_TEMPLATE_RETRIEVAL,
                meta_data=doc.metadata,
            )
        elif method == "training":
            parsed_response = training_parser.parse(doc.page_content)
            return to_markdown(
                parsed_response,
                template=MARKDOWN_TEMPLATE_TRAINING,
                meta_data=doc.metadata,
            )
        else:
            return general_parser.parse(doc.page_content).description
    except Exception as e:
        return doc.page_content
