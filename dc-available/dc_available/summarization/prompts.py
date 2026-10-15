from langchain_core.prompts import PromptTemplate

DOCUMENT_VARIABLE_NAME = "context"
DOCUMENT_CONTEXT_VARIABLE_NAME = "prev_context"
DOCUMENT_LANGCHAIN_PROMPT = PromptTemplate(
    input_variables=["page_content"], template="{page_content}"
)

TRAINING_SUMMARY_FORMAT_INSTRUCTIONS = """Format the output as JSON with the following keys:
title
course_description
content
tags
learning_objectives
target_audience
requirements

More infos:
- The title should be max. 50 chars.
- The course_description contains a short description of the analyzed file of your documents and we can show you hereby,
that it definitely makes sense to put this content in a short summary, so that a learner more easily can identify
if this document is really beneficial for him. So sometimes less is more so that you don’t get overwhelmed
with this kind of information and can be understood really fast and efficiently.
Our objective is to have a comprehensive course description enriched with essential keywords and compelling language
to inspire and engage potential participants.
The description should be clear, precise and a detailed content overview. Incorporate key topics and keywords into the description.
Go as deep as possible into the content and provide a detailed overview of the course content.
The description should be a single paragraph.
- The content is a list of strings (keywords related to the course content) and each entry is a string with max. 100 tokens.
- Clearly define the target audience for the course as a list; Specify who the course is oriented towards.
- Clearly state the learning objectives of the course and ensure alignment between objectives and course content as a list.
- The requirements for participation sets clear expectations for potential participants and ensures they understand what's required to enrol in the course.
Outline the prerequisites and requirements for participation in the course. Ensure prospective participants understand what is needed to enrol.
"""

GENERAL_SUMMARY_FORMAT_INSTRUCTIONS = """Format the output as JSON with the following keys:
title
description
"""

SINGLE_DOCUMENT_SUMMARY_PROMPT = PromptTemplate.from_template(
    """
Summarize the following content and extract all relevant information:
{context}

Use the following output format:

"""
    + GENERAL_SUMMARY_FORMAT_INSTRUCTIONS
)

CROSS_DOCUMENT_SUMMARY_PROMPT = PromptTemplate.from_template(
    """
Here is the summary of the previous page(s).
{prev_context}

Now, summarize the following content and extract all relevant information:
{context}

Use the following output format:

"""
    + GENERAL_SUMMARY_FORMAT_INSTRUCTIONS
)

META_DOCUMENT_SUMMARY_PROMPT = PromptTemplate.from_template(
    """
Here is the content of the previous page(s).
{context}

Now, summarize them all into a single summary.

Use the following output format:

"""
    + GENERAL_SUMMARY_FORMAT_INSTRUCTIONS
)

META_TRAINING_SUMMARY_PROMPT = PromptTemplate.from_template(
    """
You should create an overview page of the documents from an online training.

Here are the summaries from the individual training documents.
{context}

Now, create an overview page of the online training covering the above content.

Use the following output format:

"""
    + TRAINING_SUMMARY_FORMAT_INSTRUCTIONS
)


IMAGE_SUMMARY_PROMPT = (
    """
Given this PPT slide, summarize the content (text, images, diagrams, etc.)
and extract all relevant information in English.
Ignore the footnote and other irrelevant information.

Use the following output format:
"""
    + GENERAL_SUMMARY_FORMAT_INSTRUCTIONS
)


IMAGE_AND_TEXT_SUMMARY_PROMPT = (
    IMAGE_SUMMARY_PROMPT
    + "\nConsider also the following textual content from the page: "
)


PREVIOUS_CONTEXT_PROMPT = "Here is the summary of the previous page(s): "


TRUST_SCORE_PROMPT = PromptTemplate.from_template(
    """
Please provide a trust categorization for the following document.

{context}

Please also take the general information we know about the document source into account:

{prev_context}

The document should be categorized as either:
- TRUSTED: The document is trustworthy and reliable.
- WRONG: The document contains incorrect information.
- OUTDATED: The document contains outdated information.
Start your response as: CATEGORY: TRUSTED, WRONG, or OUTDATED followed by a double newline and your explanation.
"""
)
