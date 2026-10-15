# The source code in this file is adapted from AutoGen
# https://github.com/microsoft/autogen/blob/cc836de13d111b0e082d71501b2b867faab68941/notebook/agentchat_RetrieveChat.ipynb
# and
# https://github.com/microsoft/autogen/blob/cc836de13d111b0e082d71501b2b867faab68941/notebook/agentchat_groupchat_RAG.ipynb
# the original source code is licensed as follows:
#
# MIT License
#
# Copyright (c) Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE
#
#
# The adaptations are subject to
#
# Copyright (C) Robert Bosch GmbH 2024.
#
# All rights reserved, also regarding any disposal, exploitation,
# reproduction, editing, distribution, as well as in the event of
# applications for industrial property rights.
#
# Code in this file was created with assistance by GitHub Copilot

from typing import Annotated

from dc_available.common.language_manager import get_language_manager

import streamlit as st

from dc_available.config.llm_config import get_autogen_llm_config, DEFAULT_MODEL
from dc_available.agents.curriculum_agents import (
    LangchainRetrieveUserProxyAgent,
)
from dc_available.config.data_config import (
    DEFAULT_INDEX_ARTIFACTS_PATH,
    DEFAULT_INGESTION_ARTIFACTS_PATH,
)
from dc_available.frontend.streamlit_helpers import (
    display_document_in_streamlit,
)

llm_config = get_autogen_llm_config(model_name=DEFAULT_MODEL)


def create_ragproxyagent():
    ragproxyagent = LangchainRetrieveUserProxyAgent(
        name="ragproxyagent",
        retrieve_config={
            "task": "qa",
            "docs_path": DEFAULT_INGESTION_ARTIFACTS_PATH,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",#"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", #"sentence-transformers/all-MiniLM-L6-v2",
            "get_or_create": True,
            "index_cache_path": DEFAULT_INDEX_ARTIFACTS_PATH,
        },
        code_execution_config=False,
    )
    return ragproxyagent


def get_information_on_topic(
        topic: Annotated[str, "the topic on which to gather information"],
        n_results: int = 3,
) -> str:
    """Use RAG to get information on a topic with language support."""
    language_manager = get_language_manager()

    # Add language context to the topic if not English
    if language_manager.current_language and language_manager.current_language != 'en':
        lang_name = language_manager.supported_languages.get(language_manager.current_language)
        topic_with_lang = f"{topic} (respond in {lang_name})"
    else:
        topic_with_lang = topic

    with st.status(f"Searching for information... [{topic}]", expanded=True) as status:
        print(f"[Retrieval] Search: {topic_with_lang}")
        st.write(f"Setup ...")
        ragproxyagent = create_ragproxyagent()
        st.write(f"Search ...")
        ragproxyagent.n_results = n_results
        context = {"problem": topic_with_lang, "n_results": n_results}
        ret_msg = ragproxyagent.message_generator(
            sender=ragproxyagent, recipient=None, context=context
        )
        st.write(f"Check ...")
        msg = {"content": ret_msg, "role": "assistant"}

        if status:
            status.update(
                label=f"Search complete! [{topic}]", state="complete", expanded=False
            )

    for training in ragproxyagent._results:
        docId = 0
        for result in training:
            if "original" in result[0]:
                for doc in result[0]["original"]:
                    display_document_in_streamlit(doc, docId)
                    docId += 1
    del ragproxyagent

    # Draw the graph traversal (vector search -> entities -> graph chunks), reusing
    # the same renderer as the QA page so both views look identical.
    traversal = st.session_state.get("graph_traversal")
    if traversal and (traversal["vector_chunks"] or traversal["graph_chunks"]):
        from dc_available.RAG.qa_functions import _build_traversal_graph_html

        with st.expander("Show Graph Traversal", expanded=False):
            vector_count = len(traversal["vector_chunks"])
            graph_count = len(traversal["graph_chunks"])
            entity_count = len(traversal.get("entity_to_expanded", {}))
            st.caption(
                f"Blue = {vector_count} Vector chunks  |  "
                f"Orange = {entity_count} entities  |  "
                f"Green = {graph_count} graph-expanded chunks"
            )
            try:
                html = _build_traversal_graph_html(traversal)
                st.components.v1.html(html, height=540, scrolling=False)
            except Exception as e:
                st.warning(f"Graph rendering failed: {e}")

    return msg