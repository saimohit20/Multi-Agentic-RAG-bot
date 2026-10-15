# Copyright (C) Robert Bosch GmbH 2024.
#
# All rights reserved, also regarding any disposal, exploitation,
# reproduction, editing, distribution, as well as in the event of
# applications for industrial property rights.
#
# Code in this file was created with assistance by GitHub Copilot


import os
from typing import Any, Callable, Dict, List, Optional

from dc_available.common.language_manager import get_language_manager, get_localized_system_message

import chromadb
import opik
import streamlit as st
from autogen import Agent, ConversableAgent, UserProxyAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

from dc_available.config.llm_config import get_autogen_llm_config, DEFAULT_MODEL
from dc_available.frontend.streamlit_helpers import (
    special_chat_quit,
)
from dc_available.RAG.retrieval_helpers import (
    get_retriever,
)
from dc_available.RAG.data import AvailableMetaData
from dc_available.common.helpers import topic_from_storage_path
from dc_available.config.data_config import DEFAULT_GRAPH_ARTIFACTS_PATH
from dc_available.graph.retrieval.entity_hop_retriever import (
    expand_chunks_via_graph_with_traversal,
    infer_topic_keys_from_storage_paths,
)

CODE_EXECUTION_CONFIG = {"work_dir": "tmp", "use_docker": False}
ASSISTANT_GRAPH_EXPANSION_MAX = 10
ASSISTANT_GRAPH_SCORE_WEIGHT = 0.5
ASSISTANT_FINAL_CONTEXT_MAX = 5  # chunks sent to the LLM after reranking
ASSISTANT_RETRIEVAL_DEBUG_DEFAULT = True


def _is_retrieval_debug_enabled() -> bool:
    value = os.getenv("RAG_DEBUG_RETRIEVAL")
    if value is None:
        return ASSISTANT_RETRIEVAL_DEBUG_DEFAULT
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _debug_print_ids(stage: str, chunk_ids: list[str], extra: str = "") -> None:
    if not _is_retrieval_debug_enabled():
        return
    limited = chunk_ids[:10]
    suffix = "" if len(chunk_ids) <= 10 else f" ... (+{len(chunk_ids) - 10} more)"
    print(f"[RetrievalDebug] {stage}: {len(chunk_ids)} -> {limited}{suffix}{extra}")


def _infer_topic_keys_from_docs(documents: list[Any]) -> list[str]:
    paths = [doc.metadata.get(AvailableMetaData.STORAGE_PATH, "") for doc in documents]
    return infer_topic_keys_from_storage_paths(paths, DEFAULT_GRAPH_ARTIFACTS_PATH)


def remove_indents(text: str) -> str:
    """Remove all leading whitespace from the beginning of each line."""
    lines = text.split("\n")
    return "\n".join([line.lstrip() for line in lines])


@opik.track(name="vector_retrieval", capture_output=False)
def _vector_retrieval(vectorstore, problem: str, n_results: int):
    """Vector search step, traced so it nests under the agent stage trace
    (mirrors the QA pipeline's vector_retrieval span). The raw langchain tuples
    are suppressed as span output; clean scored chunks are logged as metadata."""
    results_with_scores = vectorstore.similarity_search_with_score(problem, k=n_results)
    opik.update_current_span(
        metadata={
            "k": n_results,
            "vector_chunks": [
                {
                    "chunk_id": doc.metadata.get(AvailableMetaData.ID, ""),
                    # Qdrant returns cosine similarity (higher = better) — use as-is.
                    "score": round(float(score), 4),
                    "source": doc.metadata.get("source", ""),
                    "storage_path": doc.metadata.get(AvailableMetaData.STORAGE_PATH, ""),
                    "document_type": doc.metadata.get(AvailableMetaData.DOCUMENT_TYPE, ""),
                }
                for doc, score in results_with_scores
            ],
        }
    )
    return results_with_scores


@opik.track(name="reranker")
def _rerank(scored_docs: dict, top_k: int):
    """Score-sort + top-N selection, traced as a span (mirrors QA's reranker).
    Logs the surviving scores and the chunk ids dropped below the cut so the
    vector-vs-graph scoring is visible inside the trace."""
    ranked = sorted(scored_docs.values(), key=lambda x: -x[1])
    top_ranked = ranked[: max(top_k, 1)]
    results = [doc for doc, _ in top_ranked]
    opik.update_current_span(
        metadata={
            "chunks_in": len(scored_docs),
            "chunks_out": len(results),
            "top_score": round(top_ranked[0][1], 4) if top_ranked else 0,
            "top_scores": [round(score, 4) for _, score in top_ranked],
            "final_chunk_ids": [
                doc.metadata.get(AvailableMetaData.ID, "") for doc in results
            ],
            "dropped_chunk_ids": [
                doc.metadata.get(AvailableMetaData.ID, "")
                for doc, _ in ranked[max(top_k, 1):]
            ],
        }
    )
    return results


RETRIEVER_PROMPT = remove_indents(
    """You're a retrieve augmented chatbot.
    You answer user's questions based on your own knowledge and information available from your search function.
    If the user is requesting non-trivial knowledge, trigger the search and formulate an answer based on the search results.
    You must give as short an answer as possible.
    When using information from the search, cite the document source like [1] and provide all references as a list in the end."""
)

BEGINNER_KNOWLEDGE_LEVEL = remove_indents(
    """Beginner
    Understanding: Has no prior knowledge or understanding of the topic;
    Experience: Has not had any formal or informal exposure to the topic."""
)

ADVANCED_KNOWLEDGE_LEVEL = remove_indents(
    """Advanced
    Understanding: Has a basic understanding of key concepts and terminology related to the topic.
    Experience: May have completed introductory courses, hands-on training, or have informal experience."""
)

EXPERT_KNOWLEDGE_LEVEL = remove_indents(
    """Expert
    Understanding: Possesses deep, comprehensive knowledge and understanding of the topic, including advanced concepts and theories.
    Experience: Has extensive practical experience, including professional work or advanced academic study."""
)


class LangchainRetrieveUserProxyAgent(RetrieveUserProxyAgent):
    def __init__(
        self,
        name="RetrieveChatAgent",  # default set to RetrieveChatAgent
        human_input_mode: Optional[str] = "ALWAYS",
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        system_message: str = RETRIEVER_PROMPT,
        retrieve_config: Optional[Dict] = None,  # config for the retrieve agent
        **kwargs,
    ):
        # Setting the value of "client" to False to make sure the default ChromeDB
        # client is not used, however, this might actually not work.
        if "client" not in retrieve_config:
            retrieve_config["client"] = False
        # Bugfix, needed because calling chromadb.Client() multiple times caused a
        # problem
        chromadb.api.client.SharedSystemClient.clear_system_cache()

        super().__init__(
            name,
            human_input_mode,
            is_termination_msg,
            retrieve_config,
            system_message=system_message,
            **kwargs,
        )
        self._embedding_model = self._retrieve_config.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2" #"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"#"sentence-transformers/all-MiniLM-L6-v2"
        )
        self._n_results = self._retrieve_config.get("n_results", 3)
        self._index_cache_path = self._retrieve_config.get("index_cache_path", None)
        self.create_index()

    def retrieve_docs(
        self, problem: str, n_results: int = None, search_string: str = ""
    ) -> Dict[str, List[List[Any]]]:
        print("RAG", "__", problem, "__", search_string)
        if n_results is None:
            n_results = self._n_results
        if not self._collection:
            self.create_index()

        # Approach 1 retrieval in Assistant mode:
        # Vector search -> graph expansion -> merge -> rerank -> top-N
        # Wrapped in a "search_question" span so the trace tree matches QA.
        from time import perf_counter as _pc
        _lat = {}; _t_start = _pc(); _t = _pc()
        with opik.start_as_current_span(name="search_question"):
            results_with_scores = _vector_retrieval(
                self.retriever.vectorstore, problem, n_results
            )
            _lat["vector"] = _pc() - _t; _t = _pc()

            vector_ids = [doc.metadata[AvailableMetaData.ID] for doc, _ in results_with_scores]
            _debug_print_ids("Vector chunk IDs", vector_ids)

            scored_docs: dict[str, tuple[Any, float]] = {}
            for doc, score in results_with_scores:
                doc_id = doc.metadata[AvailableMetaData.ID]
                similarity = float(score)
                scored_docs[doc_id] = (doc, similarity)

            vector_docs_scored = [
                (doc, float(score)) for doc, score in results_with_scores
            ]
            topic_keys = _infer_topic_keys_from_docs([doc for doc, _ in results_with_scores])
            expanded_ids: list[str] = []
            seed_to_entities: dict[str, list[str]] = {}
            entity_to_expanded: dict[str, list[str]] = {}
            graph_docs_scored: list[tuple[Any, float]] = []
            try:
                expanded_ids, seed_to_entities, entity_to_expanded = (
                    expand_chunks_via_graph_with_traversal(
                        vector_chunk_ids=vector_ids,
                        graph_root=DEFAULT_GRAPH_ARTIFACTS_PATH,
                        topic_keys=topic_keys,
                        max_expanded=ASSISTANT_GRAPH_EXPANSION_MAX,
                    )
                )
                _debug_print_ids("Graph-added chunk IDs", expanded_ids)
                for exp_id in expanded_ids:
                    if exp_id in scored_docs:
                        continue
                    # Resolve via the retriever's custom docstore (keyed by chunk
                    # id), NOT the vector-store docstore (keyed by internal
                    # ids) - mirrors the QA path. The latter always misses graph
                    # ids, which is why graph docs never reached the LLM before.
                    exp_doc = self.retriever.docstore.mget([exp_id])[0]
                    if exp_doc is not None:
                        scored_docs[exp_id] = (exp_doc, ASSISTANT_GRAPH_SCORE_WEIGHT)
                        graph_docs_scored.append((exp_doc, ASSISTANT_GRAPH_SCORE_WEIGHT))
            except Exception:
                _debug_print_ids("Graph-added chunk IDs", [])
            _lat["graph"] = _pc() - _t; _t = _pc()

            # Stash traversal so the UI can draw the same graph as the QA page.
            try:
                st.session_state["graph_traversal"] = {
                    "vector_chunks": vector_docs_scored,
                    "graph_chunks": graph_docs_scored,
                    "seed_to_entities": seed_to_entities,
                    "entity_to_expanded": entity_to_expanded,
                }
            except Exception:
                pass

            results = _rerank(scored_docs, ASSISTANT_FINAL_CONTEXT_MAX)
            _lat["rerank"] = _pc() - _t

        print(f"[LAT] retrieve_docs vector={_lat.get('vector',0):.2f}s "
              f"graph={_lat.get('graph',0):.2f}s rerank={_lat.get('rerank',0):.2f}s "
              f"total={_pc()-_t_start:.2f}s", flush=True)

        final_ids = [doc.metadata[AvailableMetaData.ID] for doc in results]
        _debug_print_ids(
            "Final chunk IDs sent to LLM context",
            final_ids,
            extra=f" | final_count={len(results)}",
        )

        output = {"ids": [], "documents": [], "original": []}
        for document in results:
            output["ids"].append(document.metadata["source"])
            output["documents"].append(document.page_content)
            output["original"].append(document.metadata.get("documents", [document]))

        new_result_format = [
            [
                [
                    {
                        "id": output["ids"][i],
                        "content": output["documents"][i],
                        "original": output["original"][i],
                    }
                ]
                for i in range(len(output["ids"]))
            ]
        ]
        output = new_result_format

        self._results = output
        print("RAG", "->", str(output))
        return output

    def create_index(self):
        self.retriever = get_retriever(
            self._embedding_model, self._docs_path, self._index_cache_path
        )
        self._collection = True
        print("Done")


def create_user_agent():
    user = UserProxyAgent(
        name="user",
        description=remove_indents(
            """Wants to learn about a topic which he provides.
            Can answer questions regarding their educational background."""
        ),
        max_consecutive_auto_reply=10,
        code_execution_config=False,
    )
    user.register_reply([Agent, None], special_chat_quit)
    return user


def create_topic_assessor():
    llm_config = get_autogen_llm_config(model_name=DEFAULT_MODEL)
    language_manager = get_language_manager()

    base_system_message = remove_indents(
        """You are a topic assessor.
        Your goal is to figure out, what a user wants to learn. This is the only thing you care about.
        You only  gather information, you do not answer the user's questions.
        You ask the user at least one clarifying question regarding the topic the user wants to learn about.
        Once you understand what the user wants to learn, you describe it in a detailed summary. This summary begins with the word 'SUMMARY:'."""
    )

    system_message = get_localized_system_message(base_system_message, language_manager)

    topic_assessor = ConversableAgent(
        name="topic_assessor",
        description="Assesses what the user wants to learn.",
        system_message=system_message,
        human_input_mode="NEVER",
        llm_config=llm_config,
        max_consecutive_auto_reply=4,
        code_execution_config=False,
    )
    return topic_assessor


def create_knowledge_assessor():
    llm_config = get_autogen_llm_config(model_name=DEFAULT_MODEL)
    language_manager = get_language_manager()

    base_system_message = remove_indents(
        f"""You are a knowledge assessor.

        The three levels of knowledge are:

        {BEGINNER_KNOWLEDGE_LEVEL}

        {ADVANCED_KNOWLEDGE_LEVEL}

        {EXPERT_KNOWLEDGE_LEVEL}

        You ask questions, until you are confident that you have enough information to assess the user's knowledge level to be Beginner, Advanced or Expert.
        You only ask questions, you don't answer questions.
        Once you are sure about the user's knowledge level, you report the knowledge level.
        Your response options is one of the following three:

        1) "SUMMARY: {BEGINNER_KNOWLEDGE_LEVEL}",

        2) "SUMMARY: {ADVANCED_KNOWLEDGE_LEVEL}",

        3) "SUMMARY: {EXPERT_KNOWLEDGE_LEVEL}"."""
    )

    system_message = get_localized_system_message(base_system_message, language_manager)

    knowledge_assessor = ConversableAgent(
        name="knowledge_assessor",
        description="Assesses a user's knowlege level with respect to a given topic.",
        system_message=system_message,
        human_input_mode="NEVER",
        llm_config=llm_config,
        max_consecutive_auto_reply=8,
        code_execution_config=False,
    )
    return knowledge_assessor


def create_teaching_material_assessor():
    llm_config = get_autogen_llm_config(model_name=DEFAULT_MODEL)
    language_manager = get_language_manager()

    base_system_message = remove_indents(
        """You are a teaching material assessor.
        Your goal is to suggest suitable teaching material for a user. For this you have a summary of the topic the user is interested in and a summary of the users level of knowledge.
        Use the function get_information_on_topic to browse the available teaching materials and information sources. Always include the knowledge level of the user in the search query.
        Suggest a concrete document or video to the user, including a file name. Make sure to report the training it belongs to and the level of the training.
        Most importantly, be truthful and don't invent anything. It is OK to admit you don't have any teaching material for the topic at the requested level."""
    )

    system_message = get_localized_system_message(base_system_message, language_manager)

    teaching_material_assessor = ConversableAgent(
        name="teaching_material_assessor",
        description=(
            "Curates a curriculum of learning materials for the user based on the "
            "users interests and knowledge level and answer their questions directly "
            "if possible."
        ),
        system_message=system_message,
        human_input_mode="NEVER",
        llm_config=llm_config,
        max_consecutive_auto_reply=10,
        code_execution_config=False,
    )

    return teaching_material_assessor


def create_function_executor():
    llm_config = get_autogen_llm_config(model_name=DEFAULT_MODEL)
    function_executor = ConversableAgent(
        name="function_executor",
        description=remove_indents(
            """Executes all functions.
            This agent should only be prompted to execute the function get_information_on_topic to gather information on a topic.
            It should not be asked questions. Questions are asked only to the user."""
        ),
        system_message="You execute functions that are registered with you when queried by another agent to do so.",
        llm_config=llm_config,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=8,
        code_execution_config=CODE_EXECUTION_CONFIG,
    )
    return function_executor