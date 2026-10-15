from __future__ import annotations

import os
import re
from pathlib import Path
import streamlit as st

import logging
# Opik tracing for the serving Q&A pipeline. Must be set BEFORE `import opik`.
# "true" = @opik.track decorators/spans become no-ops (removes per-request span
# overhead); "false" = tracing enabled (sends traces to the Opik server).
os.environ["OPIK_TRACK_DISABLE"] = "false"
import opik
from opik.integrations.langchain import OpikTracer
import json

os.environ.setdefault("OPIK_URL_OVERRIDE", "http://localhost:5173/api")
os.environ.setdefault("OPIK_PROJECT_NAME", "dc-available-rag")
os.environ.setdefault("OPIK_WORKSPACE", "default")

# Patch pydantic v1 compat layer to support model_dump (opik 2.x + langchain 0.1.x)
try:
    import pydantic.v1 as _pv1
    if not hasattr(_pv1.BaseModel, "model_dump"):
        _pv1.BaseModel.model_dump = lambda self, **kw: self.dict(**kw)
except ImportError:
    pass

from langchain import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from dc_available.common.helpers import join_paths, topic_from_storage_path, path_display_name
from dc_available.common.language_manager import get_language_manager
from dc_available.config.llm_config import get_langchain_llm, DEFAULT_SERVE_MODEL
from dc_available import DATA_DIR
from dc_available.RAG.data import AvailableMetaData
from dc_available.RAG.retrieval import AvailableRetriever
from dc_available.RAG.retrieval_helpers import (
    get_embeddings,
    get_retriever,
    prepare_path,
)
from dc_available.config.data_config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INGESTION_ARTIFACTS_PATH,
    DEFAULT_INDEX_ARTIFACTS_PATH,
    DEFAULT_DATA_PATH,
    MAIN_FOLDER,
)
from dc_available.config.data_config import DEFAULT_GRAPH_ARTIFACTS_PATH
from dc_available.config import data_config
from dc_available.graph.retrieval.entity_hop_retriever import (
    _module_key_from_document,
    entities_for_chunks,
    expand_chunks_via_graph_with_traversal,
    retrieve as graph_retrieve,
)
from dc_available.graph.retrieval.ppr_retriever import ppr_retrieve
from dc_available.RAG import fusion as fusion_mod
from dc_available.RAG.reranker import rerank as _cross_encoder_rerank
from dc_available.trust_score.final_trust import trust_for
from dc_available.frontend.streamlit_helpers import (
    display_document_in_streamlit,
    get_trust_score_status,
)

embeddings: Embeddings
llm: LLM
vectorstore: Qdrant
retriever: AvailableRetriever

NUM_PAGES_SEARCHED = 5
MAX_PAGES_SHOWN = 3 * NUM_PAGES_SEARCHED
QA_RETRIEVAL_DEBUG_DEFAULT: bool = True

# Azure OpenAI pricing: (input USD per 1K tokens, output USD per 1K tokens)
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o":       (0.0025, 0.010),
    "gpt-4o-mini":  (0.00015, 0.0006),
    "gpt-4-turbo":  (0.010, 0.030),
    "gpt-4":        (0.030, 0.060),
    "gpt-35-turbo": (0.0005, 0.0015),
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    input_price, output_price = _MODEL_PRICING.get(model, (0.0, 0.0))
    return round((prompt_tokens * input_price + completion_tokens * output_price) / 1000, 6)


_JUDGE_LLM = None

_JUDGE_PROMPT = """\
You are evaluating a RAG (Retrieval-Augmented Generation) system response.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

ANSWER:
{answer}

Score the response on these 3 metrics and return ONLY valid JSON, no extra text.

1. hallucination (0.0 to 1.0)
   0.0 = answer is fully supported by the context
   0.5 = answer is partially supported, some claims are unverifiable
   1.0 = answer contains claims clearly not found in the context

2. answer_relevance (0.0 to 1.0)
   0.0 = completely off-topic
   1.0 = perfectly addresses the question

3. context_precision (0.0 to 1.0)
   0.0 = retrieved context is irrelevant to the question
   1.0 = retrieved context is perfectly relevant

Return exactly this JSON format:
{{
  "hallucination": <score>,
  "hallucination_reason": "<one sentence>",
  "answer_relevance": <score>,
  "answer_relevance_reason": "<one sentence>",
  "context_precision": <score>,
  "context_precision_reason": "<one sentence>"
}}"""


def _get_judge_llm():
    global _JUDGE_LLM
    if _JUDGE_LLM is None:
        _JUDGE_LLM = get_langchain_llm(model_name="gpt-4o-mini")
    return _JUDGE_LLM


def _run_llm_judge(question: str, answer: str, context_docs: list) -> None:
    try:
        from langchain.schema import HumanMessage
        context_text = "\n".join(
            f"[{i+1}] {doc.page_content}"
            for i, doc in enumerate(context_docs[:5])
            if doc.page_content
        )
        prompt = _JUDGE_PROMPT.format(
            question=question,
            context=context_text,
            answer=answer,
        )
        response = _get_judge_llm().invoke([HumanMessage(content=prompt)])
        raw_json = response.content.strip()

        print(f"\n[LLM Judge] Raw response:\n{raw_json}\n")

        scores = json.loads(raw_json)
        opik.update_current_trace(feedback_scores=[
            {"name": "hallucination",     "value": float(scores["hallucination"]),     "reason": scores.get("hallucination_reason", "")},
            {"name": "answer_relevance",  "value": float(scores["answer_relevance"]),  "reason": scores.get("answer_relevance_reason", "")},
            {"name": "context_precision", "value": float(scores["context_precision"]), "reason": scores.get("context_precision_reason", "")},
        ])
    except Exception as exc:
        logging.getLogger(__name__).warning("LLM judge scoring failed: %s", exc)


def _is_retrieval_debug_enabled() -> bool:
    if "rag_debug_retrieval" in st.session_state:
        return bool(st.session_state.get("rag_debug_retrieval"))

    env_enabled = os.getenv("RAG_DEBUG_RETRIEVAL", "false").strip().lower() in {"1", "true", "yes", "on"}
    if env_enabled:
        return True

    return QA_RETRIEVAL_DEBUG_DEFAULT


def _topic_subfolder_from_path(storage_path: str) -> tuple[str, str]:
    """Best-effort (topic, subfolder) from a storage path, for debug/graph display.

    Source-agnostic: anchors on the active source's main data folder (MAIN_FOLDER,
    e.g. 'local_sharepoint') or an extracted-media folder ('video'/'audio'/'images'),
    then reads the top-level topic and the folder holding the file. Replaces the old
    logic that was hardcoded to 'DC_data'.
    """
    parts = [p for p in storage_path.replace("\\", "/").split("/") if p]
    anchors = (MAIN_FOLDER, "video", "audio", "images", "pptx", "pdf")
    idx = None
    for i, p in enumerate(parts):
        if p in anchors:
            idx = i  # last match wins (handles nested media roots)
    if idx is None or idx + 1 >= len(parts):
        return "N/A", "N/A"
    after = parts[idx + 1:]                 # [TOPIC, (subtopic...), level, training, file]
    topic = after[0] if after else "N/A"
    holding = after[:-1] if len(after) > 1 else after   # drop the file name
    subfolder = holding[-1] if len(holding) >= 2 else "N/A"
    return topic, subfolder


def _extract_chunk_meta(doc) -> dict:
    meta = doc.metadata if doc else {}
    chunk_id = str(meta.get(AvailableMetaData.ID, "N/A"))
    level = str(meta.get(AvailableMetaData.LEVEL, "N/A"))
    doc_name = str(meta.get(AvailableMetaData.PART_OF_DOCUMENT, "N/A"))
    doc_type = str(meta.get(AvailableMetaData.DOCUMENT_TYPE, "N/A"))
    storage_path = str(meta.get(AvailableMetaData.STORAGE_PATH, ""))
    topic, subfolder = _topic_subfolder_from_path(storage_path)
    normalized = storage_path.replace("\\", "/")
    subdocument = normalized.split("/")[-1] if normalized else "N/A"
    return {
        "id": chunk_id, "topic": topic, "level": level,
        "doc": doc_name, "subfolder": subfolder,
        "subdocument": subdocument, "doc_type": doc_type,
    }


def _debug_print_chunks(stage: str, docs_and_scores: list) -> None:
    if not _is_retrieval_debug_enabled():
        return
    print(f"[RetrievalDebug] {stage} ({len(docs_and_scores)}):")
    for i, (doc, score) in enumerate(docs_and_scores, 1):
        m = _extract_chunk_meta(doc)
        short_id = m["id"][:8]
        doc_display = m["doc"][:35]
        # After trust is folded in, break the final score into rerank | trust | final.
        # Earlier stages (vector/graph/pool) have no trust yet -> plain score.
        trust = doc.metadata.get("final_trust")
        if trust is not None:
            rr = doc.metadata.get("rerank_score", score)
            score_str = f"rerank={float(rr):.3f} | trust={float(trust):.3f} | final={score:.3f}"
        else:
            score_str = f"score={score:.3f}"
        print(f"  #{i:2d} id={short_id} | topic={m['topic']:<22} | level={m['level']:<10} | doc={doc_display:<35} | {score_str}")


def _debug_print_documents(stage: str, recommended: list) -> None:
    """Print the fused DOCUMENT track (Stage D recommendation list)."""
    if not _is_retrieval_debug_enabled():
        return
    print(f"[RetrievalDebug] {stage} ({len(recommended)}):")
    for i, rec in enumerate(recommended, 1):
        legs = f"{'V' if rec.get('in_vector') else '-'}{'G' if rec.get('in_graph') else '-'}"
        doc_display = str(rec.get("document", "N/A"))[:45]
        trust = rec.get("trust")
        trust_str = f" | trust={float(trust):.3f}" if trust is not None else ""
        print(f"  #{i:2d} [{legs}] final={rec.get('score', 0.0):.4f}{trust_str} | topic={str(rec.get('topic')):<14} | doc={doc_display}")


def _opik_chunk_briefs(docs_and_scores) -> list[dict]:
    """Compact chunk list for Opik span I/O (id · doc · type · score) — no page_content."""
    briefs = []
    for doc, score in docs_and_scores:
        m = _extract_chunk_meta(doc)
        brief = {"id": m["id"][:8], "doc": m["doc"], "type": m["doc_type"], "score": round(float(score), 4)}
        # When trust has been applied, expose rerank | trust | final alongside score.
        trust = doc.metadata.get("final_trust")
        if trust is not None:
            brief["rerank_score"] = doc.metadata.get("rerank_score")
            brief["trust"] = trust
            brief["final_score"] = doc.metadata.get("final_score", round(float(score), 4))
        briefs.append(brief)
    return briefs


def _opik_pool_briefs(cand_docs, cand_items) -> list[dict]:
    """Compact union-pool list for Opik (id · doc · leg provenance vector/graph/both)."""
    briefs = []
    for doc, it in zip(cand_docs, cand_items):
        m = _extract_chunk_meta(doc)
        if it.vector_rank is not None and it.graph_rank is not None:
            leg = "both"
        elif it.vector_rank is not None:
            leg = "vector"
        else:
            leg = "graph"
        briefs.append({"id": m["id"][:8], "doc": m["doc"], "leg": leg})
    return briefs


def _graph_traversal_payload(graph_result, graph_k: int) -> tuple[dict, dict]:
    """Compact Stage-C summary for the graph_traversal Opik span -> (output, metadata).

    Always includes matched entities, the walked chunks, and the MaxP documents plus
    list_weight/concentration. The full 7-step Stage-C trace is attached only when the
    graph leg ran in debug mode (keeps eval fast; rich detail on demand).
    """
    if graph_result is None:
        return {"abstained": True, "reason": "graph leg unavailable"}, {"abstained": True}
    kept = graph_result.matched_entities or []
    entities = [
        {
            "entity": f"{e.get('label')} [{e.get('entity_type')}]",
            "topic": e.get("topic_key"),
            "fused": round(float(e.get("fused_score") or 0.0), 3),
            "idf": round(float(e.get("idf") or 0.0), 2),
            "cf": e.get("chunk_frequency"),
        }
        for e in kept[:10]
    ]
    chunks = [
        {"id": str(cid)[:8], "score": round(float(sc), 4)}
        for cid, sc in graph_result.chunks[:graph_k]
    ]
    documents = [
        {
            "document": d.get("document"), "topic": d.get("topic"),
            "score": round(float(d.get("score") or 0.0), 4), "n_chunks": d.get("n_chunks"),
        }
        for d in (graph_result.documents or [])[:5]
    ]
    output = {
        "abstained": bool(graph_result.abstained),
        "matched_entities": entities,
        "chunks": chunks,
        "documents": documents,
    }
    if graph_result.trace:
        output["stage_c_trace"] = graph_result.trace
    metadata = {
        "kept_entities": len(kept),
        "graph_chunks_total": len(graph_result.chunks),
        "list_weight": round(float(graph_result.list_weight), 4),
        "concentration": round(float(graph_result.concentration), 4),
        "fallback_mode": bool(graph_result.fallback_mode),
        "abstained": bool(graph_result.abstained),
    }
    return output, metadata


# Predefined prompt examples
PROMPT_EXAMPLES = [
    "I need to connect my Heller Machines via UMC to our SAP DM MES System within DCMLP.",
    "My customer reports problems with his hydraulic proportional valves. How do I identify the root cause?",
    "I need to sell hydraulic systems to my customers and have no knowledge about connected hydraulics.",
    "How do I troubleshoot communication issues between controllers?",
    "What are the best practices for industrial automation setup?",
    "Explain the maintenance schedule for hydraulic systems.",
    "How to configure safety systems in industrial environments?"
]

EXAMPLE_PROMPT = PromptTemplate(
    template="""Content: {page_content}
    Source: {source}

    Note: Respond in the same language as the user's question — answer in German if the question is in German, otherwise answer in English.""",
    input_variables=["page_content", "source"],
)


# Qdrant has no in-memory docstore. When we need to iterate/look up
# EVERY child chunk (not just search results), we scroll the whole collection
# once and cache it, keyed by the chunk's metadata id. The collection is static
# at serve time, so a one-time cache is safe.
_child_docs_cache: dict[str, Document] | None = None


def _get_child_docs_by_id() -> dict[str, Document]:
    """Return all child-chunk Documents from Qdrant, keyed by metadata id (cached)."""
    global _child_docs_cache
    if _child_docs_cache is None:
        cache: dict[str, Document] = {}
        offset = None
        while True:
            points, offset = vectorstore.client.scroll(
                collection_name=vectorstore.collection_name,
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            for point in points:
                payload = point.payload or {}
                meta = payload.get("metadata", {})
                cid = meta.get(AvailableMetaData.ID)
                if cid:
                    cache[cid] = Document(
                        page_content=payload.get("page_content", ""), metadata=meta
                    )
            if offset is None:
                break
        _child_docs_cache = cache
    return _child_docs_cache


def _apply_document_trust(scored):
    """Fold per-document trust into reranked scores: score * (1 + β·final_trust).

    Query-independent — it only reorders the already-relevant chunks. Records the
    resolved trust on doc.metadata["final_trust"] for telemetry/debug. No-op when
    TRUST_ENABLED is False. Does not truncate; the caller sorts + keeps top-k.
    """
    if not getattr(data_config, "TRUST_ENABLED", False):
        return scored
    beta = getattr(data_config, "TRUST_BETA", 0.5)
    out = []
    for doc, score in scored:
        t = trust_for(
            doc.metadata.get(AvailableMetaData.PART_OF_DOCUMENT),
            doc.metadata.get(AvailableMetaData.STORAGE_PATH),
        )
        final = float(score) * (1.0 + beta * float(t))
        # Stash all three components so console/Opik debug can show them separately:
        # rerank (pre-trust) -> trust -> final. Without this the pre-trust score is lost.
        doc.metadata["rerank_score"] = round(float(score), 4)
        doc.metadata["final_trust"] = round(float(t), 4)
        doc.metadata["final_score"] = round(final, 4)
        out.append((doc, final))
    return out


@opik.track(name="retrieval")
def search_question(
        question: str,
        use_audio_transcript: bool = False,
        use_image_understanding: bool = False,
        use_consolidated_understanding: bool = False,
        add_all_modalities: bool = False,
        return_outputs=True,
        legs: str | None = None,
):
    """Retrieve the chunks sent to the LLM (Stage D fusion path).

    Independent vector + query-seeded graph legs are pooled (union), reranked by a
    cross-encoder, and the top FUSION_OUTPUT_K chunks go to the LLM.

    ``legs`` ("both" | "vector" | "graph") overrides the global FUSION_LEGS for this
    request, so the UI Fast/Thinking choice is applied per query without mutating
    process-wide config.
    """
    return _search_question_fusion(question, add_all_modalities=add_all_modalities, legs=legs)


def _apply_all_modalities(documents):
    """Expand each video chunk to its sibling image/audio/video modality chunks.

    Extracted verbatim from the legacy body so both retrieval modes share it.
    """
    new_documents = []
    new_ids = []
    for doc in documents:
        if AvailableMetaData.VIDEO_PATH not in doc.metadata:
            new_ids.append(doc.metadata[AvailableMetaData.ID])
            new_documents.append(doc)
            continue

        if doc.metadata[AvailableMetaData.ID] in new_ids:
            continue

        different_modality_docs = []

        for o_doc in _get_child_docs_by_id().values():
            if AvailableMetaData.IMAGE_PATH in o_doc.metadata:
                if (
                        o_doc.metadata[AvailableMetaData.IMAGE_PATH]
                        == doc.metadata[AvailableMetaData.IMAGE_PATH]
                ):
                    # assert o_doc.metadata[AvailableMetaData.START_TIME] == doc.metadata[AvailableMetaData.START_TIME]
                    # assert o_doc.metadata[AvailableMetaData.END_TIME] == doc.metadata[AvailableMetaData.END_TIME]

                    if o_doc.metadata[AvailableMetaData.DOCUMENT_TYPE] in [
                        "audio",
                        "video",
                        "image",
                    ]:
                        is_ppt = Path(
                            o_doc.metadata[AvailableMetaData.STORAGE_PATH]
                        ).name.startswith("page_")
                        if not is_ppt:
                            different_modality_docs.append(o_doc)

        for doc in different_modality_docs:
            new_ids.append(doc.metadata[AvailableMetaData.ID])
            new_documents.append(doc)
    return new_documents


def _search_question_fusion(question: str, add_all_modalities: bool = False, legs: str | None = None):
    """Stage D retrieval: two independent legs pooled, then reranked.

    Leg 1 (vector): hybrid dense+sparse chunk search — its top FUSION_VECTOR_K chunks.
    Leg 2 (graph):  query-seeded graph_retriever.retrieve — independent of the vector
                    leg — its top FUSION_GRAPH_K chunks. On abstain/failure it yields
                    nothing, so the pool collapses cleanly to vector-only.

    FUSION_STRATEGY builds the candidate pool:
      "union" (default) — de-duplicated union of both legs' chunks; neither leg can
                          bury the other (see fusion.union_fuse).
      "rrf"             — legacy weighted RRF, kept for ablation/comparison.
    The cross-encoder reranker (Option 1) reorders the pool; the top FUSION_OUTPUT_K
    chunks go to the LLM. The recommended-documents list is the unique documents
    behind those final chunks (no separate document fusion).
    """
    import logging
    log = logging.getLogger(__name__)

    strategy = getattr(data_config, "FUSION_STRATEGY", "union")
    # Per-request leg choice (UI "Fast"/"Thinking") wins; fall back to the global
    # FUSION_LEGS so the eval harness, which sets the global, is unaffected. Passing
    # the choice as an argument (instead of mutating the global) is concurrency-safe.
    legs = legs or getattr(data_config, "FUSION_LEGS", "both")  # "both" | "vector" | "graph"

    # --- Step 2: vector leg (hybrid dense + sparse), top FUSION_VECTOR_K chunks ---
    from time import perf_counter as _pc
    _tv = _pc()
    with opik.start_as_current_span(name="vector_retrieval"):
        vector_hits = (
            []
            if legs == "graph"
            else vectorstore.similarity_search_with_score(question, k=data_config.FUSION_VECTOR_K)
        )
        vector_ids = [doc.metadata[AvailableMetaData.ID] for doc, _ in vector_hits]
        doc_by_id = {doc.metadata[AvailableMetaData.ID]: doc for doc, _ in vector_hits}
        _debug_print_chunks("Vector chunks", [(doc, float(s)) for doc, s in vector_hits])
        try:
            opik.update_current_span(
                input={"question": question, "k": data_config.FUSION_VECTOR_K},
                output={"chunks": _opik_chunk_briefs([(d, s) for d, s in vector_hits])},
            )
        except Exception:
            pass

    _lat_vec = _pc() - _tv
    # --- Step 3: graph traversal (Stage C, query-seeded). Never fatal. ---
    _tg = _pc()
    with opik.start_as_current_span(name="graph_traversal"):
        graph_result = None
        if legs != "vector":
            try:
                if getattr(data_config, "GRAPH_RETRIEVAL_MODE", "entity_1hop") == "ppr":
                    # PPR leg: seed with the vector leg's chunks (dense passage seeds).
                    # For the graph-only ablation (legs=="graph") vector_hits is empty,
                    # so the leg runs purely entity-seeded — independent of the vector leg.
                    passage_seeds = [
                        (doc.metadata[AvailableMetaData.ID], float(score))
                        for doc, score in vector_hits
                    ]
                    graph_result = ppr_retrieve(
                        question, DEFAULT_GRAPH_ARTIFACTS_PATH,
                        vector_hits=passage_seeds, debug=_is_retrieval_debug_enabled(),
                    )
                else:
                    graph_result = graph_retrieve(
                        question, DEFAULT_GRAPH_ARTIFACTS_PATH, debug=_is_retrieval_debug_enabled()
                    )
            except Exception as exc:
                log.warning("Fusion graph leg failed: %s", exc)

        if graph_result is not None and not graph_result.abstained:
            graph_pairs = graph_result.chunks[: data_config.FUSION_GRAPH_K]
        else:
            graph_pairs = []
        graph_ids = [cid for cid, _ in graph_pairs]
        graph_scores = {cid: sc for cid, sc in graph_pairs}

        # Hydrate graph-only chunk ids once (docstore); reused by the pool below.
        missing = [cid for cid in graph_ids if cid not in doc_by_id]
        if missing:
            for mid, mdoc in zip(missing, retriever.docstore.mget(missing)):
                if mdoc is not None:
                    doc_by_id[mid] = mdoc
        _debug_print_chunks(
            "Graph chunks",
            [(doc_by_id[cid], float(graph_scores[cid])) for cid in graph_ids if cid in doc_by_id],
        )
        try:
            g_out, g_meta = _graph_traversal_payload(graph_result, data_config.FUSION_GRAPH_K)
            opik.update_current_span(
                input={"question": question, "graph_k": data_config.FUSION_GRAPH_K},
                output=g_out, metadata=g_meta,
            )
        except Exception:
            pass

    _lat_graph = _pc() - _tg
    overlap = len(set(vector_ids) & set(graph_ids))

    # --- Step 4: fusion (pool -> rerank -> documents), each a child span ---
    with opik.start_as_current_span(name="fusion"):
        # Pool: union (default) or legacy RRF (ablation).
        with opik.start_as_current_span(name="union"):
            if strategy == "rrf":
                graph_weight = fusion_mod.graph_leg_weight(
                    graph_result.list_weight if graph_result is not None else 0.0,
                    graph_result.abstained if graph_result is not None else True,
                )
                pool_items = fusion_mod.rrf_fuse(vector_ids, graph_ids, graph_weight=graph_weight)[
                    : data_config.FUSION_CANDIDATE_K
                ]
                pool_label = "Fused candidates (RRF)"
            else:
                graph_weight = 0.0  # unused in union; reported as 0 in the summary/span
                pool_items = fusion_mod.union_fuse(vector_ids, graph_ids)
                pool_label = "Union pool"

            # Hydrate the pool -> Documents (all ids already resolved in doc_by_id).
            cand_docs, cand_items = [], []
            for it in pool_items:
                doc = doc_by_id.get(it.key)
                if doc is not None:
                    cand_docs.append(doc)
                    cand_items.append(it)
            _debug_print_chunks(pool_label, [(d, it.score) for d, it in zip(cand_docs, cand_items)])
            try:
                opik.update_current_span(
                    input={
                        "strategy": strategy, "vector_chunks": len(vector_ids),
                        "graph_chunks": len(graph_ids), "overlap": overlap,
                    },
                    output={"pool": _opik_pool_briefs(cand_docs, cand_items)},
                )
            except Exception:
                pass

        # Rerank (Option 1), then fold in document trust, then keep top FUSION_OUTPUT_K.
        _tr = _pc()
        with opik.start_as_current_span(name="reranker"):
            if data_config.RERANKER_ENABLED and cand_docs:
                # Score the FULL pool so trust can still reorder which chunks make the cut.
                order = _cross_encoder_rerank(
                    question, [d.page_content for d in cand_docs], top_k=len(cand_docs)
                )
                reranked = [(cand_docs[idx], score) for idx, score in order]
            else:
                reranked = [(d, it.score) for d, it in zip(cand_docs, cand_items)]
            # final = rerank_score * (1 + β·final_trust); no-op if TRUST_ENABLED is False.
            reranked = _apply_document_trust(reranked)
            reranked.sort(key=lambda ds: -ds[1])
            top_scored = reranked[: data_config.FUSION_OUTPUT_K]
            _lat_rerank = _pc() - _tr
            print(f"[LAT] retrieval breakdown  vector={_lat_vec:.2f}s  "
                  f"graph={_lat_graph:.2f}s  rerank={_lat_rerank:.2f}s", flush=True)
            documents = [doc for doc, _ in top_scored]
            _debug_print_chunks("Final chunks sent to LLM", top_scored)
            try:
                opik.update_current_span(
                    input={
                        "pool_size": len(cand_docs), "top_k": data_config.FUSION_OUTPUT_K,
                        "model": data_config.RERANKER_MODEL if data_config.RERANKER_ENABLED else None,
                    },
                    output={"chunks": _opik_chunk_briefs(top_scored)},
                )
            except Exception:
                pass

        # Recommended documents = unique documents behind the final chunks.
        with opik.start_as_current_span(name="documents"):
            item_by_id = {it.key: it for it in cand_items}
            recommended = _recommend_documents_from_chunks(top_scored, item_by_id)
            _debug_print_documents("Recommended documents", recommended)
            try:
                st.session_state["recommended_documents"] = recommended
            except Exception:
                pass
            try:
                opik.update_current_span(output={"recommended": recommended})
            except Exception:
                pass

        # Fusion parent I/O + roll-up metadata.
        try:
            opik.update_current_span(
                input={
                    "strategy": strategy, "vector_chunks": len(vector_ids),
                    "graph_chunks": len(graph_ids), "overlap": overlap,
                },
                output={"pool_size": len(cand_docs), "final_chunks": len(top_scored)},
                metadata={
                    "mode": "fusion",
                    "strategy": strategy,
                    "vector_k": data_config.FUSION_VECTOR_K,
                    "graph_k": data_config.FUSION_GRAPH_K,
                    "output_k": data_config.FUSION_OUTPUT_K,
                    "reranker_enabled": data_config.RERANKER_ENABLED,
                    "reranker_model": data_config.RERANKER_MODEL if data_config.RERANKER_ENABLED else None,
                    "leg_overlap_chunks": overlap,
                    "graph_abstained": bool(graph_result.abstained) if graph_result is not None else True,
                    "graph_list_weight": round(float(graph_result.list_weight), 4) if graph_result is not None else 0.0,
                    "graph_rrf_weight": round(float(graph_weight), 4),
                    "pool_size": len(cand_docs),
                    "final_chunks": len(top_scored),
                    "final_chunk_ids": [d.metadata.get(AvailableMetaData.ID) for d, _ in top_scored],
                    "final_scores": [round(float(s), 4) for _, s in top_scored],
                },
            )
        except Exception as exc:
            log.warning("Fusion opik metadata failed: %s", exc)

    # --- Retrieval graph for the QA UI: Document -> Chunk[leg] -> entities ---
    try:
        st.session_state["fusion_graph"] = _build_fusion_graph_data(
            top_scored, set(vector_ids), set(graph_ids)
        )
    except Exception as exc:
        log.warning("Fusion graph viz failed: %s", exc)
        try:
            st.session_state["fusion_graph"] = None
        except Exception:
            pass

    # --- One-line console summary ---
    if _is_retrieval_debug_enabled():
        print(
            f"[RetrievalDebug] Fusion summary: strategy={strategy} vector={len(vector_ids)} "
            f"graph={len(graph_ids)} overlap={overlap} pool={len(cand_docs)} final={len(top_scored)} "
            f"abstained={graph_result.abstained if graph_result is not None else True}"
        )

    if add_all_modalities:
        documents = _apply_all_modalities(documents)
    return documents


def _recommend_documents_from_chunks(scored_docs, item_by_id):
    """Recommended-documents list = the unique documents behind the final chunks.

    Since the union keeps both legs' chunks, the documents worth recommending are
    simply the ones the answer chunks came from — no separate document fusion. A
    document inherits its best (first-seen, i.e. highest-reranked) chunk's score;
    provenance flags come from that chunk's pool item. Capped at FUSION_DOC_TOPN.
    Returns [{document, topic, score, in_vector, in_graph}] descending.
    """
    recommended = []
    seen: set[str] = set()
    for doc, score in scored_docs:
        meta = doc.metadata
        name = meta.get(AvailableMetaData.PART_OF_DOCUMENT)
        if not name or name in seen:
            continue
        seen.add(name)
        it = item_by_id.get(meta.get(AvailableMetaData.ID))
        topic, _ = _topic_subfolder_from_path(str(meta.get(AvailableMetaData.STORAGE_PATH, "")))
        recommended.append(
            {
                "document": name,
                "topic": topic,
                "score": float(score),               # kept for UI/session_state compatibility
                "final_score": round(float(score), 4),
                "trust": meta.get("final_trust"),
                "in_vector": bool(it and it.vector_rank is not None),
                "in_graph": bool(it and it.graph_rank is not None),
            }
        )
        if len(recommended) >= data_config.FUSION_DOC_TOPN:
            break
    return recommended


class QAAnswer(BaseModel):
    """Structured QA output: the prose answer plus the chunk ids actually used."""

    answer: str = Field(
        description="Answer to the question, in the same language as the question (German if "
        "the question is in German, otherwise English), using "
        "only the provided context. Prose only — do NOT include ids or a 'sources' line here."
    )
    sources: list[str] = Field(
        default_factory=list,
        description="REQUIRED. The id values (the value after 'id:') of every context chunk "
        "you used to write the answer, as bare ids without brackets. Must not be empty when "
        "the answer draws on the context.",
    )


_QA_SYSTEM_INSTRUCTIONS = (
    "You answer questions strictly from the provided context. Do not use outside "
    "knowledge; if the context is insufficient, say so. Answer in the same language as "
    "the question: if the question is in German, answer in German; otherwise answer in English. "
    "Put the prose in 'answer' (no id list there). In 'sources' you MUST list "
    "the bare id of every chunk whose content you used — never leave 'sources' empty when "
    "the answer is based on the context. "
    "Each chunk may carry a 'trust' value between 0 and 1 indicating how reliable its "
    "source document is. When two or more chunks contain the same or conflicting "
    "information, rely on the chunk with the higher trust value."
)


def _generate_structured_answer(question, documents, callbacks=None):
    """LLM answer as structured JSON (prose + cited chunk ids) via function calling.

    Each context chunk is tagged with its chunk id, and the model returns the ids it
    used — so attribution is exact, with no text parsing. Raises on any failure so the
    caller can fall back to the legacy text chain.
    """
    def _ctx_tag(d):
        cid = d.metadata.get(AvailableMetaData.ID, "")
        t = d.metadata.get("final_trust")   # set by _apply_document_trust when TRUST_ENABLED
        trust = f" | trust: {float(t):.2f}" if t is not None else ""
        return f"[id: {cid}{trust}]"

    context = "\n\n".join(f"{_ctx_tag(d)}\n{d.page_content}" for d in documents)
    prompt = f"{_QA_SYSTEM_INSTRUCTIONS}\n\nContext:\n{context}\n\nQuestion: {question}"
    structured_llm = llm.with_structured_output(QAAnswer)
    result = structured_llm.invoke(
        prompt, config={"callbacks": callbacks} if callbacks else None
    )
    if isinstance(result, QAAnswer):
        return (result.answer or "").strip(), [str(s) for s in (result.sources or [])]
    if isinstance(result, dict):
        return str(result.get("answer", "")).strip(), [str(s) for s in (result.get("sources") or [])]
    raise ValueError(f"Unexpected structured output type: {type(result)!r}")


@opik.track(name="rag_pipeline")
def ask_question(
        question: str,
        history: list[tuple[str, str]],
        use_audio_transcript: bool = False,
        use_image_understanding: bool = False,
        use_consolidated_understanding: bool = False,
        add_all_modalities: bool = False,
        content_dir: Path = DEFAULT_DATA_PATH,
        main_folder: str = MAIN_FOLDER,
        thread_id: str | None = None,
        retrieval_mode: str = "thinking",
):
    global llm
    llm = get_langchain_llm(model_name=DEFAULT_SERVE_MODEL)
    print(f"[QA] LLM deployment: {DEFAULT_SERVE_MODEL}")

    # UI mode -> retrieval legs. "fast" = vector-only (skips the graph leg);
    # "thinking" = vector + graph (the default hybrid path).
    legs = "vector" if retrieval_mode == "fast" else "both"
    print(f"[QA] retrieval_mode={retrieval_mode} -> legs={legs}")

    if thread_id:
        opik.update_current_trace(thread_id=thread_id)

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", input_key="question", output_key="text", k=5
    )
    for user_question, model_answer in history[:-1]:
        memory.save_context({"question": user_question}, {"text": model_answer})

    from time import perf_counter as _pc
    _t = _pc()
    relevant_documents = search_question(
        question,
        use_audio_transcript=use_audio_transcript,
        use_image_understanding=use_image_understanding,
        use_consolidated_understanding=use_consolidated_understanding,
        add_all_modalities=add_all_modalities,
        return_outputs=False,
        legs=legs,
    )
    _lat_retrieval = _pc() - _t
    relevant_documents_dict = {
        doc.metadata[AvailableMetaData.ID]: doc for doc in relevant_documents
    }

    # Structured answer (prose + cited chunk ids via function calling). Falls back to
    # the legacy text chain if the deployment can't do structured output.
    cited_ids: list[str] = []
    structured_ok = False
    _t_ans = _pc()
    with get_openai_callback() as cb:
        try:
            answer, cited_ids = _generate_structured_answer(
                question, relevant_documents
            )
            structured_ok = True
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Structured QA failed (%s); falling back to the text chain.", exc
            )
            chain = load_qa_with_sources_chain(
                llm, chain_type="stuff", output_key="text", verbose=True,
                document_prompt=EXAMPLE_PROMPT, memory=memory,
            )
            response = chain(
                {"input_documents": relevant_documents, "question": question},
                return_only_outputs=True,
            )
            answer = response["text"].strip()

    _lat_answer = _pc() - _t_ans
    model_name = getattr(llm, "model_name", DEFAULT_SERVE_MODEL)
    est_cost = _estimate_cost(model_name, cb.prompt_tokens, cb.completion_tokens)
    opik.update_current_span(
        usage={
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
            "total_tokens": cb.total_tokens,
        },
        metadata={"model": model_name, "structured_output": structured_ok},
        total_cost=est_cost,
    )

    # Resolve the documents the LLM cited.
    cited_chunks, seen_cited = [], set()
    if structured_ok:
        # Structured path: gather ids from the `sources` field AND any the model inlined
        # in the answer prose (some models write a "sources:" line instead of / as well as
        # filling the field). Extract the bare UUID from each before lookup.
        raw_ids = list(cited_ids) + re.findall(_UUID_REGEX, answer or "")
        for raw in raw_ids:
            m = re.search(_UUID_REGEX, str(raw))
            cid = m.group(0) if m else str(raw).strip()
            if not cid or cid in seen_cited:
                continue
            doc = relevant_documents_dict.get(cid) or _get_child_docs_by_id().get(cid)
            if doc is not None:
                cited_chunks.append(doc)
                seen_cited.add(cid)
        # Clean the prose too, in case the model inlined a "sources:" line / ids.
        clean_answer = _clean_answer_text(answer or "")
    else:
        # Fallback path: match retrieved docs against the raw answer text.
        for doc in relevant_documents:
            meta = doc.metadata
            cid = str(meta.get(AvailableMetaData.ID, ""))
            if cid and cid in seen_cited:
                continue
            identifiers = [
                str(meta.get("source", "")).strip(),
                cid,
                str(meta.get(AvailableMetaData.STORAGE_PATH, "")).strip(),
            ]
            if any(ident and ident in answer for ident in identifiers):
                cited_chunks.append(doc)
                seen_cited.add(cid)
        for match in re.finditer(_UUID_REGEX, answer):
            idx = match.group(0)
            if idx in seen_cited:
                continue
            doc = relevant_documents_dict.get(idx) or _get_child_docs_by_id().get(idx)
            if doc is not None:
                cited_chunks.append(doc)
                seen_cited.add(idx)
        clean_answer = _clean_answer_text(answer)

    # If the model cited nothing, fall back to the top 5 unique documents as sources.
    used_fallback = False
    if not cited_chunks:
        used_fallback = True
        seen_docs: set[str] = set()
        for doc in relevant_documents:
            name = doc.metadata.get(AvailableMetaData.PART_OF_DOCUMENT)
            if not name or name in seen_docs:
                continue
            seen_docs.add(name)
            cited_chunks.append(doc)
            if len(cited_chunks) >= 5:
                break

    if _is_retrieval_debug_enabled():
        tag = "top-5 fallback" if used_fallback else "LLM-cited"
        print(f"[QA] structured={structured_ok} cited_docs={len(cited_chunks)} ({tag}) raw_sources={cited_ids}")
        print(f"[QA] LLM answer ({len(clean_answer)} chars):\n{clean_answer[:800]}")
        print("[QA] source document names:")
        for i, d in enumerate(cited_chunks, 1):
            print(f"     {i}. {_doc_name(d)}")
    opik.update_current_trace(output={"answer": clean_answer})
    _t_judge = _pc()
    # LLM judge is an offline-eval tool, not needed to answer. Off by default in
    # production (saves a full extra LLM call). Enable with AVAILABLE_ENABLE_JUDGE=1.
    if os.environ.get("AVAILABLE_ENABLE_JUDGE", "0") == "1":
        _run_llm_judge(question, clean_answer, relevant_documents)
    _lat_judge = _pc() - _t_judge
    print(f"[LAT] QA breakdown  retrieval={_lat_retrieval:.2f}s  "
          f"answer_llm={_lat_answer:.2f}s  judge_llm={_lat_judge:.2f}s  "
          f"total={_lat_retrieval + _lat_answer + _lat_judge:.2f}s", flush=True)

    # Split retrieved docs into Sources (cited) + Recommendations (rest), and cache
    # the whole run so the Graph/Chunks buttons open panels without re-querying.
    sources, recommendations = _assemble_source_panels(
        relevant_documents, cited_chunks, content_dir, main_folder
    )
    try:
        st.session_state["qa_result"] = {
            "question": question,
            "answer": clean_answer,
            "sources": sources,
            "recommendations": recommendations,
            "chunks": relevant_documents,
            "graph": st.session_state.get("fusion_graph"),
            "opik_url": _opik_trace_url(),
            "retrieval_mode": retrieval_mode,
        }
    except Exception:
        pass
    return clean_answer


def init_global_vars():
    """Initialize global variables.
    Not ideal performance programming, 3 global variables (3 lookup objects) are fine since they're not the bottleneck
    The problem is Gradio can only see the global variables
    """

    global embeddings
    global llm
    global vectorstore
    global retriever
    global _child_docs_cache

    embeddings = get_embeddings(DEFAULT_EMBEDDING_MODEL)
    retriever = get_retriever(
        DEFAULT_EMBEDDING_MODEL,
        DEFAULT_INGESTION_ARTIFACTS_PATH,
        DEFAULT_INDEX_ARTIFACTS_PATH,
    )
    vectorstore = retriever.vectorstore
    _child_docs_cache = None  # rebuilt lazily against the (possibly refreshed) collection
    print("[QA] Load llm")
    llm = get_langchain_llm(model_name=DEFAULT_SERVE_MODEL)


init_global_vars()


# Colors for the QA retrieval graph (Document -> Chunk[leg] -> entities).
_FUSION_GRAPH_LEG_COLORS = {
    "vector": "#4a9eff",   # blue   — found by the vector leg only
    "graph": "#2ecc71",    # green  — found by the graph leg only
    "both": "#a78bfa",     # purple — found by both legs (overlap)
}

# The one topic that carries a middle "module" tier (e.g. O2C, P2P) between Topic
# and Document. Every other topic goes Topic -> Document directly (no module).
_MODULE_TOPIC = "grownfield"


def _module_for_chunk(topic: str, document_id: str) -> str | None:
    """Module tier for a chunk, or None when the topic has no module level.

    Only the GROWNFIELD topic is organised into modules (O2C-02.010 -> O2C). Other
    topics return None so the renderer wires Topic straight to Document.
    """
    if str(topic).lower() != _MODULE_TOPIC:
        return None
    module = _module_key_from_document(document_id)
    return module if module and module != "unknown" else None


def _build_fusion_graph_data(final_scored, vector_id_set, graph_id_set) -> dict:
    """Assemble the QA retrieval-graph payload from the final (reranked) chunks.

    Shape: {version, chunks:[{id, doc, doc_type, level, topic, subdocument, leg,
    score, entities:[entity_id,...]}], counts:{...}}. ``leg`` is vector/graph/both
    from the two legs' id sets; entities come from the graph index, capped per chunk.
    """
    chunk_ids = [d.metadata.get(AvailableMetaData.ID) for d, _ in final_scored]
    ent_map = entities_for_chunks(
        DEFAULT_GRAPH_ARTIFACTS_PATH,
        [cid for cid in chunk_ids if cid],
        max_per_chunk=data_config.FUSION_GRAPH_VIZ_ENTITIES_PER_CHUNK,
    )
    chunks, n_vec, n_graph, n_both = [], 0, 0, 0
    for doc, score in final_scored:
        m = _extract_chunk_meta(doc)
        cid = m["id"]
        in_v, in_g = cid in vector_id_set, cid in graph_id_set
        leg = "both" if (in_v and in_g) else ("vector" if in_v else "graph")
        if leg == "both":
            n_both += 1
        elif leg == "vector":
            n_vec += 1
        else:
            n_graph += 1
        chunks.append(
            {
                "id": cid, "doc": m["doc"], "doc_type": m["doc_type"],
                "level": m["level"], "topic": m["topic"], "subdocument": m["subdocument"],
                "module": _module_for_chunk(m["topic"], m["doc"]),
                "leg": leg, "score": float(score), "entities": ent_map.get(cid, []),
            }
        )
    entity_total = len({e for c in chunks for e in c["entities"]})
    return {
        "version": "fusion_kg",
        "chunks": chunks,
        "counts": {
            "topics": len({c["topic"] for c in chunks}),
            "modules": len({c["module"] for c in chunks if c["module"]}),
            "documents": len({c["doc"] for c in chunks}), "chunks": len(chunks),
            "vector": n_vec, "graph": n_graph, "both": n_both, "entities": entity_total,
        },
    }


# Node styling per hierarchy tier (Topic -> Module -> Document -> Chunk -> Entity).
# Chunks are excluded here — they are coloured by leg (_FUSION_GRAPH_LEG_COLORS).
_FUSION_TIER_STYLE = {
    "topic":    {"shape": "star",     "color": "#e879f9", "size": 30},  # magenta
    "module":   {"shape": "triangle", "color": "#f472b6", "size": 24},  # pink
    "document": {"shape": "ellipse",  "color": "#64748b", "size": 22},  # slate
    "entity":   {"shape": "dot",      "color": "#f39c12", "size": 12},  # orange
}


def _build_fusion_graph_html(data: dict, layout: str = "graph") -> str:
    """Render the QA retrieval graph as a 5-tier hierarchy:

        Topic -> Module -> Document -> Chunk -> Entity

    The module tier appears only for the GROWNFIELD topic (``chunk["module"]``);
    every other topic wires Topic straight to Document. Chunk fill encodes which
    leg(s) found it (blue vector / green graph / purple both); entities hang off
    chunks.

    layout:
      * ``"graph"`` — force-directed. Entity nodes are SHARED, so a concept
        mentioned by two chunks becomes one node that visibly bridges them across
        documents (the knowledge-graph payoff / "how the chunks connect").
      * ``"tree"``  — left-to-right hierarchy. Entity nodes are namespaced per
        chunk, so every branch is a clean drill-down from topic to entity.
    """
    from pyvis.network import Network

    tree = str(layout).lower() == "tree"
    net = Network(
        height="560px", width="100%", directed=True,
        bgcolor="#0f172a", font_color="white",
    )
    if tree:
        net.set_options(
            '{"layout":{"hierarchical":{"enabled":true,"direction":"LR",'
            '"sortMethod":"directed","levelSeparation":220,"nodeSpacing":90,'
            '"treeSpacing":140}},"physics":{"enabled":false}}'
        )
    else:
        net.barnes_hut(
            gravity=-8000, central_gravity=0.3,
            spring_length=160, spring_strength=0.05, damping=0.09,
        )

    seen: set[str] = set()
    edges: set[tuple[str, str]] = set()
    leg_label = {"vector": "vector", "graph": "graph", "both": "vector + graph"}

    def _tier(node_id: str, tier: str, label: str, level: int, title: str) -> None:
        if node_id in seen:
            return
        seen.add(node_id)
        style = _FUSION_TIER_STYLE[tier]
        net.add_node(
            node_id, label=label, shape=style["shape"], color=style["color"],
            size=style["size"], level=level, title=title,
        )

    def _link(src: str, dst: str, color: str, title: str = "") -> None:
        if (src, dst) in edges:
            return
        edges.add((src, dst))
        net.add_edge(src, dst, color=color, arrows="to", title=title)

    for c in data.get("chunks", []):
        topic = c.get("topic") or "N/A"
        module = c.get("module")
        doc = c.get("doc") or "N/A"

        topic_id = f"topic::{topic}"
        doc_id = f"doc::{doc}"
        _tier(topic_id, "topic", str(topic)[:24], 0, f"Topic\n{topic}")
        _tier(doc_id, "document", str(doc)[:28], 2, f"Document\n{doc}\nTopic: {topic}")

        # Topic -> [Module ->] Document. Module tier is GROWNFIELD-only.
        if module:
            module_id = f"module::{topic}::{module}"
            _tier(module_id, "module", str(module)[:20], 1,
                  f"Module\n{module}\nTopic: {topic}")
            _link(topic_id, module_id, "#475569")
            _link(module_id, doc_id, "#475569")
        else:
            _link(topic_id, doc_id, "#475569")

        # Chunk node — colored by which leg(s) found it.
        color = _FUSION_GRAPH_LEG_COLORS.get(c["leg"], "#4a9eff")
        net.add_node(
            c["id"], label=f"{c['doc_type']}\n{c['id'][:8]}", shape="box",
            color=color, size=18, level=3,
            title=(
                f"Chunk {c['id'][:8]}  ({leg_label.get(c['leg'], c['leg'])})\n"
                f"Document: {doc}\n"
                f"Type: {c['doc_type']}   Level: {c['level']}\n"
                f"Rerank score: {c['score']:.3f}"
            ),
        )
        _link(doc_id, c["id"], "#475569")

        # Entities. Graph view shares one node per entity (cross-links chunks);
        # tree view namespaces per chunk so each branch stays a strict tree.
        style = _FUSION_TIER_STYLE["entity"]
        for eid in c.get("entities", []):
            node_id = f"{c['id']}::{eid}" if tree else eid
            if node_id not in seen:
                seen.add(node_id)
                net.add_node(
                    node_id, label=str(eid).split("::")[-1][:24], shape=style["shape"],
                    color=style["color"], size=style["size"], level=4,
                    title=f"Entity\n{eid}",
                )
            _link(c["id"], node_id, "#f39c12", "mentions")

    return net.generate_html()


def _build_traversal_graph_html(traversal: dict) -> str:
    from pyvis.network import Network

    net = Network(
        height="520px", width="100%", directed=True,
        bgcolor="#0f172a", font_color="white",
    )
    net.barnes_hut(
        gravity=-8000, central_gravity=0.3,
        spring_length=160, spring_strength=0.05, damping=0.09,
    )

    vector_ids = {doc.metadata.get(AvailableMetaData.ID, "") for doc, _ in traversal["vector_chunks"]}
    graph_ids = {doc.metadata.get(AvailableMetaData.ID, "") for doc, _ in traversal["graph_chunks"]}
    entity_ids = {
        eid for eid, chunks in traversal.get("entity_to_expanded", {}).items() if chunks
    }

    def _chunk_title(tag: str, m: dict, score: float) -> str:
        return (
            f"{tag}\n"
            f"Topic:       {m['topic']}\n"
            f"Level:       {m['level']}\n"
            f"Subfolder:   {m['subfolder']}\n"
            f"Document:    {m['doc']}\n"
            f"Subdocument: {m['subdocument']}\n"
            f"Type:        {m['doc_type']}\n"
            f"Score:       {score:.3f}\n"
            f"ID:          {m['id']}"
        )

    # Vector chunk nodes — blue
    for doc, score in traversal["vector_chunks"]:
        m = _extract_chunk_meta(doc)
        label = f"{m['doc'][:22]}\n{m['id'][:8]}"
        net.add_node(m['id'], label=label, title=_chunk_title("Vector chunk", m, score),
                     color="#4a9eff", shape="box", size=20)

    # Entity nodes — orange
    for entity_id in entity_ids:
        label = entity_id.split("::")[-1][:24]
        net.add_node(entity_id, label=label, title=f"Entity\n{entity_id}",
                     color="#f39c12", shape="dot", size=16)

    # Graph-expanded chunk nodes — green
    for doc, score in traversal["graph_chunks"]:
        m = _extract_chunk_meta(doc)
        label = f"{m['doc'][:22]}\n{m['id'][:8]}"
        net.add_node(m['id'], label=label, title=_chunk_title("Graph chunk", m, score),
                     color="#2ecc71", shape="box", size=20)

    # Edges: Vector chunk → entity
    for chunk_id, entities in traversal.get("seed_to_entities", {}).items():
        for entity_id in entities:
            if entity_id in entity_ids:
                net.add_edge(chunk_id, entity_id, color="#4a9eff", title="mentions", arrows="to")

    # Edges: entity → graph chunk
    for entity_id, chunks in traversal.get("entity_to_expanded", {}).items():
        if entity_id in entity_ids:
            for chunk_id in chunks:
                if chunk_id in graph_ids:
                    net.add_edge(entity_id, chunk_id, color="#2ecc71", title="mentions", arrows="to")

    return net.generate_html()


# --- QA UI helpers (Claude-style: answer + references, details behind panels) ---

_UUID_REGEX = r"[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}"
_LEG_LABELS = {"vector": "vector", "graph": "graph", "both": "vector + graph"}


def _clean_answer_text(answer: str) -> str:
    """Answer prose only — drop the trailing 'SOURCES:' block and any stray chunk ids."""
    m = re.search(r"\n?\s*SOURCES?\s*:", answer, flags=re.IGNORECASE)
    if m:
        answer = answer[: m.start()]
    answer = re.sub(_UUID_REGEX, "", answer)        # stray inline ids
    answer = re.sub(r"[(\[]\s*[)\]]", "", answer)    # empty () or [] left behind
    return answer.strip()


def _doc_name(doc) -> str:
    return str(doc.metadata.get(AvailableMetaData.PART_OF_DOCUMENT) or "Untitled")


def _source_entry(doc, content_dir, main_folder) -> dict:
    """Display entry for one document (shared by the Sources and Recommendations lists).

    Robust to chunks whose storage path can't be normalised (e.g. training chunks that
    lack the main data folder): falls back to the document name instead of crashing.
    """
    name = _doc_name(doc)
    title = name
    try:
        storage_path = prepare_path(doc, AvailableMetaData.STORAGE_PATH)
        storage_path = join_paths(storage_path, content_dir, main_folder)
        title = path_display_name(storage_path, DATA_DIR)
    except Exception:
        pass
    try:
        trust = get_trust_score_status(doc)
    except Exception:
        trust = ""
    # Numeric TrustRank score folded in by _apply_document_trust (distinct from the
    # categorical manual-check `trust` status above). None when trust is disabled.
    return {
        "document": name,
        "title": title,
        "trust": trust,
        "final_trust": doc.metadata.get("final_trust"),
    }


def _assemble_source_panels(relevant_documents, cited_chunks, content_dir, main_folder):
    """Split retrieved docs into Sources (cited) and Recommendations (the rest).

    - Sources: unique documents behind the cited chunks.
    - Recommendations: every other unique document among the retrieved chunks, minus
      the Sources (cited wins). Each is tagged with the leg (vector/graph/both)
      rolled up from the retrieval graph's per-chunk provenance.
    """
    doc_legs: dict[str, set] = {}
    graph = st.session_state.get("fusion_graph") or {}
    for c in graph.get("chunks", []):
        doc_legs.setdefault(c.get("doc"), set()).add(c.get("leg"))

    def _leg(name: str):
        legs = doc_legs.get(name) or set()
        if not legs:
            return None
        if "both" in legs or ({"vector", "graph"} <= legs):
            return "both"
        return next(iter(legs))

    sources, source_names = [], set()
    for doc in cited_chunks:
        if not doc.metadata.get(AvailableMetaData.PART_OF_DOCUMENT):
            continue  # skip chunks with no real document (e.g. training chunks)
        name = _doc_name(doc)
        if name in source_names:
            continue
        source_names.add(name)
        entry = _source_entry(doc, content_dir, main_folder)
        entry["leg"] = _leg(name)
        sources.append(entry)

    recommendations, rec_names = [], set()
    for doc in relevant_documents:
        if not doc.metadata.get(AvailableMetaData.PART_OF_DOCUMENT):
            continue  # skip chunks with no real document (e.g. training chunks)
        name = _doc_name(doc)
        if name in source_names or name in rec_names:
            continue
        rec_names.add(name)
        entry = _source_entry(doc, content_dir, main_folder)
        entry["leg"] = _leg(name)
        recommendations.append(entry)

    return sources, recommendations


def _opik_trace_url() -> str:
    """Best-effort deep link to the current Opik trace (falls back to the UI base)."""
    base = os.getenv("OPIK_UI_URL", "http://localhost:5173")
    try:
        import base64
        from opik import opik_context

        trace_data = opik_context.get_current_trace_data()
        trace_id = getattr(trace_data, "id", None)
        if trace_id:
            path_b64 = base64.b64encode(f"{base}/api".encode()).decode()
            return f"{base}/api/v1/session/redirect/projects/?trace_id={trace_id}&path={path_b64}"
    except Exception:
        pass
    return base


def _render_qa_answer_area() -> None:
    result = st.session_state.get("qa_result")
    if not result:
        st.info("Ask a question below to get started.")
        return

    # Answer — clean prose.
    st.markdown(result.get("answer") or "_No answer generated._")

    # Which mode produced this answer (explains why the graph view may be absent).
    mode = result.get("retrieval_mode", "thinking")
    mode_meta = _RETRIEVAL_MODES.get(mode, _RETRIEVAL_MODES["thinking"])
    st.caption(f"{mode_meta['label']} mode · {mode_meta['desc']}")

    # Sources — always visible, the documents the LLM cited.
    sources = result.get("sources") or []
    if sources:
        st.markdown("**Sources** — used in this answer")
        for i, s in enumerate(sources, 1):
            t = s.get("final_trust")
            trust_suffix = f"  —  trust {float(t):.2f}" if isinstance(t, (int, float)) else ""
            st.markdown(f"{i}. {s['title']}{trust_suffix}")

    # Related documents — the recommendations (inline expander).
    recs = result.get("recommendations") or []
    if recs:
        with st.expander(f"Related documents ({len(recs)})"):
            for r in recs:
                leg = _LEG_LABELS.get(r.get("leg"))
                suffix = f"  —  {leg}" if leg else ""
                t = r.get("final_trust")
                trust_suffix = f"  —  trust {float(t):.2f}" if isinstance(t, (int, float)) else ""
                st.markdown(f"- {r['title']}{suffix}{trust_suffix}")

    # Retrieved chunks — the evidence sent to the LLM (inline expander).
    chunks = result.get("chunks") or []
    if chunks:
        with st.expander(f"Retrieved chunks ({len(chunks)})"):
            for n, doc in enumerate(chunks):
                display_document_in_streamlit(doc, n)

    # Retrieval graph (inline expander) — Thinking mode only. In Fast (vector-only)
    # mode the graph leg never runs, so there is no traversal to show.
    graph = result.get("graph")
    if mode != "fast" and graph and graph.get("chunks"):
        with st.expander("Retrieval graph"):
            cnt = graph.get("counts", {})
            st.caption(
                f"{cnt.get('topics', 0)} topics, {cnt.get('modules', 0)} modules, "
                f"{cnt.get('documents', 0)} documents, {cnt.get('chunks', 0)} chunks "
                f"({cnt.get('vector', 0)} vector, {cnt.get('graph', 0)} graph, "
                f"{cnt.get('both', 0)} both), {cnt.get('entities', 0)} entities"
            )
            view = st.radio(
                "Layout", ["Graph", "Tree"], horizontal=True,
                key="fusion_graph_layout", label_visibility="collapsed",
            )
            try:
                st.components.v1.html(
                    _build_fusion_graph_html(graph, layout=view.lower()),
                    height=580, scrolling=True,
                )
            except Exception as e:
                st.warning(f"Graph rendering failed: {e}")

    # Logs — link to this run's Opik trace (inline expander).
    with st.expander("Logs"):
        url = result.get("opik_url") or "http://localhost:5173"
        st.markdown(f"[Open this run in Opik]({url})")


# Retrieval modes surfaced in the Q&A input. "thinking" = vector + graph (default,
# deeper); "fast" = vector only (quicker, no graph view).
_RETRIEVAL_MODES = {
    "thinking": {"label": "Thinking", "desc": "vector + knowledge graph"},
    "fast":     {"label": "Fast",     "desc": "vector only"},
}
# Stable order + label<->key maps for the input dropdown.
_MODE_KEYS = list(_RETRIEVAL_MODES.keys())
_MODE_LABELS = [m["label"] for m in _RETRIEVAL_MODES.values()]
_LABEL_TO_MODE = {m["label"]: key for key, m in _RETRIEVAL_MODES.items()}


def _render_qa_input() -> None:
    st.divider()
    current_mode = st.session_state.get("retrieval_mode", "thinking")
    with st.form("qa_form", clear_on_submit=True):
        # Examples · question · mode (left of Send) · Send — one compact row.
        c1, c2, c3, c4 = st.columns([1.2, 3.6, 1.0, 0.8])
        example = c1.selectbox(
            "Examples", [""] + PROMPT_EXAMPLES,
            format_func=lambda x: "Examples..." if x == "" else (x[:32] + "..." if len(x) > 32 else x),
            label_visibility="collapsed",
        )
        typed = c2.text_input(
            "Question", placeholder="Ask a question...", label_visibility="collapsed"
        )
        mode_label = c3.selectbox(
            "Mode", _MODE_LABELS,
            index=_MODE_KEYS.index(current_mode),
            label_visibility="collapsed",
        )
        sent = c4.form_submit_button("Send", use_container_width=True)
    if sent:
        # Mode is read at submit time (form-batched) — persists for the next render.
        st.session_state["retrieval_mode"] = _LABEL_TO_MODE.get(mode_label, "thinking")
        chosen = typed.strip() or example.strip()
        if chosen:
            st.session_state["pending_question"] = chosen
            st.rerun()


def show_qa():
    st.title("Q&A")
    st.session_state.setdefault("qa_result", None)
    st.session_state.setdefault("retrieval_mode", "thinking")

    # Persist a thread_id so all questions in this session group into one Opik thread.
    if "opik_thread_id" not in st.session_state:
        import uuid
        st.session_state["opik_thread_id"] = str(uuid.uuid4())

    # Process a freshly submitted question, then render from the cached result.
    pending = st.session_state.pop("pending_question", None)
    if pending:
        with st.spinner("Searching for relevant information..."):
            get_language_manager().update_language(pending)
            ask_question(
                pending,
                history=[],
                use_audio_transcript=True,
                use_image_understanding=True,
                use_consolidated_understanding=True,
                add_all_modalities=True,
                thread_id=st.session_state["opik_thread_id"],
                retrieval_mode=st.session_state.get("retrieval_mode", "thinking"),
            )

    _render_qa_answer_area()
    _render_qa_input()