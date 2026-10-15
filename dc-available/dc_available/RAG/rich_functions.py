"""
dc_available/RAG/rich_functions.py — the RICH feature.

RAG over RICH's OWN Qdrant collections (the interviews), deliberately separate from the
QA pipeline: a vector leg + a graph (PPR) leg are union-fused, then a structured LLM
answer + a sources list are produced. Reuses QA's pure fusion/graph utilities but points
them at RICH's own graph (graph/rich) and entity collection (available_rich_entities).
Trust reordering is not applied here (handled in a later step).

Self-contained: it uses its own retriever singleton (get_rich_retriever) and its own
LLM, and never touches qa_functions' globals or session state.

CLI (build the RICH collection from the rich artifacts):
    python -m dc_available.RAG.rich_functions --build
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import streamlit as st
from pydantic import BaseModel, Field
from qdrant_client.http import models as qmodels

from dc_available.RAG.data import AvailableMetaData
from dc_available.RAG.retrieval_helpers import get_rich_retriever
from dc_available.config import data_config
from dc_available.config.llm_config import get_langchain_llm, DEFAULT_SERVE_MODEL
from dc_available.frontend.streamlit_helpers import (
    AVATARS,
    clear_page_content,
    display_document_in_streamlit,
)
from dc_available.frontend.components.background_image import set_svg_as_page_bg
from dc_available.frontend.components.footer import footer
from dc_available.common.img import IMG_DIR

_UUID_REGEX = r"[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}"


# =============================================================================
# Answer generation — structured (prose + cited chunk ids), RICH prompt
# =============================================================================

# Adapted from the RICH_QA_TEMPLATE prototype: keep the identity + strict-grounding +
# language + "I don't have information" rules; the inline `Source: path | file | page`
# formatting is dropped because the UI shows a separate sources list.
RICH_SYSTEM_INSTRUCTIONS = (
    "You are RICH (Rexroth Intelligent Chatbot Helper), an assistant for Bosch Rexroth "
    "interview data. Answer strictly from the provided interview context — do NOT use "
    "outside knowledge. If the answer is not in the context, say \"I don't have "
    "information about that.\" Answer in the same language as the question (German if the "
    "question is in German, otherwise English). "
    "Put the prose answer in 'answer' (no id list there). In 'sources' you MUST list the "
    "bare id of every context chunk whose content you used — never leave 'sources' empty "
    "when the answer draws on the context."
)


class RichAnswer(BaseModel):
    """Structured RICH output: the prose answer plus the chunk ids actually used."""

    answer: str = Field(
        description="Answer to the question, in the same language as the question, using "
        "only the provided interview context. Prose only — no ids or 'sources' line here."
    )
    sources: list[str] = Field(
        default_factory=list,
        description="REQUIRED. The id values (after 'id:') of every context chunk you used, "
        "as bare ids. Must not be empty when the answer draws on the context.",
    )


def _generate_rich_answer(llm, question, documents):
    """Structured answer (prose + cited ids) via function calling. Raises on failure."""
    def _ctx_tag(d):
        cid = d.metadata.get(AvailableMetaData.ID, "")
        return f"[id: {cid}]"

    context = "\n\n".join(f"{_ctx_tag(d)}\n{d.page_content}" for d in documents)
    prompt = f"{RICH_SYSTEM_INSTRUCTIONS}\n\nContext:\n{context}\n\nQuestion: {question}"
    result = llm.with_structured_output(RichAnswer).invoke(prompt)
    if isinstance(result, RichAnswer):
        return (result.answer or "").strip(), [str(s) for s in (result.sources or [])]
    if isinstance(result, dict):
        return str(result.get("answer", "")).strip(), [str(s) for s in (result.get("sources") or [])]
    raise ValueError(f"Unexpected structured output type: {type(result)!r}")


def _clean_answer_text(answer: str) -> str:
    """Answer prose only — drop a trailing 'SOURCES:' block and any stray chunk ids."""
    m = re.search(r"\n?\s*SOURCES?\s*:", answer, flags=re.IGNORECASE)
    if m:
        answer = answer[: m.start()]
    answer = re.sub(_UUID_REGEX, "", answer)
    answer = re.sub(r"[(\[]\s*[)\]]", "", answer)
    return answer.strip()


# =============================================================================
# Sources
# =============================================================================

def _folder_name(doc) -> str:
    sp = str(doc.metadata.get(AvailableMetaData.STORAGE_PATH) or "")
    return Path(sp).name if sp else ""


def _doc_title(doc) -> str:
    return (
        str(doc.metadata.get(AvailableMetaData.PART_OF_DOCUMENT) or "")
        or _folder_name(doc)
        or "Document"
    )


def _rich_sources(cited_docs) -> list[dict]:
    """Unique source documents behind the cited chunks (title + trust)."""
    out, seen = [], set()
    for doc in cited_docs:
        title = _doc_title(doc)
        if title in seen:
            continue
        seen.add(title)
        # The trust applied during fusion (final_trust); fall back to the ingestion field.
        ft = doc.metadata.get("final_trust")
        trust = f"{ft:.3f}" if isinstance(ft, (int, float)) else str(
            doc.metadata.get(AvailableMetaData.TRUST_SCORE, "") or ""
        )
        out.append({"title": title, "trust": trust})
    return out


# =============================================================================
# Query — vector + graph fusion with optional interview scope filter
# =============================================================================

def _rich_trust_file() -> Path:
    """RICH's own metadata_trust.jsonl (Artifacts/rich/trust)."""
    return data_config.source_artifact_paths("rich")["ingestion"].parent / "trust" / "metadata_trust.jsonl"


def _apply_rich_trust(scored):
    """Fold per-interview document trust into the fused scores: score * (1 + β·final_trust).

    Mirrors QA's _apply_document_trust but reads RICH's own trust file. Query-independent — it
    only reorders already-relevant chunks. No-op when TRUST_ENABLED is False. Records the
    resolved trust on doc.metadata['final_trust'] for the sources display. Does not truncate.
    """
    if not getattr(data_config, "TRUST_ENABLED", False):
        return scored
    from dc_available.trust_score.final_trust import trust_for

    beta = getattr(data_config, "TRUST_BETA", 0.5)
    trust_file = _rich_trust_file()
    out = []
    for doc, score in scored:
        t = trust_for(
            doc.metadata.get(AvailableMetaData.PART_OF_DOCUMENT),
            doc.metadata.get(AvailableMetaData.STORAGE_PATH),
            trust_file=trust_file,
        )
        doc.metadata["rerank_score"] = float(score)
        doc.metadata["final_trust"] = round(float(t), 4)
        final = float(score) * (1.0 + beta * float(t))
        doc.metadata["final_score"] = final
        out.append((doc, final))
    return out


def _scope_filter(scope_id: str | None):
    """Qdrant filter restricting the search to one interview (by part_of_training).

    Chunk metadata is stored under the payload 'metadata' key, hence the dotted path.
    """
    if not scope_id:
        return None
    return qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key=f"metadata.{AvailableMetaData.PART_OF_TRAINING.value}",
                match=qmodels.MatchValue(value=scope_id),
            )
        ]
    )


def _search_rich_fusion(question: str, scope_id: str | None = None):
    """Vector + graph (PPR) fusion over RICH's collections. Mirrors QA's
    ``_search_question_fusion`` but uses RICH's own retriever, graph (graph/rich) and
    entity collection (available_rich_entities). Returns ``(documents, docs_by_id)``:
    the top ``FUSION_OUTPUT_K`` fused chunks plus a lookup by chunk id.

    When ``scope_id`` is set (one interview selected), the vector leg is filtered to it
    and graph chunks from other interviews are dropped, so context stays in-interview.
    Trust is intentionally not applied here (handled in a later step).
    """
    from dc_available.graph.retrieval.ppr_retriever import ppr_retrieve
    from dc_available.RAG.rich_graph_build import rich_graph_root
    from dc_available.RAG import fusion as fusion_mod

    retriever = get_rich_retriever()
    search_filter = _scope_filter(scope_id)

    # --- Vector leg (hybrid dense + sparse), top FUSION_VECTOR_K ---
    vector_hits = retriever.vectorstore.similarity_search_with_score(
        question, k=data_config.FUSION_VECTOR_K, filter=search_filter
    )
    vector_ids = [d.metadata.get(AvailableMetaData.ID) for d, _ in vector_hits]
    doc_by_id = {d.metadata.get(AvailableMetaData.ID): d for d, _ in vector_hits}

    # --- Graph leg (PPR over the RICH graph, seeded from RICH's entity collection) ---
    # Never fatal: on any failure / abstain the query degrades to vector-only.
    graph_ids: list[str] = []
    try:
        passage_seeds = [
            (d.metadata.get(AvailableMetaData.ID), float(s)) for d, s in vector_hits
        ]
        graph_result = ppr_retrieve(
            question,
            rich_graph_root(),
            vector_hits=passage_seeds,
            entity_collection=data_config.QDRANT_RICH_ENTITY_COLLECTION,
        )
        if graph_result is not None and not graph_result.abstained:
            graph_ids = [cid for cid, _ in graph_result.chunks[: data_config.FUSION_GRAPH_K]]
    except Exception as exc:  # pragma: no cover - defensive, matches QA's never-fatal leg
        print(f"[RICH] graph leg failed ({exc}); using vector-only for this query.")

    # Hydrate graph-only chunk ids from the RICH docstore.
    missing = [cid for cid in graph_ids if cid not in doc_by_id]
    if missing:
        for mid, mdoc in zip(missing, retriever.docstore.mget(missing)):
            if mdoc is not None:
                doc_by_id[mid] = mdoc

    # Scope the graph leg to the selected interview (the vector leg is already filtered).
    if scope_id:
        graph_ids = [
            cid
            for cid in graph_ids
            if doc_by_id.get(cid) is not None
            and doc_by_id[cid].metadata.get(AvailableMetaData.PART_OF_TRAINING) == scope_id
        ]

    # --- Fuse (union) -> optional rerank -> top FUSION_OUTPUT_K ---
    pool_items = fusion_mod.union_fuse(vector_ids, graph_ids)
    cand = [(doc_by_id[it.key], it.score) for it in pool_items if doc_by_id.get(it.key) is not None]

    if data_config.RERANKER_ENABLED and cand:
        from dc_available.RAG.reranker import rerank

        order = rerank(question, [d.page_content for d, _ in cand], top_k=len(cand))
        cand = [(cand[idx][0], score) for idx, score in order]
    else:
        # union_fuse leaves every score at 0.0 (it preserves ORDER, not magnitudes, and QA
        # relies on the reranker for real scores). With the reranker off, turn the fused
        # order into a descending 0-1 base score (rank 1 -> 1.0) so the trust multiplier
        # below has something non-zero to reorder.
        n = len(cand)
        cand = [(doc, (n - i) / n) for i, (doc, _) in enumerate(cand)]

    # Fold in per-interview document trust: final = score * (1 + β·final_trust).
    # No-op when TRUST_ENABLED is False; reads RICH's own metadata_trust.jsonl.
    cand = _apply_rich_trust(cand)
    cand.sort(key=lambda ds: -ds[1])
    top_scored = cand[: data_config.FUSION_OUTPUT_K]
    documents = [d for d, _ in top_scored]
    docs_by_id = {d.metadata.get(AvailableMetaData.ID): d for d in documents}
    graph_data = _build_rich_graph_data(top_scored, set(vector_ids), set(graph_ids))
    return documents, docs_by_id, graph_data


def _build_rich_graph_data(scored, vector_id_set, graph_id_set) -> dict:
    """Assemble the RICH retrieval-graph payload from the final fused chunks.

    Same shape QA's viz consumes (``_build_fusion_graph_html``): one entry per chunk with
    its leg (vector/graph/both), score and mentioned entities. Entities are read from the
    RICH graph (graph/rich) so the view reflects RICH, not QA. RICH has a single topic and
    no module tier, so ``topic="RICH"`` and ``module=None``.
    """
    from dc_available.graph.retrieval.entity_hop_retriever import entities_for_chunks
    from dc_available.RAG.rich_graph_build import rich_graph_root

    chunk_ids = [d.metadata.get(AvailableMetaData.ID) for d, _ in scored]
    ent_map = entities_for_chunks(
        rich_graph_root(),
        [cid for cid in chunk_ids if cid],
        max_per_chunk=getattr(data_config, "FUSION_GRAPH_VIZ_ENTITIES_PER_CHUNK", 5),
    )
    chunks, n_vec, n_graph, n_both = [], 0, 0, 0
    for doc, score in scored:
        m = doc.metadata
        cid = m.get(AvailableMetaData.ID)
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
                "id": cid,
                "doc": _doc_title(doc),
                "doc_type": m.get(AvailableMetaData.DOCUMENT_TYPE) or "chunk",
                "level": m.get(AvailableMetaData.LEVEL) or "",
                "topic": "RICH",
                "subdocument": None,
                "module": None,
                "leg": leg,
                "score": float(score),
                "entities": ent_map.get(cid, []),
            }
        )
    entity_total = len({e for c in chunks for e in c["entities"]})
    return {
        "version": "fusion_kg",
        "chunks": chunks,
        "counts": {
            "topics": 1 if chunks else 0,
            "modules": 0,
            "documents": len({c["doc"] for c in chunks}),
            "chunks": len(chunks),
            "vector": n_vec,
            "graph": n_graph,
            "both": n_both,
            "entities": entity_total,
        },
    }


def ask_rich_question(question: str, scope_id: str | None = None) -> str:
    """RAG: vector + graph fusion on the RICH collection -> answer + sources.

    When scope_id is given (an interview selected), retrieval is filtered to that
    interview; otherwise it spans all RICH documents. Result is cached under
    st.session_state['rich_result'].
    """
    documents, docs_by_id, graph_data = _search_rich_fusion(question, scope_id)

    llm = get_langchain_llm(model_name=DEFAULT_SERVE_MODEL)
    try:
        answer, cited_ids = _generate_rich_answer(llm, question, documents)
    except Exception as exc:
        print(f"[RICH] structured answer failed ({exc}); using plain text answer.")
        context = "\n\n".join(d.page_content for d in documents)
        prompt = f"{RICH_SYSTEM_INSTRUCTIONS}\n\nContext:\n{context}\n\nQuestion: {question}"
        answer = str(getattr(llm.invoke(prompt), "content", "")).strip()
        cited_ids = []

    # Resolve the documents the model cited (ids from the field + any inlined in prose).
    cited_chunks, seen = [], set()
    for raw in list(cited_ids) + re.findall(_UUID_REGEX, answer or ""):
        m = re.search(_UUID_REGEX, str(raw))
        cid = m.group(0) if m else str(raw).strip()
        if not cid or cid in seen:
            continue
        doc = docs_by_id.get(cid)
        if doc is not None:
            cited_chunks.append(doc)
            seen.add(cid)

    # If nothing was cited, fall back to the top unique documents.
    if not cited_chunks:
        for doc in documents:
            name = _doc_title(doc)
            if name in {_doc_title(c) for c in cited_chunks}:
                continue
            cited_chunks.append(doc)
            if len(cited_chunks) >= 5:
                break

    clean_answer = _clean_answer_text(answer)
    st.session_state["rich_result"] = {
        "question": question,
        "answer": clean_answer,
        "sources": _rich_sources(cited_chunks),
        "chunks": documents,
        "graph": graph_data,
    }
    return clean_answer


# =============================================================================
# Interview list (sidebar selector) — from standard docstore metadata
# =============================================================================

def _title_from_summary(content: str) -> str:
    """The interview title (same as summary.txt's title) from the meta-summary doc.

    The doc's page_content is a fenced ```json {"title": ..., ...}``` blob; return the
    title, or "" if it can't be parsed.
    """
    c = (content or "").strip()
    if c.startswith("```json"):
        c = c[7:]
    elif c.startswith("```"):
        c = c[3:]
    if c.endswith("```"):
        c = c[:-3]
    try:
        return str(json.loads(c.strip()).get("title", "")).strip()
    except Exception:
        return ""


def _list_rich_interviews(retriever) -> list[dict]:
    """List interviews from the RICH docstore (display name + id + trust).

    Uses the per-folder meta-summary document: its id is the part_of_training of its
    chunks (the scope-filter value), and its content carries the interview title shown
    in summary.txt. Falls back to the folder name if the title can't be parsed.
    """
    store = getattr(retriever, "docstore", None)
    if store is None:
        return []
    out = []
    try:
        keys = list(store.yield_keys())
        for key, doc in zip(keys, store.mget(keys)):
            if doc is None:
                continue
            meta = doc.metadata
            is_meta = meta.get(AvailableMetaData.IS_META_SUMMARY) or str(
                meta.get(AvailableMetaData.PAGE_ID)
            ) == "-1"
            if not is_meta:
                continue
            if meta.get(AvailableMetaData.DOCUMENT_TYPE) not in ("training", "folder", "rich_interview"):
                continue
            sp = str(meta.get(AvailableMetaData.STORAGE_PATH) or "")
            folder = Path(sp).name if sp else str(meta.get(AvailableMetaData.PART_OF_TRAINING) or key)
            name = _title_from_summary(doc.page_content) or folder or "Interview"
            out.append({
                "id": key,
                "name": name,
                "trust": str(meta.get(AvailableMetaData.TRUST_SCORE, "") or ""),
            })
    except Exception as exc:
        print(f"[RICH] interview list failed: {exc}")
    return sorted(out, key=lambda x: x["name"])


# =============================================================================
# UI
# =============================================================================

def set_rich_streamlit_layout():
    """Page config + RICH-branded header (colored R-I-C-H + Bosch ribbon + footer)."""
    st.set_page_config(
        page_title="RICH Chatbot",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    clear_page_content()
    set_svg_as_page_bg(None, f"{IMG_DIR}/bosch-ribbon.png")
    st.html(
        """
        <p style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 3.0rem; font-weight: bold; text-align: center;">
                <span style="color: #FF0000;">R</span><span style="color: #00A8E6;">I</span><span style="color: #808080;">C</span><span style="color: #808080;">H</span>
            </span>
            <span style="font-size: 1.0rem; text-align: center; color: #B2B3B5;">
                Developed by Robert Bosch GmbH and Bosch Rexroth AG
            </span>
            <span style="font-size: 2.0rem; text-align: right;">
                <span style="font-weight: bold;">LEARN</span>
                <span style="color: black;">#LikeA</span><span style="font-weight: bold;">Bosch</span>
            </span>
        </p>
        """
    )
    footer()


def _render_rich_answer_area() -> None:
    result = st.session_state.get("rich_result")
    if not result:
        st.info("Ask a question below to get started.")
        return

    st.markdown(result.get("answer") or "_No answer generated._")

    sources = result.get("sources") or []
    if sources:
        st.markdown("**Sources** — used in this answer")
        for i, s in enumerate(sources, 1):
            trust = f"  —  Trust: {s['trust']}" if s.get("trust") else ""
            st.markdown(f"{i}. {s['title']}{trust}")

    chunks = result.get("chunks") or []
    if chunks:
        with st.expander(f"Retrieved chunks ({len(chunks)})"):
            for n, doc in enumerate(chunks):
                display_document_in_streamlit(doc, n)

    # Retrieval graph — same view QA uses, but built from the RICH graph. Chunk fill
    # encodes which leg(s) found it (blue vector / green graph / purple both);
    # entities (orange) hang off the chunks that mention them.
    graph = result.get("graph")
    if graph and graph.get("chunks"):
        from dc_available.RAG.qa_functions import _build_fusion_graph_html

        with st.expander("Retrieval graph"):
            cnt = graph.get("counts", {})
            st.caption(
                f"{cnt.get('documents', 0)} documents, {cnt.get('chunks', 0)} chunks "
                f"({cnt.get('vector', 0)} vector, {cnt.get('graph', 0)} graph, "
                f"{cnt.get('both', 0)} both), {cnt.get('entities', 0)} entities"
            )
            view = st.radio(
                "Layout", ["Graph", "Tree"], horizontal=True,
                key="rich_graph_layout", label_visibility="collapsed",
            )
            try:
                st.components.v1.html(
                    _build_fusion_graph_html(graph, layout=view.lower()),
                    height=580, scrolling=True,
                )
            except Exception as e:
                st.warning(f"Graph rendering failed: {e}")


def _render_rich_input() -> None:
    st.divider()
    with st.form("rich_form", clear_on_submit=True):
        c1, c2 = st.columns([5, 0.8])
        typed = c1.text_input(
            "Question", placeholder="Ask a question about the interviews...",
            label_visibility="collapsed",
        )
        sent = c2.form_submit_button("Send", use_container_width=True)
    if sent and typed.strip():
        st.session_state["pending_rich_question"] = typed.strip()
        st.rerun()


def show_rich():
    st.title("RICH")
    st.session_state.setdefault("rich_result", None)

    with st.spinner("Loading RICH database..."):
        retriever = get_rich_retriever()

    interviews = _list_rich_interviews(retriever)

    # Sidebar interview selector. "All Interviews" = no scope filter.
    with st.sidebar:
        st.markdown("### 📋 Interview scope")
        options = ["All Interviews"] + [i["name"] for i in interviews]
        selected = st.selectbox("Scope", options, label_visibility="collapsed")
    scope_id = None
    if selected != "All Interviews":
        match = next((i for i in interviews if i["name"] == selected), None)
        if match:
            scope_id = match["id"]
            if match.get("trust"):
                st.sidebar.caption(f"Trust: {match['trust']}")

    # Reset the shown answer when the scope changes.
    if st.session_state.get("rich_scope") != selected:
        st.session_state["rich_scope"] = selected
        st.session_state["rich_result"] = None

    if not interviews:
        st.warning("No interviews found in the RICH database. Run sync + preprocess + --build first.")

    pending = st.session_state.pop("pending_rich_question", None)
    if pending:
        with st.spinner("Searching the interviews..."):
            ask_rich_question(pending, scope_id=scope_id)

    _render_rich_answer_area()
    _render_rich_input()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="RICH — build the RICH Qdrant collection from rich artifacts")
    ap.add_argument("--build", action="store_true", help="(re)build the RICH vector collection")
    ap.add_argument("--build-graph", action="store_true", help="build the RICH graph + entity index")
    ap.add_argument("--structural-only", action="store_true", help="--build-graph without the LLM semantic phase")
    ap.add_argument("--refresh", action="store_true", help="--build-graph: clean rebuild of the graph root")
    args = ap.parse_args()
    if args.build:
        get_rich_retriever(refresh=True)
        print("[RICH] build complete.")
    elif args.build_graph:
        from dc_available.RAG.rich_graph_build import build_rich_graph

        build_rich_graph(refresh=args.refresh, semantic=not args.structural_only)
        print("[RICH] graph build complete.")
    else:
        ap.print_help()
