"""Per-stage Opik trace lifecycle for the AutoGen curriculum agents.

Streamlit re-runs `show_assistant()` from the top on every user message, while a
single agent conversation spans several such re-runs until it emits SUMMARY.
A plain trace object would be wiped on each re-run, forcing a new (mostly empty)
trace per message.

These helpers keep ONE trace per agent alive across re-runs by storing its
``TraceData`` in ``st.session_state`` and re-activating it as Opik's "current
trace" on every run. Re-activation is what lets the RAG spans
(vector_retrieval / graph_retrieval / reranker) nest under the stage trace, the
same way the QA pipeline nests under ``rag_pipeline``.
"""
from __future__ import annotations

from typing import Any

import opik
import streamlit as st
from opik import context_storage, datetime_helpers
from opik.api_objects import helpers
from opik.api_objects.trace.trace_data import TraceData


def _state_key(key: str) -> str:
    return f"opik_trace_{key}"


def has_stage_trace(key: str) -> bool:
    """Whether a trace for this stage already exists (i.e. not the first turn)."""
    return _state_key(key) in st.session_state


def activate_stage_trace(
    key: str,
    name: str,
    input: dict[str, Any],
    thread_id: str,
    project_name: str,
) -> TraceData:
    """Create the stage trace once, then re-activate it as current every run.

    On the first turn the trace is created and a "start" is sent. On every turn
    (including the first) the trace is set as the current trace so that any
    ``@opik.track`` spans executed during this run nest underneath it.
    """
    sk = _state_key(key)
    client = opik.get_global_client()
    if sk not in st.session_state:
        trace_data = TraceData(
            id=helpers.generate_id(),
            start_time=datetime_helpers.local_timestamp(),
            name=name,
            input=input,
            project_name=project_name,
            thread_id=thread_id,
        )
        st.session_state[sk] = trace_data
        client.__internal_api__trace__(**trace_data.as_start_parameters)

    trace_data = st.session_state[sk]
    context_storage.set_trace_data(trace_data)
    return trace_data


def finish_stage_trace(
    key: str,
    output: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    """End and send the stage trace, then clear it from context and session."""
    sk = _state_key(key)
    trace_data = st.session_state.get(sk)
    if trace_data is None:
        return
    trace_data.update(output=output, metadata=metadata)
    client = opik.get_global_client()
    client.__internal_api__trace__(**trace_data.init_end_time().as_parameters)
    context_storage.pop_trace_data()
    del st.session_state[sk]


def detach_stage_trace() -> None:
    """Run ended without SUMMARY: unset the current trace but keep it alive.

    The trace stays in ``st.session_state`` so the next re-run reuses it; we only
    remove it from the per-run context storage so unrelated later code in this
    run does not accidentally nest under it.
    """
    context_storage.pop_trace_data()
