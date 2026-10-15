# Copyright (C) Robert Bosch GmbH 2024.
#
# All rights reserved, also regarding any disposal, exploitation,
# reproduction, editing, distribution, as well as in the event of
# applications for industrial property rights.
#
#
# Code generation was assisted by GitHub Copilot.

import os
import uuid

import streamlit as st
from autogen.agentchat.utils import gather_usage_summary

os.environ.setdefault("OPIK_URL_OVERRIDE", "http://localhost:5173/api")
os.environ.setdefault("OPIK_PROJECT_NAME", "dc-available-rag")
os.environ.setdefault("OPIK_WORKSPACE", "default")

from dc_available.config.llm_config import get_autogen_llm_config, DEFAULT_MODEL

from dc_available.RAG.qa_functions import (
    init_global_vars,
)
from dc_available.agents.curriculum_agents import (
    create_knowledge_assessor,
    create_teaching_material_assessor,
    create_topic_assessor,
    create_user_agent,
)
from dc_available.agents.curriculum_chat import (
    controlled_chat,
    gen_groupchat,
)
from dc_available.agents.agent_tracing import (
    activate_stage_trace,
    detach_stage_trace,
    finish_stage_trace,
    has_stage_trace,
)
from dc_available.config.data_config import (
    update_data_config,
)
from dc_available.frontend.streamlit_helpers import (
    prep_streamlit_session,
)
from dc_available.RAG.retrieval_helpers import (
    prepare_image_cache,
)


AGENTS_PROJECT_NAME = "dc-available-rag"


def _reset_usage(chat_manager):
    """Clear per-agent token counters so each stage reports its own usage only.

    AutoGen accumulates usage on each agent's client across reruns; without this
    reset the token counts would be cumulative across all three stages."""
    for agent in chat_manager.groupchat.agents:
        if getattr(agent, "client", None):
            agent.client.clear_usage_summary()


def _extract_usage(chat_manager) -> dict:
    """Aggregate (non-cached) token usage and cost across all agents."""
    cost = gather_usage_summary(chat_manager.groupchat.agents)
    usage = cost.get("usage_excluding_cached_inference", {})
    prompt_tokens = completion_tokens = total_tokens = 0
    for key, val in usage.items():
        if key != "total_cost" and isinstance(val, dict):
            prompt_tokens += val.get("prompt_tokens", 0)
            completion_tokens += val.get("completion_tokens", 0)
            total_tokens += val.get("total_tokens", 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "total_cost": round(usage.get("total_cost", 0) or 0, 6),
    }


def _stage_metadata(chat_manager) -> dict:
    return {
        **_extract_usage(chat_manager),
        "groupchat_turns": len(chat_manager.groupchat.messages),
    }


def show_assistant():

    # Configure the agents and setup the group chats
    llm_config = get_autogen_llm_config(model_name=DEFAULT_MODEL)
    user = create_user_agent()
    topic_assessor = create_topic_assessor()
    knowledge_assessor = create_knowledge_assessor()
    teaching_material_assessor = create_teaching_material_assessor()
    topic_assessment_manager = gen_groupchat(
        assistant=topic_assessor, user=user, llm_config=llm_config, rag=False
    )
    knowledge_assessment_manager = gen_groupchat(
        assistant=knowledge_assessor, user=user, llm_config=llm_config, rag=False
    )
    teaching_material_assessment_manager = gen_groupchat(
        assistant=teaching_material_assessor, user=user, llm_config=llm_config, rag=True
    )
    AGENTS = {
        "user": user,
        "topic_assessor": topic_assessor,
        "knowledge_assessor": knowledge_assessor,
        "teaching_material_assessor": teaching_material_assessor,
        "topic_assessment_manager": topic_assessment_manager,
        "knowledge_assessment_manager": knowledge_assessment_manager,
        "teaching_material_assessment_manager": teaching_material_assessment_manager,
    }

    stream = True

    if stream:
        agents = prep_streamlit_session(AGENTS)
    else:
        agents = AGENTS

    # Reuse the same thread id as the RAG pipeline so agent traces and any Q&A
    # traces from this session are grouped under one Opik thread.
    if "opik_thread_id" not in st.session_state:
        st.session_state["opik_thread_id"] = str(uuid.uuid4())
    _thread_id = st.session_state["opik_thread_id"]

    topic_summary = st.session_state.get("topic_summary", None)
    knowledge_summary = st.session_state.get("knowledge_summary", None)
    teaching_material_summary = st.session_state.get("teaching_material_summary", None)

    # Topic assessment chat
    if not stream or st.session_state["conversation_state"] == 0:
        _manager = agents["topic_assessment_manager"]
        if not has_stage_trace("topic"):
            _reset_usage(_manager)  # reset token counters only on the first turn
        activate_stage_trace(
            "topic",
            "topic_assessor",
            {"initial_message": "Hi"},
            _thread_id,
            AGENTS_PROJECT_NAME,
        )
        topic_summary = controlled_chat(
            user=agents["user"],
            chat_manager=_manager,
            initial_message="Hi",
            stream=stream,
        )
        if topic_summary:
            topic_summary = topic_summary[topic_summary.find("SUMMARY:") + 8 :]
            finish_stage_trace(
                "topic",
                {"topic_summary": topic_summary},
                _stage_metadata(_manager),
            )
            st.session_state["topic_summary"] = topic_summary
            st.write(f"_{topic_summary}_")
            print(topic_summary)
            st.rerun()
        else:
            detach_stage_trace()

    # Knowledge assessment chat
    if not stream or st.session_state["conversation_state"] == 1:
        _manager = agents["knowledge_assessment_manager"]
        _initial_message = "Please assess my knowledge on the following topic explained in the following context."
        if not has_stage_trace("knowledge"):
            _reset_usage(_manager)  # reset token counters only on the first turn
        activate_stage_trace(
            "knowledge",
            "knowledge_assessor",
            {
                "topic_summary": topic_summary,  # output from Trace 1
                "initial_message": _initial_message,
            },
            _thread_id,
            AGENTS_PROJECT_NAME,
        )
        knowledge_summary = controlled_chat(
            user=agents["user"],
            chat_manager=_manager,
            initial_message=_initial_message,
            context=f"""TOPIC ASSESSMENT CONTEXT: {topic_summary}""",
            stream=stream,
        )
        if knowledge_summary:
            knowledge_summary = knowledge_summary[
                knowledge_summary.find("SUMMARY:") + 8 :
            ]
            finish_stage_trace(
                "knowledge",
                {"knowledge_summary": knowledge_summary},
                _stage_metadata(_manager),
            )
            st.session_state["knowledge_summary"] = knowledge_summary
            st.write(f"_{knowledge_summary}_")
            print(knowledge_summary)
            st.rerun()
        else:
            detach_stage_trace()

    # Teaching material assessment chat
    if not stream or st.session_state["conversation_state"] == 2:
        _manager = agents["teaching_material_assessment_manager"]
        if not has_stage_trace("teaching"):
            _reset_usage(_manager)  # reset token counters only on the first turn
        # activate_stage_trace sets this trace as the current Opik trace, so the
        # RAG spans (vector_retrieval, graph_retrieval, reranker) run during this
        # turn nest underneath it - mirroring the QA rag_pipeline tree.
        activate_stage_trace(
            "teaching",
            "teaching_material_assessor",
            {
                "topic_summary": topic_summary,        # output from Trace 1
                "knowledge_summary": knowledge_summary,  # output from Trace 2
            },
            _thread_id,
            AGENTS_PROJECT_NAME,
        )
        teaching_material_summary = controlled_chat(
            user=agents["user"],
            chat_manager=_manager,
            initial_message="Please assess suggest me teaching material.",
            context=f""""TOPIC ASSESSMENT CONTEXT: {topic_summary}\n
            KNOWLEGE ASSESSMENT CONTEXT: {knowledge_summary}""",
            stream=stream,
        )
        if teaching_material_summary:
            teaching_material_summary = teaching_material_summary[
                teaching_material_summary.find("SUMMARY:") + 8 :
            ]
            finish_stage_trace(
                "teaching",
                {"teaching_material_summary": teaching_material_summary},
                _stage_metadata(_manager),
            )
            st.session_state["teaching_material_summary"] = teaching_material_summary
            print(teaching_material_summary)
            st.rerun()
        else:
            detach_stage_trace()

    print("THE NEW CONVERSATION STATE IS:", st.session_state["conversation_state"])


if __name__ == "__main__":
    import sys

    if "--setup-only" in sys.argv:
        if "-s" in sys.argv:
            source = sys.argv[sys.argv.index("-s") + 1]
            print(f"[Setup] Update data config to {source}")
            update_data_config(source)

        print("[Setup] Generate index")
        init_global_vars()

        print("[Setup] Create cached files")
        # This downloads the embedding model, creates the vector index and copies preview images
        prepare_image_cache()
