# Copyright (C) Robert Bosch GmbH 2024.
#
# All rights reserved, also regarding any disposal, exploitation,
# reproduction, editing, distribution, as well as in the event of
# applications for industrial property rights.
#
# Code in this file was created with assistance by GitHub Copilot

from dc_available.common.language_manager import get_language_manager, get_localized_system_message
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import streamlit as st
from autogen import (
    Agent,
    Cache,
    ConversableAgent,
    GroupChat,
    GroupChatManager,
    UserProxyAgent,
    register_function,
)

from dc_available.agents.curriculum_agents import (
    create_function_executor,
)
from dc_available.agents.curriculum_rag import (
    get_information_on_topic,
)
from dc_available.frontend.streamlit_helpers import (
    AVATARS,
)


def controlled_chat(
        user: UserProxyAgent,
        chat_manager: GroupChatManager,
        initial_message: str,
        stream: bool = True,
        context: Optional[str] = None,
        user_proxy: Optional[ConversableAgent] = None,
        max_iterations: int = 20,
):
    language_manager = get_language_manager()
    iterations = 0

    while iterations < max_iterations:
        if stream:
            prompt = st.chat_input()
            if prompt is None and len(chat_manager.chat_messages.values()) > 0:
                return

        # Update language and maintain context
        if stream and prompt:
            current_lang = language_manager.update_language(prompt)
            print(f"Current conversation language: {current_lang}")

        if stream:
            agent_cache = Cache.disk(cache_seed=st.session_state["conversation_id"])
        else:
            agent_cache = None

        if len(chat_manager.chat_messages.values()) == 0:
            # Add language context only to initial system context
            language_context = language_manager.get_conversation_context()
            if language_context:
                context = (context or "") + language_context

            context = "\nContext: " + context if context else ""
            intro_message = {
                "role": "user",
                "name": "system",
                "content": initial_message + context,
            }
        else:
            # Just send clean user message - no language instructions visible
            intro_message = {
                "role": "user",
                "name": "user",
                "content": prompt,  # Clean prompt only
            }
        from time import perf_counter as _pc
        _t_chat = _pc()
        chat_result = user.initiate_chat(
            chat_manager,
            message=intro_message,
            summary_method="last_msg",
            clear_history=False,
            cache=agent_cache,
            is_termination_msg=lambda x: False,
            silent=False,
        )
        print(f"[LAT] agent initiate_chat total={_pc()-_t_chat:.2f}s "
              f"messages={len(chat_manager.groupchat.messages)}", flush=True)

        if (
                "SUMMARY" in chat_manager.groupchat.messages[-1]["content"]
                or "TERMINATE" in chat_manager.groupchat.messages[-1]["content"]
        ):
            return chat_manager.groupchat.messages[-1]["content"]
        else:
            if stream:
                return
            elif user_proxy:
                messages = []
                for msg in chat_manager.groupchat.visible_chat_messages:
                    messages.append(
                        {
                            "role": msg["role"]
                            .replace("topic_assessor", "assistant")
                            .replace("knowledge_assessor", "assistant")
                            .replace("teaching_material_assessor", "assistant"),
                            "content": msg["content"],
                        }
                    )
                max_tries = 3
                for _ in range(max_tries):
                    response = user_proxy.generate_reply(messages=messages)
                    if response:
                        prompt = response
                        break
                if not response:
                    raise ValueError(
                        f"Could not generate a response. (Tried {max_tries} times.)"
                    )
            else:
                prompt = input("user:")
        iterations += 1

    return chat_manager.groupchat.messages[-1]["content"]


@dataclass
class SLGroupChat(GroupChat):
    """A wrapper around a GroupChat that adds support for displaying the relevant
    chat messages, called `visible_chat_messages` in a streamlit chat widget."""

    visible_chat_messages: List[Dict] = field(default_factory=list)

    def append(self, message: Dict, speaker: Agent):
        # Filter out system messages and language instructions
        should_display = (
                message["role"] != "function"
                and speaker.name in AVATARS
                and message["content"]
                and "SUMMARY" not in message["content"]
                and not ("name" in message and message["name"] == "system")
                and not message["content"].startswith("[System:")
                and not "IMPORTANT:" in message["content"]
                and not "<LANGUAGE_INSTRUCTION>" in message["content"]
        )

        if should_display:
            name = speaker.name
            avatar = AVATARS[speaker.name]
            msg = message["content"]

            try:
                msg_dict = {"role": name, "content": msg, "avatar": avatar}
                self.visible_chat_messages.append(msg_dict)
                st.session_state["messages"].append(msg_dict)
                with st.chat_message(speaker.name, avatar=avatar):
                    st.write(msg)
            except:
                pass

        super().append(message, speaker)


def custom_speaker_selection_method(last_speaker, groupchat):
    """This custom speaker selection method increases the probability of a productive
    chat by selecting the most appropriate agent to speak next based on the last
    agent and message. It is currently hardcoded to the curriculum use-case."""
    message = groupchat.messages[-1]
    if "SUMMARY" in message.get("content", ""):
        return groupchat.agent_by_name("user")
    if last_speaker.name in [
        "topic_assessor",
        "knowledge_assessor",
        "teaching_material_assessor",
    ]:
        if message.get("tool_calls", ""):
            return groupchat.agent_by_name("function_executor")
        elif any(
                [
                    keyword in message.get("content", "")
                    for keyword in ["function_executor", "get_information_on_topic"]
                ]
        ):
            return groupchat.agent_by_name("function_executor")
        else:
            return groupchat.agent_by_name("user")
    if last_speaker.name == "user":
        for assistant_name in [
            "topic_assessor",
            "knowledge_assessor",
            "teaching_material_assessor",
        ]:
            if groupchat.agent_by_name(assistant_name) is not None:
                return groupchat.agent_by_name(assistant_name)

    return "auto"


def gen_groupchat(assistant, user, llm_config, rag=False):
    """Generates a chat manager for a group chat with a user, an assistant, and a
    function executor. The assistant is responsible for guiding the conversation and
    the function executor is responsible for executing RAG calls made by the assistant.

    Args:
        assistant: The main LLM agent of the group chat.
        user: An agent representing the user. The user agent is not responsible for
            generating content but is used as a technical solution to allow input from
            the human user.
        llm_config: The LLM configuration.
        rag: Whether to include a function executor for RAG calls.
    """
    if rag:
        function_executor = create_function_executor()
        register_function(
            get_information_on_topic,
            caller=assistant,
            executor=function_executor,
            name="get_information_on_topic",
            description="Retrieve information on a topic.",
        )
        groupchat = SLGroupChat(
            agents=[
                assistant,
                function_executor,
                user,
            ],
            messages=[],
            send_introductions=True,
            max_round=10,
            speaker_selection_method=custom_speaker_selection_method,
            allowed_or_disallowed_speaker_transitions={
                assistant: [user, function_executor],
                function_executor: [assistant],
                user: [assistant],
            },
            speaker_transitions_type="allowed",
        )
    else:
        groupchat = SLGroupChat(
            agents=[
                assistant,
                user,
            ],
            messages=[],
            send_introductions=True,
            max_round=10,
            speaker_selection_method=custom_speaker_selection_method,
            allowed_or_disallowed_speaker_transitions={
                assistant: [user],
                user: [assistant],
            },
            speaker_transitions_type="allowed",
        )
    chat_manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
    )
    return chat_manager