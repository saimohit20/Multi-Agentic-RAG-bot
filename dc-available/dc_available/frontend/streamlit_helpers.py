# streamlit_helpers.py - Fixed version with proper page isolation

import random
import re
import time
import urllib.parse
from typing import Any, Dict, List, Optional

import streamlit as st
from autogen import Agent, ConversableAgent
from langchain_core.documents import Document

from dc_available.common.img import IMG_DIR
from dc_available import DATA_DIR
from dc_available.common.helpers import path_display_name
from dc_available.RAG.retrieval_helpers import (
    prepare_path,
    get_image_cache_name,
)
from dc_available.frontend.components.background_image import (
    set_svg_as_page_bg,
)
from dc_available.frontend.components.footer import (
    footer,
)
from dc_available.RAG.data import AvailableMetaData
from dc_available.config.data_config import (
    DEFAULT_IMAGE_PATH,
)
from dc_available.summarization.trust_score import (
    AvailableTrustScore,
)

AVATARS = {
    "user": "👤",
    "topic_assessor": "📟",
    "knowledge_assessor": "🦖",
    "teaching_material_assessor": "🤖",
}


def set_streamlit_layout():
    """Set up Streamlit page configuration with proper cloud environment handling"""
    st.set_page_config(
        page_title="AVAILABLE Chatbot",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Clear any existing content first - critical for cloud environments
    clear_page_content()

    set_svg_as_page_bg(None, f"{IMG_DIR}/bosch-ribbon.png")

    # Header with consistent styling
    st.html(
        """
        <p style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 3.0rem; font-weight: bold; text-align: center;">
                <span style="color: black;">AV</span><span style="color: #00C8F7;">AI</span><span style="color: black;">L</span><span style="color: #00B050;">ABLE</span>
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


def clear_page_content():
    """Comprehensive content clearing for cloud environments"""
    # Force clear the main content area
    main_container = st.container()
    main_container.empty()

    # Clear any cached content
    if hasattr(st, '_main_container'):
        st._main_container.empty()

    # Add CSS to ensure clean state
    st.markdown("""
    <style>
    /* Force clear any overlapping content */
    .main .block-container {
        clear: both !important;
        position: relative !important;
        z-index: 1 !important;
    }

    /* Ensure proper content isolation */
    .element-container {
        clear: both !important;
    }

    /* Reset any floating elements */
    .stApp > div {
        position: relative !important;
        z-index: 1 !important;
        clear: both !important;
    }

    /* Hide any residual content */
    .stApp::before {
        content: '';
        display: block;
        clear: both;
    }
    </style>
    """, unsafe_allow_html=True)


def prep_streamlit_session(agents):
    """Enhanced session preparation with better state isolation"""
    # Clear any residual state that might cause conflicts
    cleanup_session_state()

    if "conversation_id" not in st.session_state:
        conversation_id = "01" + str(int(time.time())) + str(random.randint(1000, 9999))
        st.session_state["conversation_id"] = conversation_id
        st.session_state["conversation_state"] = 0
        print("Started new conversation with ID:", conversation_id)

    if "await_user_input" not in st.session_state:
        st.session_state["await_user_input"] = True

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "agents" not in st.session_state:
        st.session_state["agents"] = agents

    agents = st.session_state["agents"]

    # Display messages with proper container isolation
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            if message["content"]:
                st.markdown(message["content"], unsafe_allow_html=False)

    return agents


def cleanup_session_state():
    """Clean up session state to prevent conflicts between pages"""
    # List of keys that should be cleared when switching pages
    transient_keys = [
        'form_submitted', 'active_tab', 'uploaded_files', 'need_file_update',
        'file_to_remove', 'should_clear_files', 'reset_uploader',
        'selected_topic_option', 'custom_topic', 'folder_topic'
    ]

    # Only clear if we're switching contexts (detect page change)
    try:
        # Try to get the current script context
        current_script = st.runtime.scriptrunner.get_script_run_ctx()
        if current_script and hasattr(current_script, 'session_id'):
            session_key = f"last_page_{current_script.session_id}"

            # Get current page name safely
            try:
                # Try different methods to get current script path
                current_page = None

                # Method 1: Try to get from script runner context
                if hasattr(current_script, 'script_path'):
                    current_page = str(current_script.script_path)
                elif hasattr(current_script, 'script_name'):
                    current_page = str(current_script.script_name)

                # Method 2: Try to get from main script
                if not current_page:
                    import __main__
                    if hasattr(__main__, '__file__') and __main__.__file__:
                        current_page = __main__.__file__

                # Method 3: Use a default fallback
                if not current_page:
                    current_page = 'assistant_page'

            except Exception:
                current_page = 'unknown_page'

            if session_key in st.session_state:
                if st.session_state[session_key] != current_page:
                    # Page changed, clean up transient state
                    for key in transient_keys:
                        if key in st.session_state:
                            del st.session_state[key]

            st.session_state[session_key] = current_page

    except Exception as e:
        # If any error occurs in cleanup, just skip it to avoid breaking the app
        print(f"Warning: Session state cleanup failed: {e}")
        pass


def special_chat_quit(
        recipient: ConversableAgent,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
):
    if len(messages) == 0:
        return False, None
    last_message = messages[-1]
    try:
        if "name" in last_message and "SUMMARY" in last_message["content"]:
            st.session_state["conversation_state"] += 1
    except:
        pass
    return True, None


def display_document_in_streamlit(document: Document, n=-1):
    """Display document with proper container isolation"""
    txt = document.page_content

    video = None
    video_txt = None
    start_time = 0
    end_time = None

    if AvailableMetaData.VIDEO_PATH in document.metadata:
        video_path = prepare_path(document, AvailableMetaData.VIDEO_PATH)
        if video_path.exists() and n == 0:
            video = video_path

        m = re.search(
            r"(DigitalTransformation|Hydraulics)/(Basic|Expert|Advanced)/(Training\d)/(.+\.mp4)",
            video_path.as_posix(),
        )
        if not m:
            print(
                f"Video path {document.metadata[AvailableMetaData.VIDEO_PATH]} does not match the DC pattern"
            )
            txt += "\n\n_No video link available_"
            link = None
        else:
            VIDEO_LINK_TEMPLATE = "https://bosch-my.sharepoint.com/personal/mdm1lo_bosch_com/_layouts/15/stream.aspx?id=%2Fpersonal%2Fmdm1lo%5Fbosch%5Fcom%2FDocuments%2FGenAI%5FCR%2DDC%5FCase%2F$TOPIC%2F$LEVEL%2F$TRAINING%2F$FILE&referrer=StreamWebApp%2EWeb&referrerScenario=AddressBarCopied%2Eview%2E8d4ab5ca%2D827a%2D4c03%2Db85f%2Df027f7b8416a"

            link = VIDEO_LINK_TEMPLATE.replace("$TOPIC", urllib.parse.quote(m.group(1)))
            link = link.replace("$LEVEL", urllib.parse.quote(m.group(2)))
            link = link.replace("$TRAINING", urllib.parse.quote(m.group(3)))
            link = link.replace("$FILE", urllib.parse.quote(m.group(4)))

            txt = txt + f"\n\n[Watch the full video here]({link})"

        if AvailableMetaData.START_TIME in document.metadata:
            start_time = float(document.metadata[AvailableMetaData.START_TIME])
            end_time = float(document.metadata[AvailableMetaData.END_TIME])
            start_time_in_min = f"{int(start_time // 60)}:{int(start_time % 60):02d}"
            end_time_in_min = f"{int(end_time // 60)}:{int(end_time % 60):02d}"

            if link is not None:
                link_with_timestamp = (
                        link
                        + '&nav={"playbackOptions":{"startTimeInSeconds":'
                        + str(int(start_time))
                        + "}}"
                )
                txt = txt.replace(link, link_with_timestamp)
            txt += f"\nSelected part from {start_time_in_min} to {end_time_in_min}"

        video_txt = txt

    img = None
    if AvailableMetaData.IMAGE_PATH in document.metadata:
        img_path = prepare_path(document, AvailableMetaData.IMAGE_PATH)
        cached_img_path = DEFAULT_IMAGE_PATH / get_image_cache_name(img_path)
        if cached_img_path.exists():
            img = cached_img_path
        elif img_path.exists():
            img = img_path

    storage_path = prepare_path(document, AvailableMetaData.STORAGE_PATH)
    short_title = path_display_name(storage_path, DATA_DIR)

    if document.metadata[AvailableMetaData.DOCUMENT_TYPE] == "audio":
        txt = "**from audio**\n\n" + txt
        short_title += " (audio)"
    elif document.metadata[AvailableMetaData.DOCUMENT_TYPE] == "image":
        txt = "**from image**\n\n" + txt
        short_title += " (image)"
    elif document.metadata[AvailableMetaData.DOCUMENT_TYPE] == "video":
        txt = "**from video**\n\n" + txt
        short_title += " (video)"

    # Use isolated container for each document
    with st.container():
        with st.status(f"Document {short_title}", state="complete", expanded=n == 0):
            get_trust_score_status(document, write=True)
            st.write(txt)
            if video:
                st.video(str(video), start_time=start_time, end_time=end_time)
                st.write(video_txt)
            elif img:
                st.image(str(img))


def get_trust_score_status(
        document: Document, skip_unchecked: bool = False, write: bool = False
):
    if AvailableMetaData.TRUST_SCORE not in document.metadata:
        if skip_unchecked:
            return
        else:
            document.metadata[
                AvailableMetaData.TRUST_SCORE
            ] = AvailableTrustScore.TRUSTED

    if document.metadata[AvailableMetaData.TRUST_SCORE] == AvailableTrustScore.TRUSTED:
        cmd, color, text, icon = (
            st.info,
            "green",
            "Document was checked and is considered trustworthy!",
            "✅",
        )
    elif (
            document.metadata[AvailableMetaData.TRUST_SCORE]
            == AvailableTrustScore.UNCHECKED
    ):
        cmd, color, text, icon = (
            st.warning,
            "yellow",
            "Document was not checked for wrong and outdated information!",
            "⚠️",
        )
    elif (
            document.metadata[AvailableMetaData.TRUST_SCORE] == AvailableTrustScore.OUTDATED
    ):
        cmd, color, text, icon = (
            st.error,
            "red",
            "Document contains outdated information!",
            "🚨",
        )
    elif document.metadata[AvailableMetaData.TRUST_SCORE] == AvailableTrustScore.WRONG:
        cmd, color, text, icon = (
            st.error,
            "red",
            "Document contains wrong information!",
            "🚨",
        )
    else:
        raise ValueError("Unknown trust state")

    if write:
        cmd(text, icon=icon)
    else:
        return f"{icon}"


def clear_streamlit_cache():
    """Clear Streamlit cache and session state for fresh page load"""
    st.cache_data.clear()
    st.cache_resource.clear()


def safe_rerun():
    """Safe rerun that clears state before rerunning"""
    clear_page_content()
    st.rerun()


def create_page_container():
    """Create a properly isolated page container"""
    clear_page_content()
    return st.container()


def force_page_refresh():
    """Force a complete page refresh - use sparingly"""
    st.markdown("""
    <script>
    window.parent.location.reload();
    </script>
    """, unsafe_allow_html=True)