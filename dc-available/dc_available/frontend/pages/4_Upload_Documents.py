# 4_Upload_Documents.py - Fixed version with better state management

import streamlit as st
import os
import uuid
import json
from datetime import datetime
from dotenv import load_dotenv

from dc_available.frontend.components.footer import footer
from dc_available.frontend.utils.file_handler import save_uploaded_files, get_existing_topics
from dc_available.frontend.utils.teams_sender import send_teams_notification

# Load environment variables
load_dotenv()

# Constants
TEMP_DIR = 'resources/temp_uploads'
MAIN_FOLDER = 'resources/DC_data'
EXCLUDED_FOLDERS = ['vector_index', 'images']
MAX_FILE_SIZE_MB = 200

# Ensure directories exist
os.makedirs(MAIN_FOLDER, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


def initialize_session_state():
    """Initialize all session state variables with default values"""
    default_states = {
        'form_submitted': False,
        'active_tab': 0,
        'uploaded_files': [],
        'need_file_update': False,
        'file_to_remove': None,
        'should_clear_files': False,
        'reset_uploader': False,
        'uploader_key_count': 0,
        'selected_topic_option': "Select a topic...",
        'custom_topic': "",
        'folder_topic': "",
        'selected_level': 'Basic',
        'training_folder': "Training1",
        'user_comment': "",
        'connection_info': ""
    }

    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def reset_form_state():
    """Reset form to initial state"""
    st.session_state.form_submitted = False
    st.session_state.uploaded_files = []
    st.session_state.active_tab = 0
    st.session_state.uploader_key_count += 1
    st.session_state.selected_topic_option = "Select a topic..."
    st.session_state.custom_topic = ""
    st.session_state.folder_topic = ""
    st.session_state.selected_level = 'Basic'
    st.session_state.training_folder = "Training1"
    st.session_state.user_comment = ""
    st.session_state.connection_info = ""


def display_file_list(uploaded_files, allow_removal=False):
    """Display uploaded files with size information and validation."""
    if not uploaded_files:
        return True, [], []

    file_errors = []
    valid_uploads = True

    file_container = st.container()

    with file_container:
        for i, file in enumerate(uploaded_files):
            file_size_mb = file.size / (1024 * 1024)
            file_size_str = f"{file_size_mb:.1f} MB"

            if file_size_mb > MAX_FILE_SIZE_MB:
                valid_uploads = False
                file_errors.append(f"{file.name} exceeds the maximum file size of {MAX_FILE_SIZE_MB}MB")
                status_icon = "❌"
                size_color = "red"
            else:
                status_icon = "✅"
                size_color = "green"

            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(
                    f"""
                    <div style="display:flex; align-items:center; margin-bottom:8px; padding:8px; border-radius:5px; background-color:#f8f9fa;">
                        <div style="flex-grow:1;">{status_icon} <b>{file.name}</b></div>
                        <div style="color:{size_color}; font-weight:500;">{file_size_str}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            if allow_removal:
                with col2:
                    remove_key = f"remove_file_{i}_{file.name.replace(' ', '_')}"
                    if st.button("🗑️ Remove", key=remove_key):
                        st.session_state.file_to_remove = i
                        st.session_state.need_file_update = True
                        st.rerun()

    if file_errors:
        with st.expander("View file size errors", expanded=True):
            for error in file_errors:
                st.error(error)

    return valid_uploads, file_errors


def setup_sidebar():
    """Configure sidebar with help information."""
    with st.sidebar:
        st.title("Help & Information")
        st.markdown("""
        ### How to use this portal
        1. Upload your documents by dragging them into the upload area
        2. Fill in the document information
        3. Select an existing topic or create a new one
        4. Configure additional options
        5. Click Submit to upload your documents

        ### Supported file types
        - Documents: PDF, DOCX, TXT, XLSX
        - Presentations: PPT, PPTX
        - Audio: MP3, WAV, OGG, FLAC, M4A
        - Video: MP4, MOV, AVI, MKV, WEBM

        ### Contact
        For assistance, contact mohamedamine.guedria@bosch.com
        """)


def main():
    # Page configuration - moved to top
    st.set_page_config(
        page_title="Document Upload Portal",
        page_icon="📤",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Initialize session state first
    initialize_session_state()

    # Add custom CSS to prevent overlap
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 8rem;  /* Increased bottom padding */
        max-width: 1000px;
    }
    .stButton button {
        width: 100%;
        height: 3rem;
        font-weight: 600;
        font-size: 1rem;
    }
    h1, h2, h3 {
        padding-top: 0.5rem;
    }
    .upload-section {
        margin-bottom: 2rem;
    }
    .success-message {
        padding: 20px; 
        border-radius: 8px; 
        background-color: #e8f4ea;
        border-left: 5px solid #4CAF50;
        margin: 20px 0;
    }
    /* Force clear previous content */
    .element-container {
        clear: both;
    }
    /* Ensure proper stacking context */
    .stApp > div {
        position: relative;
        z-index: 1;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
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
    setup_sidebar()

    # Process file removal and clearing operations
    if st.session_state.file_to_remove is not None:
        if 0 <= st.session_state.file_to_remove < len(st.session_state.uploaded_files):
            del st.session_state.uploaded_files[st.session_state.file_to_remove]
        st.session_state.file_to_remove = None
        st.session_state.reset_uploader = True

    if st.session_state.should_clear_files:
        st.session_state.uploaded_files = []
        st.session_state.should_clear_files = False
        st.session_state.reset_uploader = True

    if st.session_state.reset_uploader:
        st.session_state.uploader_key_count += 1
        st.session_state.reset_uploader = False

    # Define a dynamic key for the uploader
    uploader_key = f"file_uploader_{st.session_state.uploader_key_count}"

    # Navigation function
    def go_to_tab(tab_index):
        st.session_state.active_tab = tab_index
        st.rerun()

    # Show confirmation if form was submitted
    if st.session_state.form_submitted:
        submission_details = st.session_state.submission_details
        st.markdown(f"""
        <div class="success-message">
            <h3>✅ Documents submitted successfully!</h3>
            <p><b>Submission ID:</b> {submission_details['id']}</p>
            <p><b>Topic:</b> {submission_details['topic']}</p>
            <p><b>Files:</b> {submission_details['file_count']} document(s)</p>
            <p><b>Status:</b> An expert will review your submission soon.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Submit New Documents"):
            reset_form_state()
            st.rerun()
        return

    # Clear any previous content
    st.empty()

    # Header Text
    st.title("📥 Document Upload Portal")
    st.markdown("Upload documents for processing and review")

    # Display content based on active tab
    if st.session_state.active_tab == 0:
        # File Upload Tab
        uploaded_files = st.file_uploader("",
                                          type=[
                                              'pdf', 'docx', 'txt', 'xlsx',
                                              'ppt', 'pptx',
                                              'mp3', 'wav', 'ogg', 'flac', 'm4a',
                                              'mp4', 'mov', 'avi', 'mkv', 'webm', 'mpeg4'
                                          ],
                                          accept_multiple_files=True,
                                          help=f"Each file must be under {MAX_FILE_SIZE_MB}MB.",
                                          key=uploader_key
                                          )

        if uploaded_files is not None:
            st.session_state.uploaded_files = uploaded_files

        if st.session_state.uploaded_files:
            st.divider()
            st.subheader("Uploaded Files")

            valid_uploads, file_errors = display_file_list(
                st.session_state.uploaded_files,
                allow_removal=True
            )

            if st.session_state.uploaded_files:
                if st.button("🗑️ Clear All Files", key="clear_all_files_btn"):
                    st.session_state.should_clear_files = True
                    st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)
            if st.session_state.uploaded_files:
                if st.button("Next: Document Information →", key="next_to_info"):
                    go_to_tab(1)

    elif st.session_state.active_tab == 1:
        # Document Information Tab
        st.subheader("📝 Document Information")

        existing_topics = get_existing_topics(MAIN_FOLDER, EXCLUDED_FOLDERS)
        col1, col2 = st.columns(2)

        with col1:
            topic_options = ["Select a topic..."] + existing_topics + ["➕ Create new topic"]
            selected_topic_option = st.selectbox(
                "Topic",
                options=topic_options,
                index=topic_options.index(
                    st.session_state.selected_topic_option) if st.session_state.selected_topic_option in topic_options else 0,
                help="Choose an existing topic or create a new one",
                key="topic_selector"
            )
            st.session_state.selected_topic_option = selected_topic_option

        with col2:
            folder_topic = ""
            if selected_topic_option == "➕ Create new topic":
                custom_topic = st.text_input(
                    "New Topic Name",
                    value=st.session_state.custom_topic,
                    placeholder="Enter a name for the new topic",
                    key="custom_topic_input"
                )
                st.session_state.custom_topic = custom_topic
                if custom_topic:
                    folder_topic = custom_topic.strip().replace(" ", "_")
            else:
                if selected_topic_option != "Select a topic...":
                    folder_topic = selected_topic_option
                    st.info(f"Using existing topic: {selected_topic_option}")
                else:
                    st.warning("Please select a topic or create a new one")
            st.session_state.folder_topic = folder_topic

        # Rest of the document information form...
        st.divider()
        level_col1, level_col2 = st.columns(2)

        with level_col1:
            level_options = ['Basic', 'Intermediate', 'Advanced', 'Expert']
            selected_level = st.selectbox(
                "Content Level",
                options=level_options,
                index=level_options.index(st.session_state.selected_level),
                help="Indicate the complexity level of these documents",
                key="level_selector"
            )
            st.session_state.selected_level = selected_level

        with level_col2:
            training_folder = st.text_input(
                "Training Folder Name",
                value=st.session_state.training_folder,
                help="Specify the name for the training folder",
                key="training_folder_input"
            )
            st.session_state.training_folder = training_folder

        user_comment = st.text_area(
            "Comments",
            value=st.session_state.user_comment,
            placeholder="Provide context about these documents and how they should be used...",
            key="comment_input"
        )
        st.session_state.user_comment = user_comment

        connection_info = st.text_input(
            "Contact Information",
            value=st.session_state.connection_info,
            placeholder="Your contact details or reference information",
            key="connection_input"
        )
        st.session_state.connection_info = connection_info

        # Navigation buttons
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Upload", key="back_to_upload"):
                go_to_tab(0)
        with col2:
            if st.button("Next: Review & Submit →", key="next_to_review"):
                go_to_tab(2)

    elif st.session_state.active_tab == 2:
        # Review & Submit Tab (implement remaining logic...)
        st.subheader("📋 Review & Submit")

        if not st.session_state.uploaded_files:
            st.warning("⚠️ No files uploaded. Please go back and select files.")
            if st.button("← Back to Upload", key="back_to_upload_from_review"):
                go_to_tab(0)
            return

        if not st.session_state.get("folder_topic"):
            st.warning("⚠️ Topic not selected. Please go back and select or create a topic.")
            if st.button("← Back to Document Information", key="back_to_doc_info"):
                go_to_tab(1)
            return

        # Display files and information for review
        st.markdown("### Files")
        display_file_list(st.session_state.uploaded_files, allow_removal=False)

        st.markdown("### Document Information")
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.markdown(f"**Topic:** {st.session_state.folder_topic}")
            st.markdown(f"**Content Level:** {st.session_state.selected_level}")
            st.markdown(f"**Training Folder:** {st.session_state.training_folder}")

        with info_col2:
            if st.session_state.user_comment:
                st.markdown(f"**Comments:** {st.session_state.user_comment}")
            if st.session_state.connection_info:
                st.markdown(f"**Contact Info:** {st.session_state.connection_info}")

        # Submit functionality
        st.divider()
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("← Edit Information", key="edit_info"):
                go_to_tab(1)
        with col2:
            if st.button("🚀 Submit Documents", key="final_submit"):
                with st.spinner("Processing your documents..."):
                    try:
                        submission_id = str(uuid.uuid4())
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                        file_paths = save_uploaded_files(st.session_state.uploaded_files, submission_id)

                        metadata = {
                            "submission_id": submission_id,
                            "timestamp": timestamp,
                            "user_comment": st.session_state.user_comment,
                            "connection_info": st.session_state.connection_info,
                            "file_paths": file_paths,
                            "file_types": [f.name.split('.')[-1] for f in st.session_state.uploaded_files],
                            "file_sizes": [f.size for f in st.session_state.uploaded_files],
                            "topic": st.session_state.folder_topic,
                            "level": st.session_state.selected_level,
                            "training_folder_name": st.session_state.training_folder,
                            "status": "pending"
                        }

                        if st.session_state.selected_topic_option == "➕ Create new topic" and st.session_state.folder_topic:
                            new_topic_path = os.path.join(MAIN_FOLDER, st.session_state.folder_topic)
                            os.makedirs(new_topic_path, exist_ok=True)

                        temp_subdir = os.path.join(TEMP_DIR, submission_id)
                        os.makedirs(temp_subdir, exist_ok=True)
                        with open(os.path.join(temp_subdir, "metadata.json"), "w") as f:
                            json.dump(metadata, f, indent=4)

                        notification_sent = send_teams_notification(submission_id, metadata)

                        st.session_state.submission_details = {
                            "id": submission_id,
                            "topic": st.session_state.folder_topic,
                            "file_count": len(st.session_state.uploaded_files)
                        }

                        st.session_state.form_submitted = True
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error processing your upload: {str(e)}")


if __name__ == "__main__":
    main()