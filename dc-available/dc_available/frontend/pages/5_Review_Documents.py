# review_portal.py

import streamlit as st
import os
import json
from datetime import datetime
from dotenv import load_dotenv
import logging

from dc_available.frontend.components.footer import footer
from dc_available.frontend.utils.file_handler import move_files, move_files_rejected, get_existing_topics
from dc_available.frontend.utils.teams_sender import send_teams_notification
from PIL import Image  # For handling image files
import fitz  # PyMuPDF for handling PDFs

# Import the retrieval helper
from dc_available.RAG.retrieval_helpers import get_retriever


# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    filename='expert_review.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# Configuration Management
def get_config(key, default=None):
    """Get configuration from environment variables."""
    return os.getenv(key, default)


# Constants
EXPERT_PASSWORD = get_config('EXPERT_PASSWORD', 'default_password_not_secure')

TEMP_DIR = 'main_data_test/temp_uploads'
MAIN_FOLDER = 'main_data_test/DC_data'
REJECTED_DIR = 'main_data_test/rejected_documents'
EXCLUDED_FOLDERS = ['vector_index', 'images']
DEFAULT_LEVELS = ['Basic', 'Intermediate', 'Advanced', 'Expert']


def display_file(file_path):
    """Display file content based on file type."""
    try:
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == ".pdf":
            st.write("**PDF Preview**")
            display_pdf(file_path)

        elif file_extension in [".doc", ".docx"]:
            st.write("**Word Document Preview**")
            st.warning("Word documents cannot be previewed directly. Please download the file to view it.")
            with open(file_path, "rb") as f:
                st.download_button("Download Word Document", f, file_name=os.path.basename(file_path))

        elif file_extension == ".txt":
            st.write("**Text File Preview**")
            with open(file_path, "r") as f:
                st.text(f.read())

        elif file_extension in [".xls", ".xlsx"]:
            st.write("**Excel File Preview**")
            st.warning("Excel files cannot be previewed directly. Please download the file to view it.")
            with open(file_path, "rb") as f:
                st.download_button("Download Excel File", f, file_name=os.path.basename(file_path))

        elif file_extension in [".ppt", ".pptx"]:
            st.write("**PowerPoint File Preview**")
            st.warning("PowerPoint files cannot be previewed directly. Please download the file to view it.")
            with open(file_path, "rb") as f:
                st.download_button("Download PowerPoint File", f, file_name=os.path.basename(file_path))

        elif file_extension in [".mp3", ".wav"]:
            st.write("**Audio Preview**")
            st.audio(file_path)

        elif file_extension in [".mp4", ".avi", ".mov"]:
            st.write("**Video Preview**")
            st.video(file_path)

        elif file_extension in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
            st.write("**Image Preview**")
            image = Image.open(file_path)
            st.image(image, caption=os.path.basename(file_path))

        else:
            st.warning(f"Unsupported file type: {file_extension}")
            with open(file_path, "rb") as f:
                st.download_button("Download File", f, file_name=os.path.basename(file_path))
    except Exception as e:
        logger.error(f"Error displaying file {file_path}: {e}")
        st.error(f"❌ Failed to display file: {e}")


def display_pdf(file_path):
    """Display PDF files as images"""
    try:
        pdf_document = fitz.open(file_path)
        total_pages = len(pdf_document)
        st.write(f"Total pages: {total_pages}")

        if total_pages > 1:
            page_selection = st.slider("Select page", 1, total_pages, 1)
            pages_to_display = [page_selection - 1]  # 0-based index
        else:
            pages_to_display = range(total_pages)

        for page_num in pages_to_display:
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Increased resolution
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            st.image(img, caption=f"Page {page_num + 1}")

    except Exception as e:
        logger.error(f"❌ Failed to display PDF {file_path}: {e}")
        st.error(f"❌ Failed to display PDF: {e}")


def setup_sidebar():
    """Configure sidebar with help information."""
    with st.sidebar:
        st.title("Help & Information")

        st.markdown("""
        ### How to Use This Portal

        This portal allows expert reviewers to:

        1. **Review Submissions**: Examine documents uploaded by users through the Upload Portal

        2. **Preview Files**: View different file types including PDFs, images, text, audio, and video

        3. **Categorize Content**: Modify topic categorization, content level, and training folder name

        4. **Approve or Reject**: Make decisions on document submissions with reviewer comments

        5. **Organize Knowledge Base**: Ensure proper document organization within the system

        ### Review Workflow

        1. Review the document content and metadata
        2. Verify or modify the topic categorization
        3. Adjust the content level if needed
        4. Provide detailed reviewer comments
        5. Approve to add to knowledge base or reject inappropriate submissions

        ### Need Help?

        For assistance with the review portal, contact the system administrator.
        mohamedamine.guedria@bosch.com

        """)


def main():
    st.set_page_config(
        page_title="🔍 Expert Review Portal",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

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

    # Custom CSS
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .stButton button {
        height: 3rem;
        font-weight: 600;
        font-size: 1rem;
    }
    h1, h2, h3 {
        padding-top: 0.5rem;
    }
    .success-message {
        padding: 20px; 
        border-radius: 8px; 
        background-color: #e8f4ea;
        border-left: 5px solid #4CAF50;
        margin: 20px 0;
    }
    /* Footer styling handled by the footer_rexroth component */
    .file-preview {
        border: 1px solid #f0f0f0;
        border-radius: 5px;
        padding: 20px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    </style>
    """, unsafe_allow_html=True)

    # Setup sidebar
    setup_sidebar()

    st.title("🔍 Expert Review Portal")
    st.markdown("Review and approve document submissions to the knowledge base")

    # Expert login
    if "expert_logged_in" not in st.session_state:
        with st.form("login_form", clear_on_submit=True):
            st.write("### Expert Login")
            password = st.text_input("Enter expert password", type="password", key="password_input")
            submit_button = st.form_submit_button("Login")

            if submit_button:
                if password == EXPERT_PASSWORD:
                    st.session_state.expert_logged_in = True
                    st.success("✅ Logged in successfully!")
                    st.rerun()
                else:
                    st.error("❌ Incorrect password. Please try again.")
        return  # Stop further execution until logged in

    # Initialize other session states for form data
    if "selected_topic_option" not in st.session_state:
        st.session_state.selected_topic_option = "Select a topic..."

    if "custom_topic" not in st.session_state:
        st.session_state.custom_topic = ""

    if "folder_topic" not in st.session_state:
        st.session_state.folder_topic = ""

    if "selected_level" not in st.session_state:
        st.session_state.selected_level = "Basic"

    if "training_folder" not in st.session_state:
        st.session_state.training_folder = "Training1"

    if "review_comment" not in st.session_state:
        st.session_state.review_comment = ""

    # Show logout button if logged in
    if st.session_state.get("expert_logged_in", False):
        if st.sidebar.button("Logout"):
            st.session_state.pop("expert_logged_in")
            st.success("🔒 Logged out successfully.")
            st.rerun()

        st.sidebar.success("🌟 Logged in as Expert")

    # List submissions
    submissions = []
    try:
        submissions = [d for d in os.listdir(TEMP_DIR) if os.path.isdir(os.path.join(TEMP_DIR, d))]
        logger.info(f"Found {len(submissions)} submissions.")
    except Exception as e:
        logger.error(f"❌ Error listing submissions: {e}")
        st.error(f"❌ Error accessing submissions directory: {e}")

    if not submissions:
        st.info("ℹ️ No pending submissions to review.")
        st.markdown("""
        When users upload documents through the Upload Portal, they will appear here for expert review.

        As an expert reviewer, you can:
        - Preview uploaded files
        - Verify and modify document categorization
        - Add expert comments to submissions
        - Approve documents to be added to the knowledge base
        - Reject inappropriate or irrelevant submissions
        """)
        return

    # Select submission to review
    st.subheader("Select a Submission to Review")
    selected_submission = st.selectbox(
        "Choose from pending submissions",
        submissions,
        key="selected_submission",
        help="Select a submission to review its content and metadata"
    )

    submission_path = os.path.join(TEMP_DIR, selected_submission)
    metadata_path = os.path.join(submission_path, "metadata.json")

    # Load metadata
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        logger.info(f"✅ Loaded metadata for submission {selected_submission}")
    except Exception as e:
        logger.error(f"❌ Error loading metadata for {selected_submission}: {e}")
        st.error(f"❌ Error loading submission metadata: {e}")
        return

    # Display submission details
    with st.expander("📄 Submission Details", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Basic Information")
            st.write(f"**Submission ID:** {metadata.get('submission_id', 'N/A')}")
            timestamp = metadata.get('timestamp', 'N/A')
            if timestamp != 'N/A':
                try:
                    formatted_time = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%B %d, %Y at %I:%M %p")
                except ValueError:
                    formatted_time = timestamp
            else:
                formatted_time = timestamp
            st.write(f"**Submitted at:** {formatted_time}")

        with col2:
            st.markdown("#### User Information")
            st.write(f"**User Comment:** {metadata.get('user_comment', 'N/A')}")
            st.write(f"**Connection Info:** {metadata.get('connection_info', 'N/A')}")

        st.markdown("#### Current Classification")
        current_topic = metadata.get('topic', 'Uncategorized')
        st.write(f"**Topic:** {current_topic.replace('_', ' ')}")
        st.write(f"**Level:** {metadata.get('level', 'N/A')}")
        st.write(f"**Training Folder:** {metadata.get('training_folder_name', 'Training1')}")

    # Display uploaded files
    st.subheader("📁 Uploaded Files")

    if 'file_paths' not in metadata or not metadata['file_paths']:
        st.warning("⚠️ No files found in this submission.")
    else:
        # Create tabs for each file
        file_tabs = st.tabs([os.path.basename(path) for path in metadata['file_paths']])

        for i, file_path in enumerate(metadata['file_paths']):
            with file_tabs[i]:
                st.markdown(f"**File:** {os.path.basename(file_path)}")
                st.markdown(f"**Type:** {os.path.splitext(file_path)[1].upper()[1:]} file")
                st.markdown(f"**Size:** {os.path.getsize(file_path) / (1024 * 1024):.2f} MB")

                st.markdown("---")
                st.markdown("### File Preview")
                with st.container(border=True):
                    display_file(file_path)

    # Review actions
    st.subheader("🔄 Review and Decision")

    # Get existing topics and current values from metadata
    existing_topics = get_existing_topics(MAIN_FOLDER, EXCLUDED_FOLDERS)

    # Fetch current folder info
    current_topic_raw = metadata.get('topic', 'Uncategorized').replace("_", " ")
    current_level = metadata.get('level', 'Basic')
    current_training_folder = metadata.get('training_folder_name', 'Training1')

    # Topic selection - Outside the form to handle dynamic UI
    st.markdown("### Modify Categorization (If Needed)")

    topic_col1, topic_col2 = st.columns(2)

    with topic_col1:
        # Topic options
        topic_options = ["Select a topic..."] + existing_topics + ["➕ Create new topic"]

        # Find index of the current topic
        if current_topic_raw.replace(" ", "_") in existing_topics:
            default_index = existing_topics.index(current_topic_raw.replace(" ", "_")) + 1  # +1 for "Select a topic"
        elif st.session_state.selected_topic_option in topic_options:
            default_index = topic_options.index(st.session_state.selected_topic_option)
        else:
            default_index = 0

        selected_topic_option = st.selectbox(
            "Select Topic Folder",
            options=topic_options,
            index=default_index,
            help="Choose an existing topic or create a new one",
            key="topic_selector"
        )

        # Store in session state
        st.session_state.selected_topic_option = selected_topic_option

    # Dynamic field for new topic
    with topic_col2:
        # Initialize folder_topic
        folder_topic = ""

        if selected_topic_option == "➕ Create new topic":
            custom_topic = st.text_input(
                "New Topic Name",
                value=st.session_state.custom_topic,
                placeholder="Enter a name for the new topic",
                key="custom_topic_input"
            )

            # Update session state
            st.session_state.custom_topic = custom_topic

            if custom_topic:
                folder_topic = custom_topic.strip().replace(" ", "_")
        else:
            if selected_topic_option != "Select a topic...":
                folder_topic = selected_topic_option
                st.info(f"Using existing topic: {selected_topic_option}")
            else:
                st.warning("Please select a topic or create a new one")

    # Store folder_topic in session state
    st.session_state.folder_topic = folder_topic

    # Level and training folder
    st.markdown("---")
    level_col1, level_col2 = st.columns(2)

    with level_col1:
        # Level options
        level_options = DEFAULT_LEVELS
        if current_level in level_options:
            default_level_index = level_options.index(current_level)
        else:
            default_level_index = 0

        level = st.selectbox(
            "Content Level",
            options=level_options,
            index=default_level_index,
            key="level_selector",
            help="Select the complexity level of the content"
        )

        # Store in session state
        st.session_state.selected_level = level

    with level_col2:
        training_folder = st.text_input(
            "Training Folder Name",
            value=current_training_folder,
            key="training_folder_input",
            help="Specify the name for the Training folder (e.g., Training1)"
        )

        # Store in session state
        st.session_state.training_folder = training_folder

    # Expert Review Comments
    st.markdown("---")
    st.markdown("### Review Comments")
    review_comment = st.text_area(
        "Add Your Review Notes",
        value=st.session_state.review_comment,
        placeholder="Enter your comments about this submission (e.g., quality, relevance, suggestions)",
        key="review_comment_input",
        help="These comments will be stored with the submission metadata"
    )

    # Store in session state
    st.session_state.review_comment = review_comment

    # Action Buttons - Place these outside the form to avoid callback errors
    st.markdown("---")
    st.markdown("### Make a Decision")

    # Validation
    valid_topic = True
    valid_folder = True

    if not folder_topic:
        valid_topic = False
        st.warning("⚠️ Please select or create a valid topic before approving.")

    if not training_folder:
        valid_folder = False
        st.warning("⚠️ Please enter a valid training folder name.")

    # Display buttons side by side
    col1, col2 = st.columns(2)

    # Use the form only for the submission buttons
    with col1:
        # Approve button form
        with st.form(key="approve_form"):
            st.markdown("**Approve this submission?**")
            approve_disabled = not (valid_topic and valid_folder)
            approve_submit = st.form_submit_button(
                "✅ Approve Documents",
                disabled=approve_disabled,
                type="primary"
            )

            if approve_submit:
                try:
                    with st.spinner("⏳ Approving and moving documents..."):
                        # Update metadata
                        metadata['review_status'] = 'approved'
                        metadata['review_comment'] = review_comment
                        metadata['topic'] = folder_topic
                        metadata['level'] = level
                        metadata['training_folder_name'] = training_folder
                        metadata['review_date'] = datetime.now().strftime("%Y%m%d_%H%M%S")

                        # Save updated metadata
                        with open(metadata_path, "w") as f:
                            json.dump(metadata, f, indent=4)
                        logger.info(f"✅ Updated metadata for submission {selected_submission}")

                        # Move files to structured folders
                        move_files(
                            submission_id=selected_submission,
                            topic=folder_topic,
                            level=level,
                            training_folder_name=training_folder,
                            temp_root=TEMP_DIR,
                            target_root=MAIN_FOLDER
                        )
                        logger.info(
                            f"✅ Moved files for submission {selected_submission} to {MAIN_FOLDER}/{folder_topic}/{level}/{training_folder}")

                        # Send Teams notification
                        notification_sent = send_teams_notification(selected_submission, metadata)

                        # Refresh the retriever
                        get_retriever(refresh=True)
                        logger.info(f"✅ Refreshed data retriever after approving submission {selected_submission}")

                    # Feedback to user
                    st.markdown("""
                    <div class="success-message">
                        <h3>✅ Documents Approved!</h3>
                        <p>The submission has been approved and the files have been moved to the knowledge base.</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if not notification_sent:
                        st.warning("⚠️ Documents moved, but failed to send Teams notification.")

                    # Clear the session state for next submission
                    st.session_state.selected_topic_option = "Select a topic..."
                    st.session_state.custom_topic = ""
                    st.session_state.folder_topic = ""
                    st.session_state.review_comment = ""

                except Exception as e:
                    logger.error(f"❌ Error approving submission {selected_submission}: {e}")
                    st.error(f"❌ Error approving submission: {e}")

    with col2:
        # Reject button form
        with st.form(key="reject_form"):
            st.markdown("**Reject this submission?**")
            reject_submit = st.form_submit_button(
                "❌ Reject Documents",
                type="secondary"
            )

            if reject_submit:
                try:
                    with st.spinner("⏳ Rejecting and moving documents..."):
                        # Update metadata
                        metadata['review_status'] = 'rejected'
                        metadata['review_comment'] = review_comment
                        metadata['review_date'] = datetime.now().strftime("%Y%m%d_%H%M%S")

                        # Save updated metadata
                        with open(metadata_path, "w") as f:
                            json.dump(metadata, f, indent=4)
                        logger.info(f"✅ Updated metadata for submission {selected_submission} as rejected")

                        # Move files to rejected directory
                        move_files_rejected(
                            submission_id=selected_submission,
                            rejected_dir=REJECTED_DIR
                        )
                        logger.info(
                            f"✅ Moved files for submission {selected_submission} to {REJECTED_DIR}/Rejected_{selected_submission}")

                    st.markdown("""
                    <div style="padding:20px; border-radius:8px; background-color:#ffebee; border-left:5px solid #f44336; margin:20px 0;">
                        <h3>❌ Documents Rejected</h3>
                        <p>The submission has been rejected and the files have been moved to the rejected folder.</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Clear the session state for next submission
                    st.session_state.selected_topic_option = "Select a topic..."
                    st.session_state.custom_topic = ""
                    st.session_state.folder_topic = ""
                    st.session_state.review_comment = ""

                except Exception as e:
                    logger.error(f"❌ Error rejecting submission {selected_submission}: {e}")
                    st.error(f"❌ Error rejecting submission: {e}")

    # Refresh button
    if st.button("View Next Submission"):
        st.rerun()




if __name__ == "__main__":
    main()