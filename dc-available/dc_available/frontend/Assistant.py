import os
from dc_available.curriculum_app import show_assistant
from dc_available.frontend.streamlit_helpers import (
    set_streamlit_layout,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    set_streamlit_layout()
    show_assistant()
