import streamlit as st
from streamlit_extras.colored_header import colored_header
from streamlit_extras.stylable_container import stylable_container

from dc_available.config.data_config import (
    DEFAULT_HANDBOOK_PATH,
    DEFAULT_COLOR,
)
from dc_available.frontend.streamlit_helpers import (
    set_streamlit_layout,
)


HANDBOOK = None


def populate_handbook():
    global HANDBOOK

    HANDBOOK = []
    for training in sorted((DEFAULT_HANDBOOK_PATH / "docs").iterdir()):
        if training.is_file():
            continue

        with open(training / "index.md", "r") as f:
            content = f.read()
            content = content.replace("\n# Overview\n\n## Overview\n", "")
            title = content.split("\n")[0].replace("## ", "")
            content = content.replace(title, "").strip()

        HANDBOOK.append((training.name, title, content.strip()))


def show_handbook():
    if HANDBOOK is None:
        populate_handbook()

    with stylable_container(
        key=f"handbook-container",
        css_styles="""
            div[data-testid="stVerticalBlockBorderWrapper"] {
                background-color: rgba(255, 255, 255, 0.9);
                height: 100%;
            }
            """,
    ):
        with st.container(border=False):
            colored_header(
                label="List of Trainings",
                description="These are the available trainings included in the demo",
                color_name=DEFAULT_COLOR,
            )

            for training_name, title, content in HANDBOOK:

                with st.status(
                    f"{training_name}: {title}", expanded=False, state="complete"
                ):
                    st.subheader(title)
                    st.caption(training_name)
                    st.markdown(content)

set_streamlit_layout()
show_handbook()
