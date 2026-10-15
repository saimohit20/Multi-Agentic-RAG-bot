import streamlit as st
from streamlit_extras.card import card
from streamlit_extras.colored_header import colored_header
from streamlit_extras.grid import grid
from streamlit_extras.stylable_container import stylable_container

from dc_available.common.data_conversion import img_to_bytes
from dc_available.common.img import IMG_DIR
from dc_available.config.data_config import DEFAULT_COLOR
from dc_available.frontend.streamlit_helpers import (
    set_streamlit_layout,
)


def show_about():
    colored_header(
        label="About",
        description="About the solution",
        color_name=DEFAULT_COLOR,
    )
    with stylable_container(
        key=f"about-container",
        css_styles="""
            div[data-testid="stVerticalBlockBorderWrapper"] {
                background-color: rgba(255, 255, 255, 0.9);
                height: 100%;
            }
            """,
    ):
        with st.container(border=False):
            st.markdown(
                """
            ## Overview

            AVAILABLE is an innovative chatbot designed to assist users in understanding and navigating
            the complex world of Bosch Rexroth Training materials. This cutting-edge RAG chatbot is based
            on a wealth of real-world training materials, including text documents, PowerPoints, videos,
            and more, sourced directly from the prestigious Bosch Rexroth Training academy.

            The primary goal of AVAILABLE is to serve as a learning helper, providing users with comprehensive
            and accurate answers to their queries while also offering references to the relevant training materials.
            Whether users are seeking information on hydraulic systems, automation technologies, or any other
            topic covered in the Bosch Rexroth Training academy, AVAILABLE is equipped to provide valuable insights
            and guidance.

            One of the key features of AVAILABLE is its ability to understand the specific issues and challenges
            faced by users. By leveraging advanced natural language processing capabilities, the chatbot can
            effectively interpret user queries and provide tailored responses that address the unique needs
            of each individual. This personalized approach ensures that users receive the most relevant and
            helpful information, ultimately enhancing their learning experience.

            Furthermore, AVAILABLE is designed to be intuitive and user-friendly, making it easy for individuals
            to access the wealth of knowledge contained within the Bosch Rexroth Training materials. Whether users
            are new to the world of industrial automation or seasoned professionals seeking to expand their expertise,
            the chatbot serves as a valuable resource for learning and development.

            In addition to providing direct answers to user queries, AVAILABLE also offers references to the
            training materials from which its knowledge is derived. This not only helps users to understand the
            reasoning behind the chatbot's responses but also encourages them to explore the training materials in
            greater depth, further enriching their understanding of the subject matter.

            Overall, AVAILABLE represents a significant advancement in the realm of learning and training support.
            By harnessing the power of real-world training materials and cutting-edge technology, the chatbot
            empowers users to gain a deeper understanding of the topics covered in the Bosch Rexroth Training
            academy, ultimately contributing to their professional growth and success.

            ## Team
            ### Development Team - CR

            The following team members from the Bosch Corporate Research (CR) department have contributed to the development of the AVAILABLE Assistant PoC.

            """
            )
            my_grid = grid(3, 1, 3, vertical_align="bottom")
            styles = {
                "card": {
                    "width": "250px",
                    "height": "250px",
                    "border-radius": "20px",
                    "box-shadow": "0 0 10px rgba(0,0,0,0.5)",
                },
                "filter": {
                    "background-color": "rgba(0, 0, 0, 0.2)"  # <- make the image not dimmed anymore
                },
                "title": {"font-size": "24px", "margin-top": "150px"},
                "text": {"font-size": "14px"},
            }
            with my_grid.container():
                card(
                    key="Lukas",
                    title="Lukas Lange",
                    text=["(CR/AIR3)", "Project Lead & LLM Expert"],
                    image="data:image/png;base64,"
                    + img_to_bytes(f"{IMG_DIR}/team_members/lukas.png"),
                    url="https://connect.bosch.com/profiles/html/profileView.do?userid=EC441172-F6CD-46AD-970D-A1472DA3430F",
                    styles=styles,
                )
            with my_grid.container():
                card(
                    key="Paul",
                    title="Paul Baireuther",
                    text=["(CR/AIR2.2)", "AI Expert"],
                    image="data:image/png;base64,"
                    + img_to_bytes(f"{IMG_DIR}/team_members/paul.png"),
                    url="https://connect.bosch.com/profiles/html/profileView.do?userid=8BF4E6EB-E9BC-442B-818B-100AB1318B13",
                    styles=styles,
                )
            with my_grid.container():
                card(
                    key="Jessica",
                    title="Zhou Zhengyu",
                    text=["(CR/RHI2-NA)", "LLM Expert"],
                    image="data:image/png;base64,"
                    + img_to_bytes(f"{IMG_DIR}/team_members/unknown.png"),
                    url="https://connect.bosch.com/profiles/html/profileView.do?userid=452D6B25-22AE-4D06-94F2-FEBFEADAC4A8",
                    styles=styles,
                )

            my_grid.markdown(
                """
            ### Customer Team - DC

            The following team members from Rexroth (DC) at Bosch have provided valuable insights and feedback throughout the development of the AVAILABLE Assistant PoC.
            """
            )

            with my_grid.container():
                card(
                    key="DC-Lukas",
                    title="Lukas Riesch",
                    text=["(DC/SSD6)", "AI Trainer"],
                    image="data:image/png;base64,"
                    + img_to_bytes(f"{IMG_DIR}/team_members/unknown.png"),
                    url="https://connect.bosch.com/profiles/html/profileView.do?key=a942c0de-33f6-48a0-b512-b3b6038741eb",
                    styles=styles,
                )

            with my_grid.container():
                card(
                    key="Mohamed",
                    title="Mohamed Amine Guedria",
                    text=["(DC/SSD6)", "AI Trainer & Software Engineer"],
                    image="data:image/png;base64,"
                          + img_to_bytes(f"{IMG_DIR}/team_members/Amine.png"),
                    url="https://connect.bosch.com/profiles/html/profileView.do?key=13a31ddf-3e0f-4a3a-814b-7a2271c07355",
                    styles=styles,
                )

set_streamlit_layout()
show_about()
