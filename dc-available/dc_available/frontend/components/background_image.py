import streamlit as st

from dc_available.common.data_conversion import img_to_bytes


def set_svg_as_page_bg(png_file, header_image):
    page_bg_img = """
    <style>
    """

    if png_file is not None:
        bin_str = img_to_bytes(png_file)
        page_bg_img += (
            """
            body {
                background-image: url("data:image/png;base64,%s");
                background-size: cover;
            }
        """
            % bin_str
        )

    if header_image is not None:
        bin_header_image = img_to_bytes(header_image)
        page_bg_img += (
            """
            header:before {
                position: absolute;
                left: 0;
                top: 0px;
                width: 100%%;
                height: 6px;
                content: "";
                background-repeat: no-repeat;
                background-size: cover;
                background-position: center center;
                background-image: url("data:image/png;base64,%s");
            }
        """
            % bin_header_image
        )

    page_bg_img += """
    .stApp {
        background-color: rgba(255, 255, 255, 0.8);
    }
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return
