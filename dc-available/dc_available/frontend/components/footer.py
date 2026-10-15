import streamlit as st


def footer():
    # Footer implementation
    footer = """
    <style>

    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #eeeeee;
        text-align: center;
        height: 64px;
        padding-top: 14px;
    }
    </style>
    <div class='footer'>
      <p>Copyright Robert Bosch GmbH © 2024</p>
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)
