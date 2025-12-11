# app/app.py

import streamlit as st
from core.state import ensure_state, sanity_checks, sidebar_once
from ui.styles import GLOBAL_CSS
from views.chat import chat_page
from views.consul import consul_page
from views.notebook import notebook_page
#from views.runtime import runtime_page
from views.settings import settings_page

MODE_TO_PAGE = {
    "chat": chat_page,
    "consul": consul_page,
    "notebook": notebook_page,
}


def main():
    st.set_page_config(page_title="Multi-Agent Playground (Ollama)", layout="wide")

    ensure_state()
    sanity_checks()

    # Global CSS
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # Sidebar (contains mode bar + All Settings button + configurable blocks)
    sidebar_once()

    # Decide which main view to render
    view = st.session_state.get("view", "playground")

    if view == "settings":
        # Lazy import to avoid circulars
        settings_page()
    else:
        mode = st.session_state.get("mode", "chat")
        page_fn = MODE_TO_PAGE.get(mode, chat_page)
        page_fn()


if __name__ == "__main__":
    main()
