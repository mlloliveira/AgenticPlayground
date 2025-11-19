import re, time, json
from typing import List, Dict, Optional
import streamlit as st


from core.config import AppConfig
from views.runtime import run_agents_for_turn, render_mode_scaffold


# ---------- page ----------

def chat_page():
    cfg: AppConfig = st.session_state.app_config
    caption = (
        f"Model mode: {'One model for all ('+cfg.global_model+')' if cfg.same_model_for_all else 'Per-agent models'} · "
        f"Summarizer: {'on' if cfg.summarizer_enabled else 'off'}"
    )
    render_mode_scaffold(
        mode="chat",
        caption=caption,
        selectbox_key="branch_select",
        input_key="chat_input",
        input_label="Type your message…",
        run_turn_fn=run_agents_for_turn,
    )