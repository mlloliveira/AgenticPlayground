# app/views/settings.py

from __future__ import annotations

import streamlit as st

from core.config import AppConfig
from core.state import (
    ALL_SIDEBAR_BLOCKS,
    SIDEBAR_BLOCK_LABELS,
    SIDEBAR_BLOCK_RENDERERS, save_preset,
    _safe_rerun, SIDEBAR_EXPANDED_DEFAULTS,
)


def _current_sidebar_layout(cfg: AppConfig) -> list[str]:
    """Return a clean layout list: only known blocks, preserve order, append missing ones at the end."""
    layout = getattr(cfg, "sidebar_layout", None) or [
        "MODELS",
        "WEB",
        "PRESETS",
        "GLOBAL_PARAMS",
        "AGENTS",
        "UI",
        "CONSUL",
        "RESET_USAGE",
    ]
    layout = [b for b in layout if b in ALL_SIDEBAR_BLOCKS]
    # Append any newly added blocks so they don't disappear accidentally
    for b in ALL_SIDEBAR_BLOCKS:
        if b not in layout:
            layout.append(b)
    return layout


def settings_page():
    cfg: AppConfig = st.session_state.app_config
    # Make sure we have a dict for expanded flags
    if not hasattr(cfg, "sidebar_expanded") or getattr(cfg, "sidebar_expanded") is None:
        cfg.sidebar_expanded = {}

    st.title("All Settings")
    st.caption(
        "This page shows *all* configuration options. "
        "You can also choose which sections show up in the sidebar, and in what order."
    )

    # --- Sidebar layout builder (active vs hidden) ---
    st.subheader("Sidebar layout")

    st.markdown(
        "Use the controls below to decide which sections appear in the sidebar and in what order. "
        "This only affects the sidebar; all sections are always available on this page. "
        "You can save your configurations by saving a new preset in the preset tab. "
    )

    # ACTIVE = exactly what is stored in cfg.sidebar_layout
    saved_layout = getattr(cfg, "sidebar_layout", None)
    if not saved_layout:
        # default layout if nothing saved yet
        active = [
            "MODELS",
            "WEB",
            "PRESETS",
            "GLOBAL_PARAMS",
            "AGENTS",
            "UI",
            "CONSUL",
            "RESET_USAGE",
        ]
    else:
        # filter out unknown block ids just in case
        active = [b for b in saved_layout if b in ALL_SIDEBAR_BLOCKS]

    # HIDDEN = blocks that exist but are NOT in active
    hidden = [b for b in ALL_SIDEBAR_BLOCKS if b not in active]

    col_active, col_hidden = st.columns(2)

    # LEFT: active list with ↑ / ↓ / Hide
    with col_active:
        st.markdown("**Shown in sidebar**")
        if not active:
            st.caption("No sections are currently shown in the sidebar.")
        for idx, bid in enumerate(active):
            label = SIDEBAR_BLOCK_LABELS.get(bid, bid)
            c1, c2, c3, c4, c5 = st.columns([0.4, 0.12, 0.12, 0.16, 0.2])
            c1.write(label)

            # Move up
            if c2.button("↑", key=f"sb_up_{bid}", help="Move this section up in the sidebar."):
                if idx > 0:
                    new_active = active[:]
                    new_active[idx - 1], new_active[idx] = new_active[idx], new_active[idx - 1]
                    cfg.sidebar_layout = new_active
                    _safe_rerun()

            # Move down
            if c3.button("↓", key=f"sb_down_{bid}", help="Move this section down in the sidebar."):
                if idx < len(active) - 1:
                    new_active = active[:]
                    new_active[idx + 1], new_active[idx] = new_active[idx], new_active[idx + 1]
                    cfg.sidebar_layout = new_active
                    _safe_rerun()

            # Hide from sidebar
            if c4.button("Hide", key=f"sb_hide_{bid}", help="Hide this section from the sidebar (it stays available here)."):
                new_active = [b for b in active if b != bid]
                cfg.sidebar_layout = new_active
                _safe_rerun()

            # Expanded by default
            exp_key = f"sb_expanded_{bid}"

            # Determine the default the same way the sidebar does
            default_expanded = SIDEBAR_EXPANDED_DEFAULTS.get(bid, True)
            current_expanded = cfg.sidebar_expanded.get(bid, default_expanded)

            exp_val = c5.checkbox(
                "Expanded",
                key=exp_key,
                value=current_expanded,
                help="Show this section expanded by default in the sidebar.",
            )

            # Sync back to config
            cfg.sidebar_expanded[bid] = exp_val

        # --- Default configuration actions (still left) ---
        st.markdown("---")
        c_def_save, c_def_reset = st.columns(2)

        with c_def_save:
            if st.button(
                "💾 Save as default",
                key="save_default_btn",
                help=(
                    "Save the current configuration (model, agents, sidebar layout, etc.) "
                    "as the default for new sessions."
                ),
            ):
                if save_preset("default", cfg):
                    st.success("Saved current configuration as default (presets/default.json).")

        with c_def_reset:
            if st.button(
                "↩ Reset defaults",
                key="reset_defaults_btn",
                help=(
                    "Reset this session back to the built-in defaults. "
                    "Click “💾 Save as default” to save the changes."
                ),
            ):
                # Overwrite the in-memory config with a fresh factory default
                st.session_state.app_config = AppConfig()
                st.success(
                    "Reset to built-in defaults for this session. "
                    "If you want this to be your new startup default, click “💾 Save as default”."
                )
                _safe_rerun()


    # RIGHT: hidden list with +Add
    with col_hidden:
        st.markdown("**Available but hidden**")
        if not hidden:
            st.caption("All sections are currently visible in the sidebar.")
        for bid in hidden:
            label = SIDEBAR_BLOCK_LABELS.get(bid, bid)
            if st.button(f"＋ {label}", key=f"sb_add_{bid}", help="Add this section back to the sidebar."):
                new_active = active[:] + [bid]
                cfg.sidebar_layout = new_active
                _safe_rerun()

    st.markdown("---")

    # --- Full settings (all blocks, regardless of sidebar layout) ---
    st.subheader("All configuration sections")

    st.caption(
        "These are the same sections you can place in the sidebar, but always accessible here. "
        "Changes you make here are applied immediately."
    )

    for bid in ALL_SIDEBAR_BLOCKS:
        renderer = SIDEBAR_BLOCK_RENDERERS.get(bid)
        if renderer:
            renderer(cfg)
