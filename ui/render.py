# ui/render.py
import html
import streamlit as st
from typing import Optional
import hashlib
import markdown as md
from core.config import Msg, AppConfig, Branch
from core.state import delete_from_here, _safe_rerun
from core.vision import make_vit_patch_preview
from ui.styles import FINAL_CARD_CSS


__all__ = ["render_message", "bubble_html"]

# ---- Deterministic palette (7 colors; no blue-700) ----
PALETTE = [
    "#059669",  # green-600
    "#D97706",  # amber-600
    "#DC2626",  # red-600
    "#2563EB",  # blue-600
    "#7C3AED",  # violet-600
    "#0D9488",  # teal-600
    "#B45309",  # orange-600
]

def _color_bucket_for(sender: str, buckets: int = 7) -> int:  # <-- default now 7
    """
    Deterministic, process-stable bucket from agent sender name.
    Uses BLAKE2b digest (2 bytes) and mods into [0, buckets-1].
    """
    if not sender:
        return 0
    h = hashlib.blake2b(sender.encode("utf-8"), digest_size=2).digest()
    return int.from_bytes(h, "big") % buckets

def _accent_for(sender: str) -> str:
    """Map the BLAKE2b bucket to the PALETTE for use in cards."""
    idx = _color_bucket_for(sender or "", buckets=len(PALETTE))
    return PALETTE[idx]

def _klass_for(msg: Msg, idx: int, cfg: AppConfig) -> str:
    base = "msg"
    if msg.role == "user":
        return f"{base} user"
    if msg.role == "summary":
        return f"{base} summary"

    klass = f"{base} agent"
    if cfg.colored_bubbles:
        bucket = _color_bucket_for(msg.sender, buckets=7)  # <-- 7 unique colors
        klass += f" c{bucket}"
    return klass


MD_EXTENSIONS = ["fenced_code", "tables", "sane_lists", "nl2br"]

def _md_to_html(text: str) -> str:
    """Convert Markdown to HTML (we already style/contain it in the bubble)."""
    return md.markdown(text or "", extensions=MD_EXTENSIONS, output_format="html5")

def _render_md_card(msg: Msg, cfg: AppConfig):
    #Render “Markdown mode” using the SAME bubble wrapper
    klass = _klass_for(msg, 0, cfg)
    meta_bits = []
    if msg.prompt_tokens or msg.completion_tokens:
        meta_bits.append(f"tokens: {msg.prompt_tokens}+{msg.completion_tokens}")
    if msg.duration_s:
        meta_bits.append(f"time: {msg.duration_s:.2f}s")
    dbg = msg.debug or {}
    if (msg.sources or dbg.get("web_used")):
        meta_bits.append("web✓")
    meta = " · ".join(meta_bits)

    body_html = _md_to_html(msg.content)

    st.markdown(
        f"""
        <div class="{klass}">
          <div class="who">{html.escape(msg.sender or "")}</div>
          <div class="body">{body_html}</div>
          <div class="meta">{meta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_final_markdown_card(msg: Msg, cfg: AppConfig):
    st.markdown(FINAL_CARD_CSS, unsafe_allow_html=True)
    accent = _accent_for(msg.sender or "Final Answer")
    body_html = _md_to_html(msg.content)
    st.markdown(
        f"""
        <div class="final-card" style="--accent:{accent}">
          <div class="final-title">Final Answer</div>
          <div class="final-meta">{html.escape(msg.sender or "")}</div>
          <div class="md-body">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def bubble_html(msg: Msg, idx: int, cfg: AppConfig) -> str:
    klass = _klass_for(msg, idx, cfg)
    meta_bits = []
    if msg.prompt_tokens or msg.completion_tokens:
        meta_bits.append(f"tokens: {msg.prompt_tokens}+{msg.completion_tokens}")
    if msg.duration_s:
        meta_bits.append(f"time: {msg.duration_s:.2f}s")
    dbg = msg.debug or {}
    if (msg.sources or dbg.get("web_used")):
        meta_bits.append("web✓")
    meta = " · ".join(meta_bits)
    safe_body = html.escape(msg.content).replace("\n", "<br/>")
    return f"""
    <div class="{klass}">
      <div class="who">{html.escape(msg.sender)}</div>
      <div class="body">{safe_body}</div>
      <div class="meta">{meta}</div>
    </div>
    """


def render_message(msg: Msg, idx: int, branch: Branch, cfg: AppConfig, editable: bool):
    """
    Renders one message.

    Rules:
      • If msg.kind == "final"  → always render as a 'Final Answer' Markdown card.
      • elif cfg.markdown_all or msg.markdown → render a generic Markdown card.
      • else → render your existing chat bubble.

    The "Context" expander (system/prompt/params + web debug) is shown under whatever
    body was just rendered, honoring cfg.show_prompt_preview.

    If `editable` is True, a tiny ✎ button appears on the right side; clicking it
    sets st.session_state["_editing"] = (branch.id, msg.id).
    """
    if msg.role == "tool":
        return

    def _render_context():
        if not (cfg.show_prompt_preview and msg.debug):
            return
        dbg = msg.debug or {}
        st.markdown("")  # small spacer
        with st.expander("👁 Context"):
            st.markdown("**System:**")
            st.code(dbg.get("system", ""), language="markdown")

            st.markdown("**Prompt body:**")
            st.code(dbg.get("prompt", ""), language="markdown")

            st.markdown("**Params:**")
            st.json(dbg.get("params", {}))

            web_used   = bool(dbg.get("web_used"))
            web_reason = dbg.get("web_reason", "")
            wq         = dbg.get("web_queries", []) or []
            web_be     = dbg.get("web_backend", "")

            st.markdown("**Web status:** " + ("used ✅" if web_used else f"not used ❌ ({web_reason})"))
            if wq:
                st.markdown("**Web queries parsed:** " + ", ".join(wq))
            if web_be:
                st.caption(f"DDG backend: {web_be}")


            # Planner info (no duplicate source list; Sources panel already uses msg.sources)
            if dbg.get("planner_used"):
                st.markdown("**Web Planner**")
                st.json({
                    "queries": dbg.get("web_queries", []),
                    "system_prompt": dbg.get("planner_system_prompt"),
                })

            # --- Tool usage details ---
            tools_used = dbg.get("tools_used", [])
            if tools_used:
                st.markdown("**Tool Calls:**")
                for t in tools_used:
                    name = t.get("name")
                    args = t.get("args")
                    err = t.get("error")
                    out = t.get("output")
                    if err:
                        st.markdown(f"- **{name}**({args}) – Error: {err}")
                    else:
                        st.markdown(f"- **{name}**({args}) – Result: {out}")
                    # If this was a RAG search, show retrieval details
                    if name == "rag_search" and not err:
                        retrieved = t.get("retrieved", [])
                        if retrieved:
                            st.markdown("**Retrieved Chunks:**")
                            for item in retrieved:
                                text = item.get("text", "")
                                score = item.get("score", 0.0)
                                source = item.get("source", "")
                                snippet = (text[:100] + "...") if len(text) > 100 else text
                                st.markdown(f"> {snippet} *(score: {score:.2f}, source: {source})*")
                        context_block = t.get("context", "")
                        if context_block:
                            st.markdown("**RAG Context Block:**")
                            st.code(context_block, language="markdown")


            # --- Vision / ViT-style patches (conceptual) ---
            image_paths = dbg.get("image_paths") or []
            if image_paths and cfg.vision.get("show_patch_grid", True):
                st.markdown("**Vision (ViT-style patches — conceptual)**")
                for idx, ipath in enumerate(image_paths, start=1):
                    info = make_vit_patch_preview(ipath, cfg)
                    if not info:
                        continue
                    preview_size = info["preview_size"]
                    display_width = max(48, preview_size // 1)

                    st.image(
                        info["preview_path"],
                        caption=(
                            f"Downsized from {info['input_size'][0]}×{info['input_size'][1]} "
                            f"to {info['preview_size']}×{info['preview_size']} with "
                            f"{info['num_patches_per_side']*info['num_patches_per_side']} "
                            f"patches ({info['patch_size']}×{info['patch_size']} each)."
                        ),
                        width=display_width,
                    )

                    st.caption(
                        "This grid is a conceptual visualization of how vision transformers tokenize images, not a true representation of the vLLM model."
                    )

            # --- Thinking trace (<think>…</think>) ---
            if getattr(cfg, "show_thinking", False) and getattr(msg, "thinking", None):
                st.markdown("**Thinking (`<think>…</think>`):**")
                st.code(msg.thinking.strip(), language="markdown")

            # Denormalized tokens (if collected for this message)
            if getattr(cfg, "show_denorm_tokens", False):
                toks = getattr(msg, "denorm_tokens", None) or []
                if toks:
                    st.markdown("**Denormalized output tokens:**")
                    pretty = []
                    for t in toks:
                        if not isinstance(t, str):
                            continue
                        # Make control chars visible but otherwise keep Ollama's denormalized text
                        s = t.replace("\n", "\\n").replace("\t", "\\t")
                        pretty.append(f"‹{s}›")
                    st.code(" ".join(pretty), language="text")
        

    with st.container():
        if editable:
            col_msg, col_btn = st.columns([0.96, 0.04])
        else:
            col_msg = st.container()
            col_btn = None

        # ---- LEFT: the bubble / card + Context ----
        with col_msg:
            if getattr(msg, "kind", "") == "final":
                # Final Answer card
                _render_final_markdown_card(msg, cfg)

                vinfo = msg.debug or {}
                voting = (vinfo.get("voting") or "").lower()
                if voting:
                    with st.expander(
                        f"🗳 How voting worked ({'Majority' if voting=='majority' else 'Confidence'})",
                        expanded=False,
                    ):
                        if voting == "majority":
                            st.markdown(
                                "Each agent cast a single vote for the participant with the strongest final answer. "
                                "Self-votes and invalid names are discarded."
                            )
                            counts = vinfo.get("vote_counts", {}) or {}
                            if counts:
                                st.markdown("**Tally**")
                                for name, n in counts.items():
                                    st.markdown(f"- **{name}**: {n}")
                            votes = vinfo.get("votes", []) or []
                            if votes:
                                st.markdown("**Agent votes & justifications**")
                                for v in votes:
                                    st.markdown(
                                        f"- **{v.get('voter','?')}** → **{v.get('choice','?')}**  \n  {v.get('justification','')}"
                                    )
                        elif voting == "confidence":
                            st.markdown(
                                "Each agent produced a final answer and self-reported **Confidence ∈ [0,1]**. "
                                "We compute selection weights with a softmax on confidences:"
                            )
                            st.latex(r"w_i = \frac{e^{\alpha c_i}}{\sum_j e^{\alpha c_j}}")
                            alpha = vinfo.get("alpha")
                            if isinstance(alpha, (int, float)):
                                st.caption(f"α = {alpha}")
                            for c in vinfo.get("candidates", []) or []:
                                st.markdown(
                                    f"- **{c.get('agent','?')}**: confidence={c.get('confidence',0):.2f}, "
                                    f"weight={c.get('weight',0):.3f}"
                                )

            elif getattr(cfg, "markdown_all", False) or getattr(msg, "markdown", False):
                _render_md_card(msg, cfg)
            else:
                st.markdown(bubble_html(msg, idx, cfg), unsafe_allow_html=True)

            _render_context()

        # ---- RIGHT: tiny edit button ----
        if editable and col_btn is not None:
            with col_btn:
                if st.button(
                    "✎",
                    key=f"edit_{branch.id}_{msg.id}",
                    help="Edit & branch",
                ):
                    # Mark this (branch, msg) as the one being edited
                    st.session_state["_editing"] = (branch.id, msg.id)
                    # Immediately rerun so render_mode_scaffold sees _editing
                    _safe_rerun()

