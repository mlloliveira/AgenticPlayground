# app/views/runtime.py

from __future__ import annotations
import time
from typing import List, Dict, Optional, Tuple, Callable

import streamlit as st

from core.config import AppConfig, Branch, AgentConfig, Msg
from core.state import is_planner, render_usage_sidebar, list_branches_for_mode, get_active_branch, switch_active_branch, add_user_message, _safe_rerun, fork_from_edit
from core.runtime_web import web_brief_for_branch
from core.agents import effective_params, effective_model
from core.prompting import build_prompt_for_agent, parse_thinking_keep_tags, build_prompt_for_summarizer
from core.ollama_client import generate_stream, generate_once
from ui.render import render_message, bubble_html

# ----------------- helpers part 1 ----------------------- #

def last_user_index(branch: Branch) -> Optional[int]:
    for i in range(len(branch.messages) - 1, -1, -1):
        if branch.messages[i].role == "user":
            return i
    return None

def prompt_branch_for_round(cfg, branch: Branch, last_user_idx: int) -> Branch:
    """
    For blind-first-turn:
      - Round 1 (just after user msg): branch contains only the last user msg.
      - Later rounds: branch contains last user msg + prior rounds' agent msgs.
    For non-blind: return the full branch.
    """
    if not getattr(cfg, "blind_first_turn", False):
        return branch

    # Include this turn only: from last user message up to current end
    msgs = branch.messages[last_user_idx:len(branch.messages)]
    return Branch(
        id=branch.id,
        label=branch.label,
        messages=msgs,
        summary=branch.summary,
    )

def prepare_turn_common(cfg: AppConfig, branch: Branch) -> Tuple[Branch, List[AgentConfig], Tuple[str, List[Dict[str,str]], List[str], str, Optional[str]]]:
    """
    Return (prompt_branch, agents, web_ctx) where:
      - prompt_branch = branch or branch cut at last user (if blind_first_turn)
      - agents = enabled agents excluding summarizer and planner
      - web_ctx = (brief, sources, queries, origin, planner_sp)
    """
    # 1) who will talk (exclude summarizer + planner)
    agents = [a for a in cfg.agents if a.enabled and not a.is_summarizer and not is_planner(a)]

    # 2) blind-first-turn: cut branch up to last user
    prompt_branch = branch
    if cfg.blind_first_turn:
        last_user = last_user_index(branch)
        if last_user is not None:
            prompt_branch = Branch(
                id=branch.id,
                label=branch.label,
                messages=branch.messages[:last_user+1],
                summary=branch.summary
            )

    # 3) web brief once (shared by all agents for this turn)
    web_ctx = web_brief_for_branch(cfg, branch)

    return prompt_branch, agents, web_ctx

def run_summarizer_if_enabled(cfg: AppConfig, branch: Branch):
    """Run summarizer and optionally render it, exactly like current Chat/Consul."""
    summ = next((a for a in cfg.agents if a.enabled and a.is_summarizer), None)
    if not (cfg.summarizer_enabled and summ):
        return

    params = effective_params(summ, cfg)
    model = effective_model(summ, cfg)
    system = summ.system_prompt.strip() or "You are a neutral summarizer. Only summarize; do not add opinions."
    s_prompt = build_prompt_for_summarizer(branch)

    out = generate_once(model=model, system=system, prompt=s_prompt, params=params)
    summary = (out.get("content", "") or "").strip()
    branch.summary = summary

    if cfg.summarizer_visible:
        m = Msg(
            role="summary", sender="Summarizer", content=summary,
            prompt_tokens=int(out.get("prompt_eval_count", 0)),
            completion_tokens=int(out.get("eval_count", 0)),
            duration_s=float(out.get("_elapsed", 0.0)),
            debug={"system": system, "prompt": s_prompt, "params": params.__dict__}
        )
        st.empty().markdown(bubble_html(m, len(branch.messages), cfg), unsafe_allow_html=True)
        branch.messages.append(m)

    # Update usage (prompt tokens for summarizer)
    t = st.session_state.totals
    t["prompt_tokens"] += int(out.get("prompt_eval_count", 0))
    t["completion_tokens"] += int(out.get("eval_count", 0))
    t["messages"] += 1
    t["last_context_tokens"] = int(out.get("prompt_eval_count", 0))
    render_usage_sidebar()

def run_agents_for_turn(cfg: AppConfig, branch: Branch):
    """
    Simple mode (Chat): stream each enabled non-special agent once, then summarizer.
    This is the old _run_agents_streaming from chat, factored.
    """
    prompt_branch, agents, web_ctx = prepare_turn_common(cfg, branch)
    if not agents:
        st.info("No enabled agents. Turn on at least one non-summarizer agent.")
        return

    holders = [st.empty() for _ in agents]
    for ag, holder in zip(agents, holders):
        stream_agent_bubble(
            holder=holder,
            agent=ag,
            prompt_branch=prompt_branch,
            final_branch=branch,
            cfg=cfg,
            web_ctx=web_ctx
        )

    run_summarizer_if_enabled(cfg, branch)

# -----------------

def render_mode_scaffold(
    mode: str,
    caption: str,
    selectbox_key: str,
    input_key: str,
    input_label: str,
    run_turn_fn: Callable[[AppConfig, Branch], None],
):
    """
    Shared “page” shell for any mode:
      - filtered branch picker (mode-only)
      - message history with edit/fork affordance
      - pending-turn trigger
      - chat input
    """
    from ui.styles import GLOBAL_CSS
    from ui.render import render_message
    from core.config import AppConfig
    from core.state import ensure_mode

    cfg: AppConfig = st.session_state.app_config
    st.session_state["mode"] = mode
    ensure_mode(mode, cfg)

    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # Filtered branches for this mode
    branch_ids = list_branches_for_mode(mode)
    active_id = st.session_state.active_branch_id
    sel_idx = branch_ids.index(active_id) if active_id in branch_ids else 0

    chosen = st.selectbox(
        "Active branch",
        options=branch_ids,
        format_func=lambda b: st.session_state.branches[b].label,
        index=sel_idx,
        key=selectbox_key,
    )
    if chosen != st.session_state.active_branch_id:
        switch_active_branch(chosen)
        _safe_rerun()

    st.caption(caption)

    branch = get_active_branch()

    # Chat-style wrapper
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

    # Render past messages (all editable)
    for i, m in enumerate(branch.messages):
        # all messages get the pencil
        render_message(m, i, branch, cfg, editable=True)

        edit_state = st.session_state.get("_editing")
        if edit_state and edit_state[0] == branch.id and edit_state[1] == m.id:
            edit_key = f"editbox_{m.id}"
            if edit_key not in st.session_state:
                st.session_state[edit_key] = m.content

            st.text_area("Edit message", key=edit_key, height=150)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Save & Fork", key=f"savefork_{m.id}"):
                    new_text = st.session_state.get(edit_key, "").strip()
                    if new_text:
                        new_id = fork_from_edit(branch.id, m.id, new_text)
                        if new_id and m.role == "user":
                            st.session_state["_pending_user_turn"] = True
                    st.session_state.pop(edit_key, None)
                    st.session_state["_editing"] = None
                    _safe_rerun()
            with c2:
                if st.button("✖️ Cancel", key=f"cancelfork_{m.id}"):
                    st.session_state.pop(edit_key, None)
                    st.session_state["_editing"] = None
                    _safe_rerun()

    # Pending turn trigger
    if st.session_state.get("_pending_user_turn"):
        st.session_state["_pending_user_turn"] = False
        # IMPORTANT: always use the current active branch AFTER any fork
        branch_to_run = get_active_branch()
        if branch_to_run is not None:
            run_turn_fn(cfg, branch_to_run)

    st.markdown("</div>", unsafe_allow_html=True)

    # Input
    user_input = st.chat_input(input_label, key=input_key)
    if user_input:
        add_user_message(user_input.strip())
        st.session_state["_pending_user_turn"] = True
        _safe_rerun()

# -----------------
WebCtx = Tuple[str, List[Dict[str, str]], List[str], str, Optional[str]]
#           web_brief,     web_sources,        web_queries, origin, planner_sp

def stream_agent_bubble(
    holder,
    agent: AgentConfig,
    prompt_branch: Branch,
    final_branch: Branch,
    cfg: AppConfig,
    web_ctx: Optional[WebCtx] = None,
) -> Optional[Msg]:
    """
    Stream a single agent into a live-updating bubble, append final Msg to final_branch,
    render the final bubble with render_message, and update usage counters.

    Returns the appended Msg (or None on exception).
    """
    web_brief, web_sources, web_queries, web_origin, planner_sp = web_ctx or ("", [], [], "", None)

    params = effective_params(agent, cfg)
    model  = effective_model(agent, cfg)
    system = (agent.system_prompt or "").strip() or f"You are {agent.name}."

    web_enabled = bool(cfg.web_tool.get("enabled"))
    use_web = web_enabled and bool(web_brief or web_sources or web_queries)
    extra = web_brief if use_web else ""

    prompt = build_prompt_for_agent(agent, prompt_branch, cfg, extra_preface=extra)

    debug_info = {
        "system": system,
        "prompt": prompt,
        "params": (params.__dict__ if hasattr(params, "__dict__") else params),
        "web_enabled": web_enabled,
        "web_used": use_web,
        "web_reason": (
            "disabled" if not web_enabled else
            (web_origin or ("not used" if not use_web else "used"))
        ),
        "web_queries": (web_queries or []),
        "web_brief": (web_brief or ""),
        "planner_system_prompt": (planner_sp if web_origin == "planner" else None),
        "planner_used": (web_origin == "planner"),
    }

    # placeholder bubble
    holder.markdown(
        bubble_html(Msg(role="agent", sender=agent.name, content="…"), len(final_branch.messages), cfg),
        unsafe_allow_html=True
    )

    # Determine whether to collect denormalized tokens for this run
    want_tokens = bool(
        getattr(cfg, "show_denorm_tokens", False)
        and st.session_state.get("ollama_supports_logprobs", True)
    )
    denorm_tokens: List[str] = []

    streamed_text = ""
    start_time = time.time()
    prompt_tok = 0
    completion_tok = 0
    api_thinking = ""

    try:
        for chunk in generate_stream(
            model,
            prompt,
            params,
            system=system,
            keep_alive="5m",
            think_api=agent.think_api,
            logprobs=want_tokens,
        ):
            if "response" in chunk and not chunk.get("done"):
                streamed_text += chunk["response"]
                holder.markdown(
                    bubble_html(
                        Msg(role="agent", sender=agent.name, content=streamed_text),
                        len(final_branch.messages),
                        cfg,
                    ),
                    unsafe_allow_html=True,
                )

            # collect denormalized tokens from logprobs (if requested)
            if want_tokens:
                lp = chunk.get("logprobs")
                if isinstance(lp, list):
                    for entry in lp:
                        tok = entry.get("token")
                        if isinstance(tok, str):
                            denorm_tokens.append(tok)

            if "thinking" in chunk and not chunk.get("done"):
                api_thinking += chunk.get("thinking", "")
            if chunk.get("done"):
                prompt_tok = int(chunk.get("prompt_eval_count", 0))
                completion_tok = int(chunk.get("eval_count", 0))
                break

        parsed_think, cleaned = parse_thinking_keep_tags(streamed_text)
        thinking = parsed_think
        if api_thinking.strip():
            #thinking = f"<think>\n{api_thinking.strip()}\n</think>" #Add <think> wrapper if the model doesn't have any wrapper 
            thinking = api_thinking.strip()
            cleaned = streamed_text if parsed_think is None else cleaned #no <think> wrapper 

        # Fallback: model returned only <think>…</think> and no explicit answer.
        if (not cleaned or not cleaned.strip()) and thinking:
            #import re as _re
            #cleaned = _re.sub(r"(?is)</?think>", "", thinking).strip() # strip the <think> wrappers but keep the inner reasoning
            cleaned = thinking.strip()

        dt = time.time() - start_time
        msg = Msg(
            role="agent",
            sender=agent.name,
            content=(cleaned or "").strip(),
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
            duration_s=dt,
            thinking=thinking,
            debug=debug_info,
            sources=(web_sources if use_web else []),
            denorm_tokens=denorm_tokens,
        )
        final_branch.messages.append(msg)
        idx = len(final_branch.messages) - 1

        # Overwrite the streaming placeholder with the canonical renderer,
        # so the new reply gets the same layout + pencil as all other messages.
        holder.empty()
        with holder.container():
            render_message(
                msg,
                idx,
                final_branch,
                cfg,
                editable=True,   # <- ensures the ✎ is shown immediately
            )

        # Update usage counters
        totals = st.session_state.totals
        totals["prompt_tokens"] += prompt_tok
        totals["completion_tokens"] += completion_tok
        totals["messages"] += 1
        totals["last_context_tokens"] = prompt_tok

        # Refresh usage sidebar if you have it wired this way
        from core.state import render_usage_sidebar
        render_usage_sidebar()

        return msg

    except Exception as e:
        err_msg = Msg(
            role="agent",
            sender=agent.name,
            content=f"(Error generating with {model}: {e})",
            debug=debug_info,
        )
        final_branch.messages.append(err_msg)
        idx = len(final_branch.messages) - 1

        holder.empty()
        with holder.container():
            render_message(err_msg, idx, final_branch, cfg, editable=True)

        return None
