# app/views/runtime.py

from __future__ import annotations
import time, json
from typing import List, Dict, Optional, Tuple, Callable

import streamlit as st

from core.config import AppConfig, Branch, AgentConfig, Msg
from core.state import ( is_planner, render_usage_sidebar, list_branches_for_mode, get_active_branch, switch_active_branch,
                         add_user_message, _safe_rerun, fork_from_edit, UPLOADS_DIR, render_usage_sidebar
                       )
from core.runtime_web import web_brief_for_branch
from core.agents import effective_params, effective_model
from core.prompting import build_prompt_for_agent, parse_thinking_keep_tags, build_prompt_for_summarizer
from core.ollama_client import generate_stream, generate_once, chat_once_with_tools
from core.tools import ToolContext, run_tool, ollama_tool_defs_for_agent
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

     # --- Image attachment indicator + input ---

    mode = st.session_state.get("mode", "chat")
    pending_images_key = f"_{mode}_pending_images"
    pending_images = st.session_state.get(pending_images_key, [])

    if pending_images:
        names = ", ".join(img["name"] for img in pending_images)
        st.caption(f"📎 Images attached: {names}")

    user_input = st.chat_input(input_label, key=input_key)
    if user_input:
        text = user_input.strip()

        images: list[str] = []
        if pending_images:
            # Add a simple textual reference to all filenames at the end of the message
            names = [img["name"] for img in pending_images]
            label = ", ".join(names)  # e.g. "1.png, 2.png"
            text = f"{text}\n{label}" if text else label

            images = [img["path"] for img in pending_images]

            # Clear pending images after using them
            st.session_state[pending_images_key] = []

        add_user_message(text, images=images)
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

    # Collect all unique image paths from the prompt branch so the model can "see" them
    image_paths: List[str] = []
    last_u_idx = last_user_index(prompt_branch)
    if last_u_idx is not None:
        last_user_msg = prompt_branch.messages[last_u_idx]
        for p in getattr(last_user_msg, "images", None) or []:
            if p:
                image_paths.append(p)

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
        "image_paths": image_paths,  # for Context transparency
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

    use_ollama_tools = bool(cfg.feature_flags.get("ollama_tools", False))

    # --- Path 1: native Ollama tools via /api/chat ---------------------------------
    if use_ollama_tools:
        # Build Ollama tool definitions for this agent (may be empty).
        tool_defs = ollama_tool_defs_for_agent(agent, cfg, final_branch)
        debug_info["ollama_tools_enabled"] = True
        debug_info["ollama_tool_defs"] = tool_defs
        debug_info["ollama_tools_for_agent"] = sorted( # Transparency: which tools are actually available to the model?
            [
                (td.get("function") or {}).get("name")
                for td in tool_defs
                if isinstance(td, dict)
            ]
        )

        # If no tools are actually available/schematized, fall back to the
        # normal generate_stream path.
        if tool_defs:
            try:
                # Build chat-style messages: a system message + one user message
                # containing the prompt built by build_prompt_for_agent.
                messages: list[dict] = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})

                final_contents: list[str] = []
                final_think_parts: list[str] = []

                while True:
                    data = chat_once_with_tools(
                        model=model,
                        messages=messages,
                        tools=tool_defs,
                        params=params,
                        think=True,
                        logprobs=want_tokens,
                    )

                    # (2) Handle Ollama error response — model likely doesn't support tools.
                    if data.get("error"):
                        err_text = (
                            f"Tool calling failed: {data['error']}. "
                            "This model may not support tools or the max number of tokens was reached. "
                            "Try switching to a model like 'gpt-oss:20b' that supports tools or increasing max_token and num_ctx."
                        )
                        debug_info["ollama_tools_error"] = data["error"]
                        cleaned_content = err_text
                        thinking = ""
                        dt = time.time() - start_time
                        # Create final Msg and render, then return.
                        msg = Msg(
                            role="agent",
                            sender=agent.name,
                            content=cleaned_content,
                            prompt_tokens=prompt_tok,
                            completion_tokens=completion_tok,
                            duration_s=dt,
                            thinking=thinking,
                            debug=debug_info,
                            sources=(web_sources if use_web else []),
                            denorm_tokens=[],
                        )
                        final_branch.messages.append(msg)
                        idx = len(final_branch.messages) - 1
                        holder.empty()
                        with holder.container():
                            render_message(msg, idx, final_branch, cfg, editable=True)
                        # Update usage counters
                        totals = st.session_state.totals
                        totals["prompt_tokens"] += prompt_tok
                        totals["completion_tokens"] += completion_tok
                        totals["messages"] += 1
                        totals["last_context_tokens"] = prompt_tok
                        render_usage_sidebar()
                        return msg

                    msg_data = data.get("message") or {}
                    messages.append(msg_data)

                    # THINKING
                    api_think = msg_data.get("thinking") or ""
                    if api_think:
                        final_think_parts.append(api_think)
                    
                    # CONTENT
                    content_piece = msg_data.get("content") or ""
                    if content_piece:
                        final_contents.append(content_piece)

                    # TOKEN COUNTS (top-level, same as /api/generate)
                    prompt_tok += int(data.get("prompt_eval_count", 0) or 0)
                    completion_tok += int(data.get("eval_count", 0) or 0)

                    # DENORMALIZED TOKENS (logprobs)
                    if want_tokens:
                        # different Ollama versions put logprobs either at top-level or inside the message
                        lp = data.get("logprobs") or msg_data.get("logprobs")
                        if isinstance(lp, list):
                            for entry in lp:
                                tok = entry.get("token")
                                if isinstance(tok, str):
                                    denorm_tokens.append(tok)

                    # FINISH
                    tool_calls = msg_data.get("tool_calls") or []
                    if not tool_calls:
                        # No more tools -> done.
                        break

                    # Transparency: log raw tool_calls in debug info
                    debug_info.setdefault("ollama_tool_calls", []).append(tool_calls)

                    # Execute each requested tool locally and append tool messages.
                    for tc in tool_calls:
                        fn_info = tc.get("function") or {}
                        tname = fn_info.get("name")
                        raw_args = fn_info.get("arguments") or {}

                        # Handle arguments as dict or JSON string.
                        if isinstance(raw_args, str):
                            try:
                                raw_args = json.loads(raw_args)
                            except Exception:
                                # Fallback: treat the raw string as expression.
                                raw_args = {"expression": raw_args}

                        tool_ctx = ToolContext(agent=agent, cfg=cfg, branch=final_branch)
                        tool_result = run_tool(tool_ctx, tname, raw_args)

                        # Append a tool message to the Ollama chat transcript
                        tool_payload: dict = {
                            "result": tool_result.output,
                        }
                        if tool_result.error:
                            tool_payload["error"] = tool_result.error

                        tool_msg_for_ollama = {
                            "role": "tool",
                            "tool_name": tname,
                            "content": json.dumps(tool_payload),
                        }
                        messages.append(tool_msg_for_ollama)

                        # Transparency: append a visible tool Msg to the branch.
                        tool_msg_content = (
                            tool_result.output
                            if tool_result.error is None
                            else f"(Tool error: {tool_result.error})"
                        )
                        tool_sender = tname or "Tool"
                        visible_tool_msg = Msg(role="tool", sender=tool_sender, content=tool_msg_content)
                        final_branch.messages.append(visible_tool_msg)

                        # Context inspector: log details
                        tools_used = debug_info.setdefault("tools_used", [])
                        tools_used.append(
                            {
                                "name": tname,
                                "args": raw_args,
                                "output": tool_result.output,
                                "error": tool_result.error,
                                "retrieved": getattr(tool_result, "retrieved", []),
                                "context": getattr(tool_result, "context", ""),
                            }
                        )

                # After the loop, build final content and thinking
                final_text = "".join(final_contents).strip()
                final_thinking = "\n".join(final_think_parts).strip()

                # (1) Warning if the model produced no final content at all.
                if not final_text:
                    final_text = "(Tool calling ended without a final answer. Maybe max token reached)"

                cleaned_content = final_text
                thinking = final_thinking
                dt = time.time() - start_time

                msg = Msg(
                    role="agent",
                    sender=agent.name,
                    content=cleaned_content,
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
                holder.empty()
                with holder.container():
                    render_message(msg, idx, final_branch, cfg, editable=True)

                totals = st.session_state.totals
                totals["prompt_tokens"] += prompt_tok
                totals["completion_tokens"] += completion_tok
                totals["messages"] += 1
                totals["last_context_tokens"] = prompt_tok
                render_usage_sidebar()
                return msg

            except Exception as e:
                err_msg = Msg(
                    role="agent",
                    sender=agent.name,
                    content=f"(Error generating with {model} (tools): {e})",
                    debug=debug_info,
                )
                final_branch.messages.append(err_msg)
                idx = len(final_branch.messages) - 1
                holder.empty()
                with holder.container():
                    render_message(err_msg, idx, final_branch, cfg, editable=True)
                return None

    # --- Path 2: existing /api/generate streaming  ---------------

    try:
        for chunk in generate_stream(
            model,
            prompt,
            params,
            system=system,
            keep_alive="5m",
            logprobs=want_tokens,
            images=image_paths,
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
            warn = "**[Warning: Model returned only reasoning and no explicit answer. Showing its internal thinking instead. Possibly Max tokens reached]**"
            cleaned = f"{warn}\n\n{cleaned}"

        cleaned_content = (cleaned or "").strip()
        dt = time.time() - start_time
        msg = Msg(
            role="agent", sender=agent.name,
            content=cleaned_content,
            prompt_tokens=prompt_tok, completion_tokens=completion_tok, duration_s=dt,
            thinking=thinking, debug=debug_info,
            sources=(web_sources if use_web else []), denorm_tokens=denorm_tokens,
        )
        final_branch.messages.append(msg)
        idx = len(final_branch.messages) - 1
        # Render the final agent message in the UI bubble
        holder.empty()
        with holder.container():
            render_message(msg, idx, final_branch, cfg, editable=True)
        # Update usage counters
        totals = st.session_state.totals
        totals["prompt_tokens"] += prompt_tok
        totals["completion_tokens"] += completion_tok
        totals["messages"] += 1
        totals["last_context_tokens"] = prompt_tok
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
