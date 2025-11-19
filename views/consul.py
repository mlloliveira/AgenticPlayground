# app/views/consul.py

import re, time, math, json
from collections import Counter
from typing import List, Dict, Optional

import streamlit as st

from core.state import should_speak
from core.config import AppConfig, Msg, Branch
from core.prompting import parse_thinking_keep_tags
from core.ollama_client import generate_once
from core.agents import effective_params, effective_model
from views.runtime import prepare_turn_common, stream_agent_bubble, run_summarizer_if_enabled, render_mode_scaffold, last_user_index, prompt_branch_for_round
from ui.render import render_message


# ---------------- Voting helpers ----------------

VOTE_RE = re.compile(r'(?im)^\s*vote\s*:\s*([A-Za-z0-9 _\-]+)\s*$', re.MULTILINE)

def softmax(xs: List[float], alpha: float = 1.0) -> List[float]:
    if not xs:
        return []
    scaled = [alpha * x for x in xs]
    m = max(scaled)
    exps = [math.exp(v - m) for v in scaled]
    z = sum(exps) or 1.0
    return [v / z for v in exps]

def _parse_confidence(text: str) -> float:
    m = re.search(r"(?im)^\s*confidence\s*[:\-]\s*([01](?:\.\d+)?|\.\d+)\s*$", text)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except Exception:
            pass
    m2 = re.search(r"(?<!\d)(0?\.\d+|1(?:\.0+)?)", text)
    if m2:
        try:
            return max(0.0, min(1.0, float(m2.group(1))))
        except Exception:
            pass
    return 0.5

def _agent_vote_prompt(transcript: str, agents: list[str], self_name: str) -> str:
    names = ", ".join(agents)
    others = ", ".join([a for a in agents if a != self_name]) or "(none)"
    return (
        f"You are {self_name}, a member of a council of AI debaters.\n"
        f"Participants: {names}\n\n"
        "Task: Vote for the participant with the single strongest final answer (you may NOT vote for yourself).\n"
        "Rules:\n"
        f"1) Do NOT vote for yourself ({self_name}).\n"
        f"2) Vote for exactly ONE of: {others}.\n"
        "3) First line must be exactly: VOTE: <participant name>\n"
        "4) Second line: a brief, 1–2 sentence justification.\n\n"
        "Debate transcript:\n"
        f"{transcript}\n"
    )

# ---------------- Orchestrate one consul turn ----------------

def _run_consul_streaming(cfg: AppConfig, branch: Branch):
    """
    For the latest user prompt:
      1) Turn context (shared bits).
      2) Run R rounds; agents can skip per should_speak().
      3) Voting; append a Final Answer bubble with detailed explanation (old style).
      4) Summarizer updates (silent unless visible).
    """
    # Identify current turn start (last user message)
    last_user_idx = last_user_index(branch)
    if last_user_idx is None:
        return

    # 1) Turn context (shared bits)
    # Note: For Consul we IGNORE prompt_branch from prepare_turn_common (it’s for Chat).
    _ignored_prompt_branch, agents, web_ctx = prepare_turn_common(cfg, branch)
    if not agents:
        st.info("No enabled debaters. Turn on at least one non-summarizer agent.")
        return

    rounds = int(cfg.consul.get("rounds", 3))
    voting = cfg.consul.get("voting", "majority")
    alpha  = float(cfg.consul.get("alpha", 2.0))

    # 2) R Rounds: stream bubble messages; prompt branch is recomputed EACH round.
    for r in range(1, rounds + 1):
        active = [a for a in agents if should_speak(a, r, rounds)]
        holders = [st.empty() for _ in active]

        # Round-aware context: if blind_first_turn=True: Round 1  → last user only ; Round 2+ → last user + prior rounds' replies from THIS turn. If blind_first_turn=False: full branch
        prompt_branch = prompt_branch_for_round(cfg, branch, last_user_idx)
        for ag, holder in zip(active, holders):
            stream_agent_bubble(
                holder=holder,
                agent=ag,
                prompt_branch=prompt_branch,
                final_branch=branch,
                cfg=cfg,
                web_ctx=web_ctx
            )

    # Transcript for THIS turn only (after all rounds)
    turn_msgs = branch.messages[last_user_idx+1:]
    transcript_str = "\n".join(f"{m.sender}: {m.content}" for m in turn_msgs if m.role != "user")

    # 3) Voting - Winner - Finalization
    final_author: str = "(none)"
    final_answer: str = "(no answer)"
    final_debug: Dict = {}

    if voting == "majority":
        all_names = [a.name for a in agents]
        votes = []
        for ag in agents:
            params = effective_params(ag, cfg)
            model = effective_model(ag, cfg)
            system = ag.system_prompt.strip() or f"You are {ag.name}."
            v_prompt = (
                f"You are {ag.name}. You are casting a vote at the end of a debate.\n"
                "First line must be exactly: VOTE: <participant name>\n"
                "Second line: a short justification.\n\n" +
                _agent_vote_prompt(transcript_str, all_names, self_name=ag.name)
            )
            out = generate_once(model=model, system=system, prompt=v_prompt, params=params)
            text = (out.get("content", "") or out.get("response", "") or "").strip()

            m = VOTE_RE.search(text)
            choice = (m.group(1).strip() if m else "UNKNOWN")
            # enforce: no self-vote; must be a known agent
            if (choice == ag.name) or (choice not in all_names):
                choice = "INVALID"

            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            justification = lines[1] if len(lines) >= 2 else "(no justification provided)"
            votes.append({"voter": ag.name, "choice": choice, "justification": justification, "raw": text})

        valid = [v for v in votes if v["choice"] in all_names]
        counts = Counter(v["choice"] for v in valid)

        if counts:
            final_author = counts.most_common(1)[0][0]
        else:
            # fallback: first enabled agent if no valid votes
            final_author = agents[0].name if agents else "(none)"

        # Build debug (no answers yet)
        final_debug = {
            "voting": "majority",
            "votes": votes,
            "vote_counts": dict(counts),
            "candidates": [{"agent": n} for n in all_names],
        }

    else:  # voting == "confidence"
        alpha = float(cfg.consul.get("alpha", 2.0))
        cand_conf = []

        # Ask only for confidence (no answers here)
        for ag in agents:
            params = effective_params(ag, cfg)
            model = effective_model(ag, cfg)
            system = ag.system_prompt.strip() or f"You are {ag.name}."
            conf_prompt = (
                f"[YOUR ROLE]\nYou are {ag.name}. We are concluding a council debate. Give us your self-confidence report.\n\n"
                "[TASK]\nProvide ONLY a line 'Confidence: <0..1>' and then ONE short justification line.\n"
                "Do NOT write a final answer or anything else here.\n\n"
                "[TRANSCRIPT]\n" + transcript_str
            )
            out = generate_once(model=model, system=system, prompt=conf_prompt, params=params)
            text = (out.get("content", "") or out.get("response", "") or "").strip()
            conf = _parse_confidence(text)
            cand_conf.append({"agent": ag.name, "confidence": conf, "raw": text})

        # softmax over confidences
        ws = softmax([c["confidence"] for c in cand_conf], alpha=alpha)
        for i, w in enumerate(ws):
            cand_conf[i]["weight"] = w

        if cand_conf:
            best_idx = max(range(len(cand_conf)), key=lambda i: cand_conf[i]["weight"])
            final_author = cand_conf[best_idx]["agent"]
        else:
            final_author = agents[0].name if agents else "(none)"

        final_debug = {
            "voting": "confidence",
            "alpha": alpha,
            "candidates": cand_conf,  # shown in the Context expander
        }

    # 3b) Generate a SINGLE Final Answer from the winner (now that we know who)
    winner_agent = next((a for a in agents if a.name == final_author), agents[0] if agents else None)
    if winner_agent:
        params = effective_params(winner_agent, cfg)
        model = effective_model(winner_agent, cfg)
        system = winner_agent.system_prompt.strip() or f"You are {winner_agent.name}."
        final_prompt = (
            f"[YOUR ROLE]\nYou won this debate, {winner_agent.name}, and you are tasked with concluding the council.\n\n"
            "[TASK]\nProvide the Final Answer. Be concise and self-contained.\n\n"
            "[TRANSCRIPT]\n" + transcript_str
        )
        out = generate_once(model=model, system=system, prompt=final_prompt, params=params)
        text = (out.get("content", "") or out.get("response", "") or "").strip()
        _, final_answer = parse_thinking_keep_tags(text)

        # Merge winner's prompt context into debug for the 👁 Context expander
        final_debug.update({
            "system": system,
            "prompt": final_prompt,
            "params": (params.__dict__ if hasattr(params, "__dict__") else params),
            "model": model,
            "prompt_tokens": int(out.get("prompt_eval_count", 0)),
            "completion_tokens": int(out.get("eval_count", 0)),
            "duration_s": float(out.get("_elapsed", 0.0)),
            "web_enabled": bool(cfg.web_tool.get("enabled", False)),
            "web_used": False,
            "web_reason": "Finalization uses the transcript; no web search performed.",
        })

    # 3c) Append Final Answer message (renderer shows it as a Final Answer card)
    final_msg = Msg(
        role="agent",
        sender=f"(written by {final_author})",
        content=final_answer,
        debug=final_debug,
        markdown=True,
        kind="final",
    )
    branch.messages.append(final_msg)
    render_message(final_msg, len(branch.messages) - 1, branch, cfg, editable=False)

    # 4) Summarizer
    run_summarizer_if_enabled(cfg, branch)


# ---------------- Consul Page ----------------

def consul_page():
    # Build caption text on each render
    cfg: AppConfig = st.session_state.app_config
    caption = (
        f"Mode: Consul · Model mode: "
        f"{'One model for all ('+cfg.global_model+')' if cfg.same_model_for_all else 'Per-agent models'} · "
        f"Rounds: {int(cfg.consul.get('rounds', 3))} · Voting: {cfg.consul.get('voting','majority')}"
    )

    # Use the common scaffold and hand it the mode-specific turn function
    render_mode_scaffold(
        mode="consul",
        caption=caption,
        selectbox_key="consul_branch_select",
        input_key="consul_chat_input",
        input_label="Ask the council…",
        run_turn_fn=_run_consul_streaming,
    )
