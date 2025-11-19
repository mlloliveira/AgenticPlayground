# core/state.py
import os, uuid, requests, json, copy, subprocess, re
import streamlit as st
from time import gmtime, strftime
from pathlib import Path
from typing  import Dict, List, Optional
from .config import AppConfig, AgentConfig, GenParams, Branch, Msg
from .config import Preset, preset_to_dict, preset_from_dict, DEFAULT_CHAT_AGENTS, DEFAULT_CONSUL_AGENTS
from .conversations import list_saved_conversations, save_current_branch, load_conversation_payload, delete_conversation
from core.ollama_client import list_running_models, unload_model

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

#PRESET_DIR = os.path.join(os.getcwd(), "presets")
#os.makedirs(PRESET_DIR, exist_ok=True)
# PRESETS_DIR = os.path.join("app", "presets")  # consistent with your tree
# os.makedirs(PRESETS_DIR, exist_ok=True)
PROJECT_APP_DIR = Path(__file__).resolve().parents[1]
PRESETS_DIR = PROJECT_APP_DIR / "presets"
os.makedirs(PRESETS_DIR, exist_ok=True)


# DEFAULT_AGENTS = [
#     AgentConfig(name="Agent A", system_prompt="You are Agent A, a pragmatic strategist. Be concise and practical.", is_summarizer=False),
#     AgentConfig(name="Agent B", system_prompt="You are Agent B, a critical analyst. Challenge assumptions respectfully.", is_summarizer=False),
#     AgentConfig(name="Summarizer", system_prompt="You are a neutral summarizer. Only summarize; do not add any opinions.", is_summarizer=True),
# ]

SIDEBAR_BLOCK_LABELS = { # Sidebar blocks (configurable order & visibility) 
    "MODELS": "Models",
    "WEB": "Internet Access",
    "PRESETS": "Presets",
    "GLOBAL_PARAMS": "Generation Parameters (Global)",
    "AGENTS": "Agents' Configuration",
    "UI": "UI & Tools",
    "CONSUL": "Consul defaults",
    "SAVE_RESET": "Save & Reset",
    "USAGE": "Usage",
}

ALL_SIDEBAR_BLOCKS = list(SIDEBAR_BLOCK_LABELS.keys())


def _sort_agents_for_display(agents):
    # Always push summarizer and web planner to the end
    return sorted(
        agents,
        key=lambda a: (
            a.is_summarizer or getattr(a, "role_tag", "") == "web_planner",
            a.name.lower(),
        )
    )

 
def _branch_belongs_to_mode(branch_id: str, mode: str) -> bool: # --- Mode/branch helpers ---
    """Return True if a branch id is scoped to a given mode."""
    return branch_id.startswith(f"main-{mode}") or branch_id.startswith(f"{mode}-")

def _next_branch_id_for_mode(mode: str) -> tuple[str, str]: # --- Mode/branch helpers ---
    """
    Compute next incremental id/label for a fork in this mode.
    e.g., ('chat-3', 'Chat #3') or ('consul-2', 'Consul #2')
    """
    prefix = f"{mode}-"
    nmax = 0
    for bid in st.session_state.branches.keys():
        if bid.startswith(prefix):
            suf = bid[len(prefix):]
            if suf.isdigit():
                nmax = max(nmax, int(suf))
    nxt = nmax + 1
    bid = f"{prefix}{nxt}"
    label = f"{mode.capitalize()} #{nxt}"
    return bid, label

def list_branches_for_mode(mode: str) -> list[str]: # --- Mode/branch helpers ---
    """Return branch ids belonging to a mode, sorted with main first, then numeric suffix."""
    brs = st.session_state.branches

    def belongs(bid: str) -> bool:
        return bid.startswith(f"main-{mode}") or bid.startswith(f"{mode}-")

    ids = [bid for bid in brs.keys() if belongs(bid)]
    if not ids:
        # Safety: ensure main exists if called very early
        main_id = f"main-{mode}"
        brs[main_id] = Branch(id=main_id, label=main_id)
        ids = [main_id]

    def sort_key(bid: str):
        if bid == f"main-{mode}":
            return (0, 0)
        prefix = f"{mode}-"
        if bid.startswith(prefix):
            suf = bid[len(prefix):]
            if suf.isdigit():
                return (1, int(suf))
        # fallback: put any odd labels at the end, stable by name
        return (2, bid)

    ids.sort(key=sort_key)
    return ids

def _set_single_main_branch_for_mode(
    mode: str,
    branch: Optional[Branch] = None,
    totals: Optional[Dict[str, int]] = None,
) -> None:
    """
    For a given mode ('chat' or 'consul'):

      • Delete ALL branches that belong to that mode.
      • Install exactly one branch: id == f'main-{mode}'.
      • Optionally overwrite usage totals.

    This is the core primitive used by:
      • Reset Chat (fresh empty main branch)
      • Load conversation (hydrate main branch from file)
      • Edge-cases in Reset Active Branch.
    """
    brs = st.session_state.branches

    # Drop all branches for this mode
    to_delete = [bid for bid in list(brs.keys()) if _branch_belongs_to_mode(bid, mode)]
    for bid in to_delete:
        del brs[bid]

    main_id = f"main-{mode}"

    # Create or adapt the branch that will become main
    if branch is None:
        branch = Branch(id=main_id, label="main")
    else:
        branch.id = main_id
        branch.label = "main"  # <--- key change: ignore old label like "Chat #2"

    brs[main_id] = branch
    st.session_state.active_branch_id = main_id

    # Keep legacy pointer for chat
    if mode == "chat":
        st.session_state.root_branch_id = main_id

    # Usage totals (default to zeroed)
    base_totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "messages": 0,
        "last_context_tokens": 0,
    }
    if totals is None:
        st.session_state.totals = dict(base_totals)
    else:
        merged = {}
        for k, default in base_totals.items():
            try:
                merged[k] = int(totals.get(k, default))
            except Exception:
                merged[k] = default
        st.session_state.totals = merged


def reset_chat_for_current_mode() -> None:
    """
    Public helper: “Reset Chat” behaviour.

    • Deletes ALL branches for the current mode.
    • Leaves a single fresh main branch (no messages, empty summary).
    • Resets usage totals.
    """
    mode = st.session_state.get("mode", "chat")
    _set_single_main_branch_for_mode(mode, branch=None, totals=None)


def reset_active_branch_for_current_mode() -> None:
    """
    Public helper: “Reset active branch” behaviour.

    • If the active branch is a NON-main branch for the current mode:
          - delete that branch object entirely
          - switch back to main-<mode>
          - leave usage totals untouched

    • If the active branch IS the main branch:
          - treat this like a full reset for that mode
            (delete all branches for that mode, new empty main, reset totals)
    """
    mode = st.session_state.get("mode", "chat")
    active_id = st.session_state.active_branch_id
    brs = st.session_state.branches

    if not _branch_belongs_to_mode(active_id, mode):
        # Nothing to do; active branch isn't even in this mode
        return

    main_id = f"main-{mode}"

    if active_id == main_id:
        # Deleting the main branch effectively means "start over"
        reset_chat_for_current_mode()
        return

    # Delete only this branch
    if active_id in brs:
        del brs[active_id]

    # Fallback to main branch for this mode
    if main_id not in brs:
        brs[main_id] = Branch(id=main_id, label=main_id)
    st.session_state.active_branch_id = main_id
    # Note: usage totals are NOT reset here on purpose.

def is_planner(agent) -> bool:
    try:
        if getattr(agent, "is_summarizer", False):
            return False
        ov = getattr(agent, "params_override", {}) or {}
        if ov.get("_role") == "planner":
            return True
        return (agent.name or "").strip().lower() == "web planner"
    except Exception:
        return False
    
def ensure_planner_agent(cfg):
    if any(is_planner(a) for a in cfg.agents):
        return
    show_time=True
    if show_time:
        now = strftime("%Y-%m-%d", gmtime())
        today_prompt = (f"Today is: {now}. Time format: YYYY-MM-DD. ")
    from .config import AgentConfig
    planner = AgentConfig(
        name="Web Planner",
        system_prompt=(today_prompt+"You are a web-query planner. Given a user prompt, produce 0–3 specific, "
        "web-searchable queries. Do not answer the questions or the queries yourself. Prepare the "
        "queries to be loaded into a web search engine. You can give an empty array if believe it is "
        "not a searchable question.  Do not use punctuation. Output STRICT JSON only: {\"queries\": [\"...\", \"...\"]}"),
        # system_prompt=("You are a web-query planner. Given a user prompt, produce 1–5 specific, "
        #                "web-searchable queries. Do not answer the questions or the queries yourself. "
        #                "Prepare the queries to be loaded into a web search engine. Do not use punctuation. "
        #                "Output STRICT JSON only: {\"queries\": [\"...\", \"...\"]}"),
        model=cfg.global_model,
        enabled=True,                           # default ON
        params_override={"_role": "planner"},   # marker
        think_api=False,
        allow_web=False,                        # planner itself does not fetch
        is_summarizer=False,
    )
    # insert directly under Summarizer
    i = next((k for k,a in enumerate(cfg.agents) if getattr(a, "is_summarizer", False)), None)
    if i is not None:
        cfg.agents.insert(i+1, planner)
    else:
        cfg.agents.append(planner)

def _safe_rerun():
    try: st.rerun()
    except Exception:
        try: st.experimental_rerun()
        except Exception: pass

def _sidebar_expanded_for(cfg: AppConfig, block_id: str, default: bool | None = None) -> bool:
    """Return whether a sidebar block should be expanded by default."""
    exp_map = getattr(cfg, "sidebar_expanded", None) or {}

    if default is None:
        default = SIDEBAR_EXPANDED_DEFAULTS.get(block_id, True)

    return bool(exp_map.get(block_id, default))


def list_ollama_models() -> List[str]:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2.5)
        r.raise_for_status()
        data = r.json()
        return sorted([m["name"] for m in data.get("models", [])])
    except Exception:
        return ["llama3:8b", "llama2:7b", "mistral", "phi3", "qwen2"]

def ensure_state():
    if "view" not in st.session_state:
        st.session_state["view"] = "playground"  # "playground" | "settings"
    if "mode" not in st.session_state:
        st.session_state["mode"] = "chat"

    # --- AppConfig bootstrap ---
    if "app_config" not in st.session_state:
        # 1) Start from compiled-in defaults (single source of truth in core.config)
        cfg = AppConfig()
        st.session_state.app_config = cfg

        # 2) If a default preset exists (presets/default.json), overlay it once
        #    We deliberately ignore errors here so a corrupted default.json
        #    never prevents the app from starting.
        try:
            load_preset("default", cfg)  # returns False if file is missing
        except Exception:
            pass

    cfg = st.session_state.app_config
    # Prime the widget state once from the config, if not already set | Avoids warning
    if "same_model_for_all" not in st.session_state:
        st.session_state["same_model_for_all"] = cfg.same_model_for_all

    # Always guarantee we have a Web Planner and keep agent order stable
    ensure_planner_agent(cfg)
    cfg.agents = _sort_agents_for_display(cfg.agents)

    # --- Branches per mode (chat/consul) ---
    if "branches" not in st.session_state:
        branches: Dict[str, Branch] = {}
        branches["main-chat"] = Branch(id="main-chat", label="main-chat")
        branches["main-consul"] = Branch(id="main-consul", label="main-consul")
        st.session_state.branches = branches
        st.session_state.active_branch_id = "main-chat"
        st.session_state.root_branch_id = "main-chat"  # legacy pointer (can be deleted later)
    else:
        # migrate existing sessions gracefully: ensure both mains exist
        br = st.session_state.branches
        if "main-chat" not in br:
            br["main-chat"] = Branch(id="main-chat", label="main-chat")
        if "main-consul" not in br:
            br["main-consul"] = Branch(id="main-consul", label="main-consul")

    # --- Models cache ---
    if "models_cache" not in st.session_state:
        st.session_state.models_cache = list_ollama_models()

    # --- Misc session flags ---
    if "_editing" not in st.session_state:
        st.session_state["_editing"] = None
    if "_autorun_fork" not in st.session_state:
        st.session_state["_autorun_fork"] = None

    # --- Usage totals ---
    if "totals" not in st.session_state:
        st.session_state.totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "messages": 0,
            "last_context_tokens": 0,
        }

    # --- Consul state ---
    if "consul_state" not in st.session_state:
        st.session_state.consul_state = {"transcript": [], "final": None, "votes": []}


# core/state.py

def ensure_mode(mode: str, cfg):
    """
    Rebuild the agent list whenever the user switches mode.
    We always keep utility agents (summarizer + web planner),
    and then append the mode-specific default agents.
    """
    import copy

    # detect mode change
    last_mode = st.session_state.get("_active_mode")
    if last_mode == mode:
        # same mode as last time → do nothing
        return

    st.session_state["_active_mode"] = mode

    # keep current utility agents (summarizer + planner) if they exist
    utils = [copy.deepcopy(a) for a in cfg.agents
             if getattr(a, "is_summarizer", False) or is_planner(a)]

    # load mode-specific defaults
    if mode == "chat":
        base = copy.deepcopy(DEFAULT_CHAT_AGENTS)
    elif mode == "consul":
        base = copy.deepcopy(DEFAULT_CONSUL_AGENTS)
    else:
        base = []

    # rebuild roster: utilities first, then mode agents
    cfg.agents = utils + base

    # make sure a planner exists and sits under the summarizer
    ensure_planner_agent(cfg)

    # Keep planner “enabled” in lockstep with Internet toggle
    for a in cfg.agents:
        if is_planner(a):
            a.enabled = bool(cfg.web_tool.get("enabled", False))

     # When entering a mode, auto-switch to that mode's main branch
    if not _branch_belongs_to_mode(st.session_state.active_branch_id, mode):
        st.session_state.active_branch_id = f"main-{mode}"

    # keep visual order stable: normal agents first, then silent ones
    cfg.agents = _sort_agents_for_display(cfg.agents)
    st.rerun()


def should_speak(agent: AgentConfig, round_idx: int, total_rounds: int) -> bool: ##
    s = getattr(agent, "schedule", None)
    if not s: return True
    pol = (s.policy or "always").lower()
    if pol == "always": return True
    if pol == "never": return False
    if pol == "first": return round_idx == 1
    if pol == "last": return round_idx == total_rounds
    if pol == "every_n":
        start = s.start_round or 1
        step = max(1, s.every_n or 1)
        return round_idx >= start and (round_idx - start) % step == 0
    if pol == "range":
        start = s.start_round or 1
        end = s.end_round or total_rounds
        return start <= round_idx <= end
    return True

def sanity_checks():
    # Basic connectivity
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=1.0)
        r.raise_for_status()
    except Exception:
        st.warning(
            f"Could not reach Ollama at {OLLAMA_HOST}. "
            "Ensure `ollama serve` is running and a model is pulled."
        )

    # --- Ollama version check for logprobs / token view ---
    supports_logprobs = False
    try:
        # Try the CLI: `ollama --version`
        res = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=1.0,
        )
        ver_text = (res.stdout or "") + (res.stderr or "")
        m = re.search(r"(\d+)\.(\d+)\.(\d+)", ver_text)
        if m:
            major, minor, patch = map(int, m.groups())
            supports_logprobs = (major, minor, patch) >= (0, 12, 11)
            if not supports_logprobs:
                st.warning(
                    f"Ollama {major}.{minor}.{patch} detected. "
                    "Denormalized token view requires Ollama 0.12.11 or newer; "
                    "the token display feature will be disabled."
                )
        else:
            st.info(
                "Could not detect Ollama version from `ollama --version`; "
                "denormalized token view may not be available."
            )
    except Exception:
        # If CLI is missing or fails, just record that we don't know
        st.info(
            "Could not run `ollama --version`; "
            "denormalized token view may not be available."
        )
        supports_logprobs = False

    # Cache this so we can disable the checkbox and avoid sending logprobs
    st.session_state["ollama_supports_logprobs"] = supports_logprobs


def get_active_branch() -> Branch:
    return st.session_state.branches[st.session_state.active_branch_id]

def switch_active_branch(branch_id: str):
    st.session_state.active_branch_id = branch_id

def add_user_message(text: str):
    get_active_branch().messages.append(Msg(role="user", sender="User", content=text))

def delete_from_here(branch_id: str, msg_id: str):
    branch = st.session_state.branches[branch_id]
    idx = next((i for i,m in enumerate(branch.messages) if m.id == msg_id), None)
    if idx is not None:
        branch.messages = branch.messages[:idx]

# core/state.py

def fork_from_edit(branch_id: str, msg_id: str, new_text: str) -> Optional[str]:
    """
    Create a new *mode-scoped* branch based on editing a single message.

    Behavior:
      - If the edited message is a USER:
          * copy all messages BEFORE it,
          * replace that message with an edited user message,
          * DROP all later messages (branch diverges; caller should re-run agents).
      - If the edited message is an AGENT (or anything else):
          * copy ALL messages,
          * replace only that message's content (preserve later messages).

    The caller decides whether to re-run agents (we do that only for user edits).
    """
    brs = st.session_state.branches
    src = brs.get(branch_id)
    if not src or not new_text.strip():
        return None

    # Find the message index
    idx = next((i for i, m in enumerate(src.messages) if getattr(m, "id", None) == msg_id), None)
    if idx is None:
        return None

    old = src.messages[idx]
    mode = st.session_state.get("mode", "chat")
    new_id, new_label = _next_branch_id_for_mode(mode)

    new_branch = Branch(id=new_id, label=new_label)

    if old.role == "user":
        # keep everything before the edited user message
        new_branch.messages = src.messages[:idx]
        # insert edited user message
        edited = Msg(
            role="user",
            sender=old.sender or "User",
            content=new_text.strip(),
            markdown=getattr(old, "markdown", False),
            thinking=getattr(old, "thinking", None),
        )
        new_branch.messages.append(edited)
        # (no later messages — they'll be regenerated on this branch)
    else:
        # copy all messages, but replace this one in-place
        new_branch.messages = list(src.messages)
        edited = Msg(
            role=old.role,
            sender=old.sender,
            content=new_text.strip(),
            markdown=getattr(old, "markdown", False),
            thinking=getattr(old, "thinking", None),
        )
        new_branch.messages[idx] = edited

    brs[new_id] = new_branch
    st.session_state.active_branch_id = new_id
    return new_id




# --- Sidebar blocks ---
SIDEBAR_EXPANDED_DEFAULTS = { # Default "expanded" state for each sidebar block
    "MODELS": True,
    "WEB": False,
    "PRESETS": False,
    "GLOBAL_PARAMS": False,
    "AGENTS": True,
    "UI": False,
    "CONSUL": False,
    "SAVE_RESET": False,
    "USAGE": True,  
}


def sidebar_block_models(cfg: AppConfig):
    with st.expander("Models", expanded=_sidebar_expanded_for(cfg, "MODELS", default=SIDEBAR_EXPANDED_DEFAULTS["MODELS"])):
        colA, colB = st.columns([0.65, 0.35])
        with colA:
            cfg.same_model_for_all = st.checkbox(
                "Use same model for all agents",
                key="same_model_for_all",
                help="When enabled, all agents share the same base model.",
                #value=True, #Streamlit will use st.session_state["same_model_for_all"]
            )
        with colB:
            if st.button("↺ Refresh models", key="refresh_models",
                         help="Query Ollama for the latest list of available models from Ollama."):
                st.session_state.models_cache = list_ollama_models()

        models = st.session_state.get("models_cache", []) or ["llama3:8b"]

        cfg.global_model = st.selectbox(
            "Global model",
            options=models,
            key="global_model_select",
            help="Base model used when 'Use same model for all agents' is ON.",
        )

        # Clean VRAM
        st.markdown("#### Running models")
        try:
            running = list_running_models()  # [{'name': 'llama3:8b', ...}, ...]
            if not running:
                st.caption("No models currently loaded.")
            else:
                for i, m in enumerate(running):
                    name = m.get("name", "(unknown)")
                    c1, c2 = st.columns([0.75, 0.25])
                    with c1:
                        st.write(name)
                    with c2:
                        if st.button("Unload", key=f"unload_{name}_{i}",
                                     help="Ask Ollama to unload this model from VRAM."):
                            try:
                                unload_model(name)
                                st.success(f"Unloaded {name}")
                            except Exception as e:
                                st.error(f"Failed to unload {name}: {e}")
        except Exception as e:
            st.error(f"Could not query running models: {e}")

def sidebar_block_web(cfg: AppConfig):
    with st.expander("Internet Access", expanded=_sidebar_expanded_for(cfg, "WEB", default=SIDEBAR_EXPANDED_DEFAULTS["WEB"])):
        cfg.web_tool["enabled"] = st.checkbox(
            "Enable internet (applies to all agents)",
            value=bool(cfg.web_tool.get("enabled", False)),
            key="enable_web_tool",
            help="When ON, the Web Planner auto-generates queries (or start your prompt with `#web:` for manual queries).",
        )
        if cfg.web_tool["enabled"]:
            cfg.web_tool["ddg_results"] = st.slider(
                "Web: results per query",
                1, 10, int(cfg.web_tool.get("ddg_results", 5)),
                1,
                key="ddg_results",
                help="Number of web search hits per query to feed into the agents.",
            )
            cfg.web_tool["max_chars"] = st.number_input(
                "Web: summary max chars",
                value=int(cfg.web_tool.get("max_chars", 8000)),
                min_value=500,
                key="web_max_chars",
                help="Maximum characters in the web brief passed in front of the agents.",
            )

        # Keep planner enabled state in sync with internet toggle
        prev_web = st.session_state.get("_prev_web_enabled")
        web_now = bool(cfg.web_tool.get("enabled", False))
        if prev_web is None or prev_web != web_now:
            st.session_state["_prev_web_enabled"] = web_now
            for a in cfg.agents:
                if is_planner(a):
                    a.enabled = web_now

def sidebar_block_presets(cfg: AppConfig):
    with st.expander("Agent Presets", expanded=_sidebar_expanded_for(cfg, "PRESETS", default=SIDEBAR_EXPANDED_DEFAULTS["PRESETS"])):

        # PRE-RESET HOOK: run before the widget is created
        if st.session_state.get("_reset_preset_pick"):
            st.session_state.pop("_reset_preset_pick")
            st.session_state["preset_pick"] = "(select)"

        new_name = st.text_input(
            "Save current agents as preset",
            key="preset_name",
            help="Give a name to save the current agents + parameters as a preset JSON file.",
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save Preset", help="Write the current setup to a preset file you can reload later.") and new_name.strip():
                if save_preset(new_name.strip(), st.session_state.app_config):
                    st.success("Preset saved")
                    st.session_state["_last_saved_preset"] = new_name.strip()
                    _safe_rerun()
        with c2:
            allp = list_presets()
            pick = st.selectbox(
                "Load preset",
                ["(select)"] + allp,
                key="preset_pick",
                help="Load a previously saved preset (agents + parameters).",
            )

            if pick != "(select)":
                already = st.session_state.get("_loaded_preset_once")
                if already != pick:
                    if load_preset(pick, st.session_state.app_config):
                        st.session_state["_loaded_preset_once"] = pick
                        st.success(f"Loaded preset '{pick}'")
                        # CRITICAL: don't set preset_pick here; defer to next run
                        st.session_state["_reset_preset_pick"] = True
                        _safe_rerun()
                else:
                    # selection persisted after a rerun; quietly clear on next run
                    st.session_state["_reset_preset_pick"] = True

def sidebar_block_global_params(cfg: AppConfig):
    with st.expander("Generation Parameters (Global)", expanded=_sidebar_expanded_for(cfg, "GLOBAL_PARAMS", default=SIDEBAR_EXPANDED_DEFAULTS["GLOBAL_PARAMS"])):
        gp = cfg.global_params
        a, b = st.columns(2)
        with a:
            gp.temperature = st.slider(
                "Temperature", 0.0, 2.0, gp.temperature, 0.05, key="g_temp",
                help="Higher = more random; lower = more deterministic.",
            )
            gp.top_k = st.number_input(
                "top_k", 0, 2048, int(gp.top_k), 10, key="g_topk",
                help="Limit sampling to the top-K tokens (0 = disabled).",
            )
            gp.num_ctx = st.number_input(
                "Context window (num_ctx)", 256, 131072, int(gp.num_ctx), 256, key="g_numctx",
                help="Maximum tokens of context to keep in the window.",
            )
        with b:
            gp.top_p = st.slider(
                "top_p", 0.0, 1.0, gp.top_p, 0.01, key="g_topp",
                help="Nucleus sampling: consider tokens whose cumulative probability ≤ top_p.",
            )
            gp.max_tokens = st.number_input(
                "Max tokens (response)", 16, 32768, int(gp.max_tokens), 16, key="g_maxtok",
                help="Maximum length of a single response.",
            )
            gp.seed = st.text_input(
                "Seed (optional)",
                value="" if gp.seed is None else str(gp.seed),
                key="g_seed",
                help="Fix this to make generations reproducible. Leave empty for random.",
            )
        cfg.global_params = gp.clamped()

def sidebar_block_agents(cfg: AppConfig):
    with st.expander("Agents' Configuration", expanded=_sidebar_expanded_for(cfg, "AGENTS", default=SIDEBAR_EXPANDED_DEFAULTS["AGENTS"])):
        # Create new agent
        with st.expander("➕ New agent", expanded=False):
            with st.form("add_agent_form", clear_on_submit=True):
                new_name = st.text_input(
                    "New agent name",
                    key="new_agent_name",
                    help="Label for the new agent.",
                )
                new_sys = st.text_area(
                    "New agent system prompt",
                    height=80,
                    key="new_agent_sys",
                    help="System prompt / instructions for this agent.",
                )
                if st.form_submit_button("Add agent", help="Insert a new agent into the current mode."):
                    if new_name.strip():
                        idx_util = next(
                            (i for i, a in enumerate(cfg.agents)
                             if a.is_summarizer or getattr(a, "params_override", {}).get("_role") == "planner"),
                            len(cfg.agents)
                        )
                        cfg.agents.insert(idx_util, AgentConfig(
                            name=new_name.strip(),
                            system_prompt=new_sys.strip() or f"You are {new_name.strip()}",
                            is_summarizer=False
                        ))
                        cfg.agents = _sort_agents_for_display(cfg.agents)
                        st.success(f"Added agent '{new_name.strip()}'")

        # Existing agents
        for i, agent in enumerate(_sort_agents_for_display(cfg.agents)):
            label = f"{agent.name}" + (" (summarizer)" if agent.is_summarizer else "")
            with st.expander(label, expanded=(i < 2)):
                agent.name = st.text_input(
                    "Agent name",
                    value=agent.name,
                    key=f"name_{agent.uid}",
                    help="Display name used in the transcript.",
                )

                if is_planner(agent):
                    # Mirror Internet toggle and make it read-only for transparency
                    agent.enabled = bool(cfg.web_tool.get("enabled", False))
                    st.checkbox(
                        "Enabled",
                        value=agent.enabled,
                        key=f"en_{agent.uid}",
                        help="The Web Planner follows the global Internet toggle and cannot be toggled here.",
                        disabled=True,
                    )
                else:
                    agent.enabled = st.checkbox(
                        "Enabled",
                        value=agent.enabled,
                        key=f"en_{agent.uid}",
                        help="Turn this agent on or off for the current mode.",
                    )

                if not cfg.same_model_for_all:
                    all_models = st.session_state.get("models_cache", []) or ["llama3:8b"]
                    idx = all_models.index(agent.model) if agent.model in all_models else 0
                    agent.model = st.selectbox(
                        "Model",
                        all_models,
                        index=idx,
                        key=f"mdl_{agent.uid}",
                        help="Override the global model for this agent.",
                    )

                agent.system_prompt = st.text_area(
                    "System prompt",
                    value=agent.system_prompt,
                    height=120,
                    key=f"sp_{agent.uid}",
                    help="Instructions / persona for this agent.",
                )

                cols = st.columns(2) #(3)
                # with cols[0]:
                #     agent.think_api = st.checkbox(
                #         "Thinking API",
                #         value=agent.think_api,
                #         key=f"think_{agent.uid}",
                #         help="When supported by the model, capture hidden 'thinking' traces.",
                #     )
                with cols[0]: #[1]
                    pass
                with cols[1]: #[2]
                    if not agent.is_summarizer:
                        if not getattr(agent, "is_planner", False):
                            if st.button(
                                "Remove agent",
                                key=f"rm_{agent.uid}",
                                help="Delete this agent from the current configuration.",
                            ):
                                del cfg.agents[i]
                                st.success("Removed agent")
                                _safe_rerun()
                        else:
                            st.caption("Planner can be disabled via the Internet toggle, not removed.")

def sidebar_block_ui(cfg: AppConfig):
    with st.expander("UI Tools", expanded=_sidebar_expanded_for(cfg, "UI", default=SIDEBAR_EXPANDED_DEFAULTS["UI"])):
        cfg.blind_first_turn = st.checkbox(
            "Blind first turn",
            value=cfg.blind_first_turn,
            key="blind_first_turn",
            help="When ON, same-turn replies are not shared among agents (they only see previous turns).",
        )
        cfg.markdown_all = st.checkbox(
            "Render all messages in Markdown",
            value=bool(getattr(cfg, "markdown_all", False)),
            key="markdown_all_toggle",
            help="When ON, all messages render as Markdown cards.",
        )
        cfg.show_thinking = st.checkbox(
            "Show the thinking for models that support reasoning",
            value=getattr(cfg, "show_thinking", False),
            key="show_thinking_toggle",
            help=(
                "When enabled, the model's internal reasoning (e.g. <think>...</think> "
                ") is shown in the Context panel."
            ),
        )
        supports_logprobs = st.session_state.get("ollama_supports_logprobs", True)
        cfg.show_denorm_tokens = st.checkbox(
            "Show denormalized output tokens",
            value=getattr(cfg, "show_denorm_tokens", False),
            key="show_denorm_tokens_toggle",
            help=(
                "When enabled, new assistant messages show the model's "
                "denormalized (without special tokens and chars) output tokens in the Context panel."
            ),
            disabled=not supports_logprobs,
        )
        if not supports_logprobs:
            st.caption("Requires Ollama ≥ 0.12.11 (logprobs support).")

def sidebar_block_consul(cfg: AppConfig):
    with st.expander("Consul defaults", expanded=_sidebar_expanded_for(cfg, "CONSUL", default=SIDEBAR_EXPANDED_DEFAULTS["CONSUL"])):
        cfg.consul["rounds"] = st.number_input(
            "Rounds",
            1, 10, int(cfg.consul.get("rounds", 3)),
            key="consul_rounds",
            help="How many debate rounds the Consul should run before voting.",
        )
        voting_opts = ["majority", "confidence", "judge"]
        cur_v = cfg.consul.get("voting", "majority")
        if cur_v not in voting_opts:
            cur_v = "majority"
        cfg.consul["voting"] = st.selectbox(
            "Voting",
            voting_opts,
            index=voting_opts.index(cur_v),
            key="consul_voting",
            help="How the council chooses a single Final Answer.",
        )
        # You can re-enable alpha / auto_web_brief as needed here later.

def sidebar_block_save_reset(cfg: AppConfig):
    """
    Save & Reset block:

      • Reset Chat      → delete all branches for current mode, fresh main-<mode>.
      • Reset Active    → delete just the active branch (or full reset if it's main).
      • Save branch     → encrypts to disk.
      • Load conv       → installs loaded branch as main-<mode>, dropping other branches.
    """
    with st.expander("Save, Load & Reset", expanded=_sidebar_expanded_for(cfg, "SAVE_RESET", default=SIDEBAR_EXPANDED_DEFAULTS["SAVE_RESET"])):
        st.markdown("**Reset Conversation**")
        #with st.expander("Reset Conversation", expanded=True):
        # --- Reset buttons (two side-by-side) ---
        col_all, col_branch = st.columns(2)

        # Reset ALL branches for the current mode
        with col_all:
            if st.button(
                "🧹 Reset Chat",
                key="reset_chat_btn",
                help="Clear all messages in all branches for the current mode (e.g. Chat and Consul).",
                use_container_width=True,
            ):
                reset_chat_for_current_mode()
                st.success("Fully reset.")
                _safe_rerun()

        # Reset ONLY the active branch 
        with col_branch:
            if st.button(
                "🧹 Reset Active Branch",
                key="reset_branch_btn",
                help="Clear messages just in the current branch.",
                use_container_width=True,
            ):
                reset_active_branch_for_current_mode()
                st.success("Active branch reset.")
                _safe_rerun()


        # --- Save current branch ---
        st.markdown("**Save Conversation**")
        #with st.expander("Save Conversation", expanded=True):
        name = st.text_input(
            "Conversation name",
            key="save_conv_name",
            help="Give a name to save the current conversation (branch).",
        )
        if st.button(
            "💾 Save conversation (branch)",
            key="save_conv_btn",
            help="Encrypt and save the current conversation (branch) to a local folder.",
            use_container_width=True,
        ):
            if not name.strip():
                st.warning("Please choose a name before saving.")
            else:
                ok, msg = save_current_branch(name.strip())
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)


        # --- Load conversation ---
        st.markdown("**Load Conversation**")
        #with st.expander("Load Conversation", expanded=True):
        all_names = list_saved_conversations()
        if not all_names:
            st.caption("No saved conversations yet.")
        else:   
            pick = st.selectbox(
                "Saved conversation",
                options=["(select)"] + all_names,
                key="load_conv_pick",
            )
            col_del,col_load = st.columns([0.5, 0.5])
            with col_del:
                if pick != "(select)" and st.button(
                    "🗑 Delete",
                    key="delete_conv_btn",
                    help="Remove this saved conversation from disk.",
                    use_container_width=True,
                ):
                    delete_conversation(pick)
                    st.success(f"Deleted saved conversation “{pick}”.")
                    _safe_rerun()
            with col_load:
                if pick != "(select)" and st.button(
                    "📂 Load",
                    key="load_conv_btn",
                    help="Load the conversation (as the main branch).",
                    use_container_width=True,
                ):
                    payload = load_conversation_payload(pick)
                    if payload is None:
                        st.error("Could not load this conversation (file missing or corrupted).")
                    else:
                        saved_mode, branch, totals = payload

                        # 1. Switch mode to the mode of the saved conversation
                        st.session_state["mode"] = saved_mode

                        # 2. Install it as the ONLY branch for that mode → main-<mode>
                        _set_single_main_branch_for_mode(saved_mode, branch=branch, totals=totals)

                        st.success(f"Loaded “{pick}” into main-{saved_mode}.")
                        _safe_rerun()

def sidebar_block_usage(cfg: AppConfig):
    with st.expander("Usage", expanded=_sidebar_expanded_for(cfg, "USAGE", default=SIDEBAR_EXPANDED_DEFAULTS["USAGE"])):
        # Put the usage panel here
        placeholder = st.empty()
        st.session_state["_usage_ph"] = placeholder
        render_usage_sidebar()


SIDEBAR_BLOCK_RENDERERS = {
    "MODELS": sidebar_block_models,
    "WEB": sidebar_block_web,
    "PRESETS": sidebar_block_presets,
    "GLOBAL_PARAMS": sidebar_block_global_params,
    "AGENTS": sidebar_block_agents,
    "UI": sidebar_block_ui,
    "CONSUL": sidebar_block_consul,
    "SAVE_RESET": sidebar_block_save_reset,
    "USAGE": sidebar_block_usage,
}

# --- Sidebar usage placeholder ---
def render_usage_sidebar():
    ph = st.session_state.get("_usage_ph")
    if ph is None: return
    t = st.session_state.totals
    ctx = int(st.session_state.app_config.global_params.num_ctx)
    with ph.container():
        st.markdown("### Usage")
        st.metric("Messages", t["messages"])
        st.metric("Tokens (prompt+completion)", f"{t['prompt_tokens']} + {t['completion_tokens']}")
        st.progress(min(t["last_context_tokens"] / max(ctx, 1), 1.0))
        st.caption(f"Context: {t['last_context_tokens']} / {ctx}")

def sidebar_once():
    cfg: AppConfig = st.session_state.app_config
    preset_ui = st.session_state.pop("_apply_preset_ui", None)
    if preset_ui:
        gp = preset_ui.get("global_params", {})
        gm = preset_ui.get("global_model")

        # ensure model is in the options list
        cache = st.session_state.get("models_cache", []) or ["llama3:8b"]
        if gm and gm not in cache:
            st.session_state["models_cache"] = cache + [gm]

        # set widget keys BEFORE creating widgets 
        st.session_state["same_model_for_all"] = preset_ui.get(
            "same_model_for_all",
            st.session_state.get("same_model_for_all", cfg.same_model_for_all),
        )
        if gm:
            st.session_state["global_model_select"] = gm

        # global params keys
        st.session_state["g_temp"]   = float(gp.get("temperature", 0.7))
        st.session_state["g_topp"]   = float(gp.get("top_p", 0.9))
        st.session_state["g_topk"]   = int(gp.get("top_k", 50))
        st.session_state["g_maxtok"] = int(gp.get("max_tokens", 512))
        st.session_state["g_numctx"] = int(gp.get("num_ctx", 4096))
        st.session_state["g_seed"]   = "" if gp.get("seed", None) in ("", None, "None") else str(gp.get("seed"))

    with st.sidebar:
        # --- Mode / Settings bar (3 tabs) ---
        st.markdown("**Mode**")

        TABS = [
            ("chat", "Chat"),
            ("consul", "Consul"),
            ("settings", "All Settings"),
        ]

        cur_view = st.session_state.get("view", "playground")
        cur_mode = st.session_state.get("mode", "chat")

        # Active tab is derived purely from view + mode
        if cur_view == "settings":
            active_tab = "settings"
        else:
            active_tab = cur_mode if cur_mode in ("chat", "consul") else "chat"

        cols = st.columns(len(TABS))
        for (code, label), col in zip(TABS, cols):
            is_active = (code == active_tab)
            btn_label = f"● {label}" if is_active else label
            if col.button(
                btn_label,
                key=f"mode_btn_{code}",
                use_container_width=True,
                help=(
                    "Open the full configuration panel."
                    if code == "settings"
                    else f"Switch to {label} mode."
                ),
            ):
                if code == "settings":
                    st.session_state["view"] = "settings"
                else:
                    st.session_state["view"] = "playground"
                    st.session_state["mode"] = code
                # Force a clean rerun so the bullet reflects the new state
                _safe_rerun()

        st.title("⚙️ Settings")

        # When in All Settings view, we don't render the blocks here
        if st.session_state.get("view", "playground") == "settings":
            st.caption("All settings are open in the main panel.")
            return

        # --- Configurable settings blocks (under the fixed header) ---
        saved_layout = getattr(cfg, "sidebar_layout", None)
        if not saved_layout:
            layout = [
                "MODELS",
                "WEB",
                "PRESETS",
                "GLOBAL_PARAMS",
                "AGENTS",
                "UI",
                "CONSUL",
                "SAVE_RESET",
                "USAGE",
            ]
        else:
            # Only use known blocks; hidden ones are simply omitted here
            layout = [bid for bid in saved_layout if bid in SIDEBAR_BLOCK_RENDERERS]

        for bid in layout:
            renderer = SIDEBAR_BLOCK_RENDERERS.get(bid)
            if renderer:
                renderer(cfg)





# --- Presets --- #
def _ensure_presets_dir():
    os.makedirs(PRESETS_DIR, exist_ok=True)

def save_preset(name: str, cfg) -> bool:
    _ensure_presets_dir()
    p = Preset(
        name=name,
        description=f"Saved from mode={st.session_state.get('mode','chat')}",
        global_model=cfg.global_model,
        global_params=cfg.global_params.__dict__,
        agents=copy.deepcopy(cfg.agents),
        sidebar_layout=getattr(cfg, "sidebar_layout", []),
        sidebar_expanded=getattr(cfg, "sidebar_expanded", {}) or {},
    )
    path = os.path.join(PRESETS_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(preset_to_dict(p), f, ensure_ascii=False, indent=2)
    return True

def list_presets() -> List[str]:
    _ensure_presets_dir()
    return [f[:-5] for f in os.listdir(PRESETS_DIR) if f.endswith(".json")]

def load_preset(name: str, cfg) -> bool:
    path = os.path.join(PRESETS_DIR, f"{name}.json")
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    p = preset_from_dict(data)

    # 1. apply global model / params
    if p.global_model:
        cfg.global_model = p.global_model
    for k, v in (p.global_params or {}).items():
        setattr(cfg.global_params, k, v)

    # 2. apply agents from preset
    cfg.agents = p.agents or []

    # 3. guarantee planner exists and order is stable
    ensure_planner_agent(cfg)
    cfg.agents = _sort_agents_for_display(cfg.agents)

    # 4. keep planner "enabled" in lockstep with Internet toggle
    web_on = bool(cfg.web_tool.get("enabled", False))
    for a in cfg.agents:
        if is_planner(a):
            a.enabled = web_on

    # 5. apply sidebar layout if present
    if p.sidebar_layout:
        cfg.sidebar_layout = [b for b in p.sidebar_layout if b in ALL_SIDEBAR_BLOCKS]

    # 6. apply sidebar expanded flags if present
    if getattr(p, "sidebar_expanded", None):
        cfg.sidebar_expanded = {
            bid: bool(val)
            for bid, val in p.sidebar_expanded.items()
            if bid in ALL_SIDEBAR_BLOCKS
        }

    # 7. tell the sidebar (next run) to sync widget state to these values
    st.session_state["_apply_preset_ui"] = {
        "global_model": cfg.global_model,
        "global_params": cfg.global_params.clamped().__dict__,
        "same_model_for_all": cfg.same_model_for_all,
    }
    return True





