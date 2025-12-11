# core/state.py
import os, uuid, requests, json, copy, subprocess, re
import streamlit as st
from time import gmtime, strftime
from cryptography.fernet import InvalidToken
from pathlib import Path
from typing  import Dict, List, Optional
from .conversations import list_saved_conversations, save_current_branch, load_conversation_payload, delete_conversation, get_fernet  
from .ollama_client import list_running_models, unload_model
from .config import(AppConfig, AgentConfig, GenParams, Branch, Msg,
                      Preset, preset_to_dict, preset_from_dict, DEFAULT_CHAT_AGENTS, DEFAULT_CONSUL_AGENTS, DEFAULT_NOTEBOOK_AGENTS,
                      DEFAULT_SIDEBAR_LAYOUT, DEFAULT_SIDEBAR_EXPANDED,
)
from .vision import purge_uploads_dir


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

PROJECT_APP_DIR = Path(__file__).resolve().parents[1]
PRESETS_DIR = PROJECT_APP_DIR / "presets"
os.makedirs(PRESETS_DIR, exist_ok=True)

UPLOADS_DIR = PROJECT_APP_DIR / "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)



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

def _parse_advanced_option_value(name: str, raw: str):
    """
    Best-effort parsing of advanced option values from the UI.

    - For `think`, we handle:
        - booleans: true/false
        - levels: low/medium/high  (kept as lowercase strings)
    - For others:
        - "true"/"false" → bool
        - ints / floats → numbers
        - JSON-looking strings ([...] or {...}) → json.loads
        - everything else → raw string
    """
    s = raw.strip()
    if not s:
        return s

    # Special handling for `think`
    if name == "think":
        low = s.lower()
        if low in ("true", "t", "1", "yes", "on"):
            return True
        if low in ("false", "f", "0", "no", "off"):
            return False
        if low in ("low", "medium", "high"):
            # For GPT-OSS / other models that expect "low"/"medium"/"high"
            return low
        # Fallback: send whatever the user typed
        return s

    # Common bools
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False

    # Int?
    try:
        if re.fullmatch(r"-?\d+", s):
            return int(s)
    except Exception:
        pass

    # Float?
    try:
        if re.fullmatch(r"-?\d+\.\d*", s):
            return float(s)
    except Exception:
        pass

    # JSON array/object? (useful for stop lists)
    if s.startswith("[") or s.startswith("{"):
        try:
            return json.loads(s)
        except Exception:
            # If it looks like JSON but fails, just pass it as a string
            return s

    # Default: string
    return s



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

def reset_notebook_files(cfg: AppConfig) -> None: #For the Notebook mode
    """
    Delete the three Notebook files (notes/setup/agent) in workspace/
    and clear notebook-related session_state so the next visit to
    Notebook mode recreates them with default content.

    Actual default contents live in views/notebook.py; we just remove
    the files and in-memory state here.
    """
    # Workspace dir from config (same as views/notebook.py)
    workspace_dir = getattr(cfg, "workspace_dir", "workspace") or "workspace"
    os.makedirs(workspace_dir, exist_ok=True)

    nb = cfg.notebook

    notes_filename = nb.get("notes_filename", "notebook_notes.md")
    setup_filename = nb.get("setup_filename", "notebook_setup.py")
    agent_filename = nb.get("agent_filename", "notebook_agent.py")

    filenames = [notes_filename, setup_filename, agent_filename]

    # Delete files if present
    for fname in filenames:
        path = os.path.join(workspace_dir, fname)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except Exception as e:
            st.warning(f"Could not delete notebook file {fname}: {e}")

    # Clear notebook-related session_state so editors re-init on next render
    for key in [
        "dual_ns",
        "setup_code", "setup_synced", "setup_mtime",
        "setup_stdout", "setup_stderr", "setup_exception",
        "agent_code", "agent_synced", "agent_mtime",
        "agent_stdout", "agent_stderr", "agent_exception",
        "notebook_md_code", "notebook_md_synced",
        "notebook_md_mtime", "notebook_md_last_run",
    ]:
        st.session_state.pop(key, None)

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
        model=cfg.global_model,
        enabled=True,                           # default ON
        params_override={"_role": "planner"},   # marker
        #think_api=False,
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
    # # Prime the widget state once from the config, if not already set | Avoids warning
    # if "same_model_for_all" not in st.session_state:
    #     st.session_state["same_model_for_all"] = cfg.same_model_for_all

    # Purge uploads once per browser session
    if "_uploads_purged" not in st.session_state:
        purge_uploads_dir()
        st.session_state["_uploads_purged"] = True

    # Always guarantee we have a Web Planner and keep agent order stable
    ensure_planner_agent(cfg)
    cfg.agents = _sort_agents_for_display(cfg.agents)

    # --- Branches per mode (chat/consul) ---
    if "branches" not in st.session_state:
        branches: Dict[str, Branch] = {}
        branches["main-chat"] = Branch(id="main-chat", label="main-chat")
        branches["main-consul"] = Branch(id="main-consul", label="main-consul")
        branches["main-notebook"] = Branch(id="main-notebook", label="main-notebook")
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
        if "main-notebook" not in br:
            br["main-notebook"] = Branch(id="main-notebook", label="main-notebook")

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
    Invariants:
    - At most ONE summarizer agent (is_summarizer=True).
    - At most ONE planner (is_planner(agent) == True).
    - Summarizer & planner are treated as utilities and preserved across modes.
    - Mode-specific agents (Chat / Consul) are rebuilt from defaults.
    """
    import copy

    # --- Detect mode change ---
    last_mode = st.session_state.get("_active_mode")
    if last_mode == mode:
        # same mode as last time → do nothing
        return
    st.session_state["_active_mode"] = mode

    # --- 1) Keep / create utility agents (summarizer + planner) ---

    # Reuse the first existing summarizer if present
    summarizer = next(
        (a for a in cfg.agents if getattr(a, "is_summarizer", False)),
        None,
    )

    # If we don't have one yet, bootstrap from the default Chat agents
    if summarizer is None:
        # Find a template summarizer in DEFAULT_CHAT_AGENTS (your single source of truth)
        template = next(
            (a for a in DEFAULT_CHAT_AGENTS if getattr(a, "is_summarizer", False)),
            None,
        )
        if template is not None:
            # deepcopy once → new AgentConfig with its own uid
            summarizer = copy.deepcopy(template)

    # Reuse the first existing planner if present (don't deepcopy so uid & edits persist)
    planner = next((a for a in cfg.agents if is_planner(a)), None)

    # --- 2) Build mode-specific agents from defaults (excluding utilities) ---

    if mode == "chat":
        base_template = DEFAULT_CHAT_AGENTS
    elif mode == "consul":
        base_template = DEFAULT_CONSUL_AGENTS
    elif mode == "notebook":
        base_template = DEFAULT_NOTEBOOK_AGENTS
    else:
        base_template = []

    mode_agents = []
    for a in base_template:
        # Skip summarizers & planners here; we manage them as utilities above
        if getattr(a, "is_summarizer", False) or is_planner(a):
            continue
        mode_agents.append(copy.deepcopy(a))

    # --- 3) Assemble the new roster: summarizer, planner, then mode agents ---

    new_agents: List[AgentConfig] = []

    if summarizer is not None:
        new_agents.append(summarizer)

    if planner is not None:
        new_agents.append(planner)

    new_agents.extend(mode_agents)

    cfg.agents = new_agents

    # --- 4) Guarantee a planner exists & sits under the summarizer ---

    ensure_planner_agent(cfg)  # no-op if a planner is already present

    # Keep planner “enabled” in lockstep with Internet toggle
    web_on = bool(cfg.web_tool.get("enabled", False))
    for a in cfg.agents:
        if is_planner(a):
            a.enabled = web_on

    # --- 5) Bind branches to mode & keep agent order stable ---

    if not _branch_belongs_to_mode(st.session_state.active_branch_id, mode):
        st.session_state.active_branch_id = f"main-{mode}"

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

def add_user_message(text: str, images: Optional[List[str]] = None):
    get_active_branch().messages.append(
        Msg(role="user", sender="User", content=text, images=images or [])
    )

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
            images=getattr(old, "images", []),
        )
        new_branch.messages.append(edited)
        # (no later messages — they'll be regenerated on this branch)
    else:
        # copy all messages,
        new_branch.messages = src.messages.copy()
        # replace only this one
        new_branch.messages[idx] = Msg(
            role=old.role,
            sender=old.sender,
            content=new_text.strip(),
            markdown=getattr(old, "markdown", False),
            thinking=getattr(old, "thinking", None),
            images=getattr(old, "images", []),
        )

    brs[new_id] = new_branch
    st.session_state.active_branch_id = new_id
    return new_id




# --- Sidebar blocks ---
SIDEBAR_EXPANDED_DEFAULTS = DEFAULT_SIDEBAR_EXPANDED # Default from core.config

def sidebar_block_models(cfg: AppConfig):
    with st.expander("Models", expanded=_sidebar_expanded_for(cfg, "MODELS", default=SIDEBAR_EXPANDED_DEFAULTS["MODELS"])):
        colA, colB = st.columns([0.65, 0.35])
        with colA:
            cfg.same_model_for_all = st.checkbox(
                "Use same model for all agents",
                help="When enabled, all agents share the same base model.",
                value=cfg.same_model_for_all,
            )
        with colB:
            if st.button("↺ Refresh", key="refresh_models",
                         help="Query Ollama for the latest list of available models from Ollama."):
                st.session_state.models_cache = list_ollama_models()

        # Verify if the global model is in the model cache of Ollama
        models = st.session_state.get("models_cache", []) 
        if not models: # No models detected from Ollama
            st.warning("No Ollama models found. Use `ollama pull <model>` in your terminal to install a model, then click “Refresh models”.")
            models = [cfg.global_model]
            model_idx = 0
        else:
            if cfg.global_model in models: #Checks if model belongs to the list of models
                model_idx = models.index(cfg.global_model)
            else:
                model_idx = 0
                cfg.global_model = models[0]

        cfg.global_model = st.selectbox(
            "Global model",
            options=models,
            index=model_idx,
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
            help="When ON, the Web Planner auto-generates queries (or start your prompt with `#web:` for manual queries).",
        )
        if cfg.web_tool["enabled"]:
            cfg.web_tool["ddg_results"] = st.slider(
                "Web: results per query",
                1, 10, int(cfg.web_tool.get("ddg_results", 5)),
                1,
                help="Number of web search hits per query to feed into the agents.",
            )
            cfg.web_tool["max_chars"] = st.number_input(
                "Web: summary max chars",
                value=int(cfg.web_tool.get("max_chars", 8000)),
                min_value=500,
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
                "Temperature", 0.0, 2.0, float(gp.temperature), 0.05,
                help="Higher = more random; lower = more deterministic.",
            )
            gp.top_k = st.number_input(
                "top_k", 0, 2048, int(gp.top_k), 10, 
                help="Limit sampling to the top-K tokens (0 = disabled).",
            )
            gp.num_ctx = st.number_input(
                "Context window (num_ctx)", 256, 131072, int(gp.num_ctx), 256,
                help="Maximum tokens of context to keep in the window.",
            )
        with b:
            gp.top_p = st.slider(
                "top_p", 0.0, 1.0, float(gp.top_p), 0.01,
                help="Nucleus sampling: consider tokens whose cumulative probability ≤ top_p.",
            )
            gp.max_tokens = st.number_input(
                "Max tokens (response)", 16, 32768, int(gp.max_tokens), 16,
                help="Maximum length of a single response.",
            )
            seed_str = st.text_input(
                "Seed (optional)",
                value="" if gp.seed in (None, "", "None") else str(gp.seed),
                help="Fix this to make generations reproducible. Leave empty for random.",
            )
            gp.seed = seed_str or None
        # --- Advanced options (min_p, stop, repeat_penalty, think, raw, etc.) ---
        with st.expander("Advanced options", expanded=False):
            # A curated list of Ollama options we don't expose directly.
            # Users can still type any key they want, but this list makes discovery easier
            ADV_KEYS = [
                "min_p",
                "stop",
                "repeat_penalty",
                "repeat_last_n",
                "mirostat",
                "mirostat_tau",
                "mirostat_eta",
                "tfs_z",
                "raw",
                "think",
                "presence_penalty",
                "frequency_penalty",
            ]

            col1, col2, col3 = st.columns([0.4, 0.4, 0.2])
            with col1:
                adv_key = st.selectbox(
                    "Parameter",
                    options=ADV_KEYS,
                    key="adv_opt_key",
                    help="Advanced Ollama option to set. These are added to the `options` dict.",
                )
            with col2:
                adv_val_raw = st.text_input(
                    "Value",
                    key="adv_opt_value",
                    placeholder='e.g. 0.05, true, ["\\nUser:", "</s>"], low',
                    help="Free-form value. Parsed as bool/number/JSON when possible; otherwise sent as a string. "
                         "It might give errors if done poorly."
                )
            with col3:
                if st.button(
                    "Set",
                    key="adv_opt_add",
                    help="Add or update this advanced option.",
                ):
                    val_str = adv_val_raw.strip()
                    if val_str:
                        parsed = _parse_advanced_option_value(adv_key, val_str)
                        if gp.extra_options is None:
                            gp.extra_options = {}
                        gp.extra_options[adv_key] = parsed
                        #_safe_rerun()

            # Show current advanced options with per-key delete buttons
            if gp.extra_options:
                st.markdown("**Current advanced options**")
                to_delete = []
                for k in sorted(gp.extra_options.keys()):
                    v = gp.extra_options[k]
                    colL, colR = st.columns([0.85, 0.15])
                    with colL:
                        # read-only display of the option and its value
                        st.text_input(
                            k,
                            value=repr(v),
                            key=f"adv_show_{k}",
                            disabled=True,
                        )
                    with colR:
                        if st.button(
                            "🗑",
                            key=f"adv_del_{k}",
                            help=f"Remove advanced option '{k}'.",
                        ):
                            to_delete.append(k)
                if to_delete:
                    for k in to_delete:
                        gp.extra_options.pop(k, None)
                    _safe_rerun()
            else:
                st.caption("No advanced options set. Use the controls above to add one.")

        # Finally, clamp and save back to config (extra_options is preserved in clamped()).
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

                cols = st.columns(2) 
                with cols[0]:
                    pass
                with cols[1]: 
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
    with st.expander("UI & Tools", expanded=_sidebar_expanded_for(cfg, "UI", default=SIDEBAR_EXPANDED_DEFAULTS["UI"])):
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
        cfg.show_time = st.checkbox(
            "Show date",
            value=cfg.show_time,
            key="show_time",
            help="When ON, agents can see the current date.",
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

        cfg.feature_flags["ollama_tools"] = st.checkbox(
            "Use Ollama tools (function calling)",
            value=cfg.feature_flags.get("ollama_tools", False),
            key="ollama_tools",
            help=(
                "Use Ollama's native tool calling via /api/chat. "
                "Some models (for example deepseek-r1:14b or gemma3:12b) "
                "do NOT support tools and will error when this is enabled."
            ),
        )

         
        # --- Tools configuration ---
        with st.expander("Tools", expanded=False):
             st.markdown("**Tools availability:**")
             st.markdown("1- Chat &nbsp; &nbsp; &nbsp;2- Consul &nbsp; &nbsp; &nbsp;3- Notebook")
             for tool_name, tool_cfg in cfg.tools.items():
                 display_name = (getattr(tool_cfg, "label", "") or tool_name.replace("_", " ").title())
                 description_info = (getattr(tool_cfg, "description", ""))
                 cols = st.columns([0.4, 0.2, 0.2, 0.2])
                 with cols[0]:
                     st.markdown(display_name, help=description_info)
                 with cols[1]:
                     current_chat = tool_cfg.enabled_modes.get("chat", False)
                     tool_cfg.enabled_modes["chat"] = st.checkbox(
                         "1", value=current_chat, key=f"tool_{tool_name}_chat",
                         help=f"Allow '{tool_name}' in Chat mode.",
                     )
                 with cols[2]:
                     current_consul = tool_cfg.enabled_modes.get("consul", False)
                     tool_cfg.enabled_modes["consul"] = st.checkbox(
                         "2", value=current_consul, key=f"tool_{tool_name}_consul",
                         help=f"Allow '{tool_name}' in Consul mode.",
                     )
                 with cols[3]:
                     current_consul = tool_cfg.enabled_modes.get("notebook", False)
                     tool_cfg.enabled_modes["notebook"] = st.checkbox(
                         "3", value=current_consul, key=f"tool_{tool_name}_notebook",
                         help=f"Allow '{tool_name}' in Notebook mode.",
                     )
                 # If the tool has specific settings, provide appropriate input widgets
                 if tool_name == "rag_search":
                     # RAG tool: select index and number of results
                     index_opts = list(cfg.rag_indexes.keys())
                     default_idx = tool_cfg.settings.get("index") or (index_opts[0] if index_opts else "")
                     if index_opts:
                         chosen_idx = st.selectbox("RAG Index", options=index_opts, index=index_opts.index(default_idx) if default_idx in index_opts else 0,
                                                   key=f"rag_index_select")
                         tool_cfg.settings["index"] = chosen_idx
                     else:
                         st.text("(No RAG indexes configured)")
                     top_k = int(tool_cfg.settings.get("top_k", 3))
                     tool_cfg.settings["top_k"] = st.number_input("Top-K results", min_value=1, max_value=10, value=top_k, key=f"rag_topk_setting")
                 elif tool_cfg.settings:
                    # Generic settings handling for other tools
                    for setting_name, setting_val in tool_cfg.settings.items():
                        setting_key = f"tool_{tool_name}_{setting_name}"

                        # Simple scalar types: editable
                        if isinstance(setting_val, bool):
                            tool_cfg.settings[setting_name] = st.checkbox(
                                f"{tool_name}: {setting_name}",
                                value=setting_val,
                                key=setting_key,
                            )
                        elif isinstance(setting_val, (int, float)):
                            tool_cfg.settings[setting_name] = st.number_input(
                                f"{tool_name}: {setting_name}",
                                value=setting_val,
                                key=setting_key,
                            )

                        # Structured settings (dict/list): show read-only JSON for transparency, but don't overwrite.
                        elif isinstance(setting_val, (dict, list)):
                            st.text_area(
                                f"{tool_name}: {setting_name} (read-only JSON)",
                                value=json.dumps(setting_val, indent=2),
                                key=setting_key,
                                height=120,
                                disabled=True,
                            )

                        # Fallback: editable text
                        else:
                            tool_cfg.settings[setting_name] = st.text_input(
                                f"{tool_name}: {setting_name}",
                                value=str(setting_val),
                                key=setting_key,
                            )

        # --- Image uploads (for vision models) ---
        with st.expander("Upload images", expanded=False):
            mode = st.session_state.get("mode", "chat")
            pending_images_key = f"_{mode}_pending_images"
            pending_images = st.session_state.get(pending_images_key, [])
            #
            uploaded_files = st.file_uploader(
                "Choose image(s) to attach to the next message",
                type=["png", "jpg", "jpeg", "webp"],
                key=f"image_uploader_{mode}",
                accept_multiple_files=True,
            )

            if uploaded_files:
                try:
                    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

                    # avoid duplicates if user reselects the same files
                    existing_paths = {img["path"] for img in pending_images}

                    for uf in uploaded_files:
                        dest = UPLOADS_DIR / uf.name
                        with open(dest, "wb") as f:
                            f.write(uf.getbuffer())
                        path_str = str(dest)
                        if path_str not in existing_paths:
                            pending_images.append({"path": path_str, "name": uf.name})
                            existing_paths.add(path_str)

                    st.session_state[pending_images_key] = pending_images
                    st.success(f"Attached {len(uploaded_files)} image(s) to the next {mode} message.")
                except Exception as e:
                    st.warning(f"Could not save uploaded image(s): {e}")

            # Show what is currently pending
            if pending_images:
                names = ", ".join(img["name"] for img in pending_images)
                st.caption(f"Pending images for next {mode} message: {names}")
                if st.button("Clear pending images", key=f"clear_imgs_{mode}"):
                    st.session_state[pending_images_key] = []


def sidebar_block_consul(cfg: AppConfig):
    with st.expander("Mode Parameters", expanded=_sidebar_expanded_for(cfg, "CONSUL", default=SIDEBAR_EXPANDED_DEFAULTS["CONSUL"])):
        # --- Consul parameters ---
        with st.expander("Consul", expanded=False):
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
            ) # You can re-enable alpha / auto_web_brief as needed here later.
        # --- Notebook parameters ---
        with st.expander("Notebook", expanded=False):
            nb = cfg.notebook

            nb["autosave"] = st.checkbox(
                "Autosave notebook files to workspace/",
                value=bool(nb.get("autosave", True)),
                key="notebook_autosave",
                help="When enabled, every Run writes the notebook cells to files under the workspace directory.",
            )

            nb["rounds"] = st.number_input(
                "Notebook rounds",
                min_value=1,
                max_value=20,
                step=1,
                value=int(nb.get("rounds", 3)),
                help="How many collaborative rounds the Notebook agents run after each user message.",
                key="notebook_rounds",
            )

            nb["max_output_chars"] = st.number_input(
                "Max output characters (agent + setup)",
                min_value=1_000,
                max_value=200_000,
                step=1_000,
                value=int(nb.get("max_output_chars", 20_000)),
                key="notebook_max_output_chars",
            )

            st.caption("Filenames (relative to workspace/) used by the Notebook mode:")

            col_notes, col_setup, col_agent = st.columns(3)

            with col_notes:
                nb["notes_filename"] = st.text_input(
                    "Notes file",
                    value=nb.get("notes_filename", "notebook_notes.md"),
                    key="notebook_notes_filename",
                )

            with col_setup:
                nb["setup_filename"] = st.text_input(
                    "Setup file",
                    value=nb.get("setup_filename", "notebook_setup.py"),
                    key="notebook_setup_filename",
                )

            with col_agent:
                nb["agent_filename"] = st.text_input(
                    "Agent file",
                    value=nb.get("agent_filename", "notebook_agent.py"),
                    key="notebook_agent_filename",
                )

            st.caption(
                "Agents can read and write these files using the generic "
                "`workspace_read` / `workspace_write` tools."
            )
            # --- Reset Notebook files to defaults ---
            if st.button(
                "↺ Reset notebook files to defaults",
                key="notebook_reset_files_btn",
                help=(
                    "Delete the notebook Notes/Setup/Agent files under workspace/ and "
                    "reset the in-memory notebook state. "
                    "Next time you open Notebook mode, the cells will be recreated "
                    "with their default content."
                ),
            ):
                reset_notebook_files(cfg)
                st.success("Notebook files reset. Open Notebook mode to see the default cells again.")
                _safe_rerun()
        # --- ... ---
        with st.expander("[Future Mode]", expanded=False):
            st.markdown("empty")
            # Space saved for more modes in the future
            

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
                #_safe_rerun()

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
                #_safe_rerun()


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
    # preset_ui = st.session_state.pop("_apply_preset_ui", None)
    # if preset_ui:
    #     gp = preset_ui.get("global_params", {})
    #     gm = preset_ui.get("global_model")

    #     # ensure model is in the options list
    #     cache = st.session_state.get("models_cache", []) or ["llama3:8b"]
    #     if gm and gm not in cache:
    #         st.session_state["models_cache"] = cache + [gm]

    #     # set widget keys BEFORE creating widgets 
    #     st.session_state["same_model_for_all"] = preset_ui.get(
    #         "same_model_for_all",
    #         st.session_state.get("same_model_for_all", cfg.same_model_for_all),
    #     )
    #     if gm:
    #         st.session_state["global_model_select"] = gm

    with st.sidebar:
        # --- Mode / Settings bar (3 tabs) ---
        st.markdown("**Mode**")

        TABS = [
            ("chat", "Chat"),
            ("consul", "Consul"),
             ("notebook", "Notebook"),
            ("settings", "All Settings"),
        ]

        cur_view = st.session_state.get("view", "playground")
        cur_mode = st.session_state.get("mode", "chat")

        # Active tab is derived purely from view + mode
        if cur_view == "settings":
            active_tab = "settings"
        else:
            active_tab = cur_mode if cur_mode in ("chat", "consul", "notebook") else "chat"

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
                
                _safe_rerun() # Force a clean rerun so the bullet reflects the new state

        st.title("⚙️ Settings")

        # When in All Settings view, we don't render the blocks here
        if st.session_state.get("view", "playground") == "settings":
            st.caption("All settings are open in the main panel.")
            return

        # --- Configurable settings blocks (under the fixed header) ---
        saved_layout = getattr(cfg, "sidebar_layout", None)
        if not saved_layout:
            layout = DEFAULT_SIDEBAR_LAYOUT.copy() # use default layout from config
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
    )

    data = preset_to_dict(p)
    raw = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")

    f = get_fernet()
    blob = f.encrypt(raw) if f else raw

    path = os.path.join(PRESETS_DIR, f"{name}.json")
    with open(path, "wb") as fh:
        fh.write(blob)

    return True

def list_presets() -> List[str]:
    _ensure_presets_dir()
    return [f[:-5] for f in os.listdir(PRESETS_DIR) if f.endswith(".json")]

def load_preset(name: str, cfg) -> bool:
    path = os.path.join(PRESETS_DIR, f"{name}.json")
    if not os.path.exists(path):
        return False

    # Read as bytes (could be encrypted or plain JSON)
    with open(path, "rb") as fh:
        blob = fh.read()

    f = get_fernet()
    if f:
        try:
            raw = f.decrypt(blob)
        except InvalidToken:
            # Preset was saved before encryption was introduced; treat as plaintext JSON
            raw = blob
    else:
        raw = blob

    data = json.loads(raw.decode("utf-8"))
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

    # All UI widgets read from cfg on the next rerun; no extra UI sync needed
    return True





