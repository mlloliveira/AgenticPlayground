# app/views/notebook.py

from __future__ import annotations

import ast
import builtins as _builtins
import io
import os
import sys
import traceback
from typing import Any

import streamlit as st

from .notebook_imports import state, Code, Editor
from core.config import AppConfig, Branch, Msg
from core.state import should_speak
from .runtime import (
    render_mode_scaffold,
    prepare_turn_common,
    stream_agent_bubble,
    run_summarizer_if_enabled,
    last_user_index,
    prompt_branch_for_round,
)

# -------------------------------------------------------------------
# Configurable editor heights
# -------------------------------------------------------------------

# Visible lines for the editors before scrolling
MAX_LINES_SETUP = 5
MAX_LINES_AGENT = 25
MAX_LINES_MARKDOWN = 25

# -------------------------------------------------------------------
# Runtime-configurable notebook settings
# (paths + limits are updated from cfg.notebook in notebook_page)
# -------------------------------------------------------------------

ENABLE_AUTOSAVE: bool = True
AUTOLOAD_ON_START: bool = True

MARKDOWN_AUTOSAVE_PATH: str = ""
SETUP_AUTOSAVE_PATH: str = ""
AGENT_AUTOSAVE_PATH: str = ""

# Limit how much output we keep (avoid huge prints)
MAX_OUTPUT_CHARS: int = 20_000

# -------------------------------------------------------------------
# Default cells
# -------------------------------------------------------------------
DEFAULT_MARKDOWN_TEXT = (
    "# Notebook markdown\n\n"
    "Describe your plan, world, or notes here…"
)

DEFAULT_SETUP_TEXT = (
    "# Setup cell (UNSAFE – imports and environment)\n"
    "# You can define imports, helper functions, globals, etc. here.\n"
    "# The agent cell can use these names, but cannot import on its own.\n"
)

DEFAULT_AGENT_TEXT = (
    "# Agent cell (SAFE – no imports). Write pure logic that uses names defined in the setup cell. No 'import' statements are allowed here; they will raise an error.\n"
    "print('hello world')"
)
# -------------------------------------------------------------------
# Optional markdown->HTML converter for nicer, scrollable preview
# -------------------------------------------------------------------

try:
    import markdown as _md_lib

    def _md_to_html(text: str) -> str:
        return _md_lib.markdown(text, extensions=["fenced_code", "tables"])
except Exception:  # optional dependency
    _md_lib = None

    def _md_to_html(text: str) -> str:
        return text  # fallback: identity


# -------------------------------------------------------------------
# Safe execution sandbox
# -------------------------------------------------------------------
# This is NOT a perfectly secure sandbox, but much safer than raw exec:
#
# - No imports allowed (AST check).
# - Builtins restricted to a small whitelist (no open, eval, exec, __import__…).
# - A step counter via sys.settrace that kills long/infinite loops.
# -------------------------------------------------------------------

SAFE_BUILTINS: dict[str, Any] = {}
for _name, _obj in _builtins.__dict__.items():
    # Block these explicitly
    if _name in {"open", "__import__", "eval", "exec", "input"}:
        continue
    SAFE_BUILTINS[_name] = _obj

# Whitelisted global modules (you can add more later if needed)
SAFE_GLOBALS_BASE: dict[str, Any] = {}

# Max executed "line" events before we kill the run
MAX_EXEC_STEPS = 100_000

def safe_exec(source: str, ns: dict) -> None:
    """
    Execute `source` in a restricted environment, mutating ns in-place.

    Restrictions:
    - No import statements.
    - No access to dangerous builtins like open, eval, exec, __import__.
    - Limited builtins via SAFE_BUILTINS.
    - Step-count limit via sys.settrace to catch infinite loops,
      but only for code coming from this notebook cell.
    """
    # 1) Parse to AST and block imports early
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError:
        # Let syntax errors surface as-is; Streamlit will show the traceback
        raise

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise RuntimeError("Import statements are disabled in the safe agent cell.")
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "__builtins__":
                raise RuntimeError("Direct access to __builtins__ is restricted in safe mode.")

    code_obj = compile(tree, "<notebook_safe_cell>", "exec")
    target_filename = code_obj.co_filename  # usually "<notebook_safe_cell>"

    # 2) Prepare safe globals – TEMPORARILY override __builtins__
    old_builtins = ns.get("__builtins__", _builtins)
    ns["__builtins__"] = SAFE_BUILTINS

    for name, val in SAFE_GLOBALS_BASE.items():
        ns.setdefault(name, val)

    # 3) Set up a line-based step counter, but only for this cell's code
    steps = 0

    def tracefunc(frame, event, arg):
        nonlocal steps
        if event == "line" and frame.f_code.co_filename == target_filename:
            steps += 1
            if steps > MAX_EXEC_STEPS:
                raise TimeoutError(
                    f"Maximum execution steps exceeded ({MAX_EXEC_STEPS}). "
                    "Your code may contain an infinite loop."
                )
        return tracefunc

    old_trace = sys.gettrace()
    sys.settrace(tracefunc)
    try:
        exec(code_obj, ns)
    finally:
        sys.settrace(old_trace)
        # Restore previous builtins so setup cell imports keep working
        ns["__builtins__"] = old_builtins


# -------------------------------------------------------------------
# Autosave helpers
# -------------------------------------------------------------------

def _autosave(path: str, text: str) -> None:
    """Save text to a file, if autosave is enabled."""
    if not ENABLE_AUTOSAVE or not path:
        return

    # Normalize: strip trailing spaces on each line, keep indentation
    lines = text.splitlines()
    cleaned = "\n".join(line.rstrip() for line in lines)

    # Preserve final newline if the original text had one
    if text.endswith("\n"):
        cleaned += "\n"

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(cleaned)
    except Exception as e:
        # Transparency over perfection
        msg = f"Autosave failed: {e}"
        try:
            st.toast(msg, icon="⚠️")
        except Exception:
            st.warning(msg)


def _autoload_if_needed(path: str, default_text: str) -> str:
    """
    Return initial text for a Code object.

    If AUTOLOAD_ON_START is True and the file exists, load it.
    Otherwise, fall back to default_text.
    """
    if AUTOLOAD_ON_START and path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return default_text
    return default_text

# -------------------------------------------------------------------
# Refresh helpers
# -------------------------------------------------------------------

def _refresh_markdown_from_disk_if_needed() -> None:
    if not MARKDOWN_AUTOSAVE_PATH or "notebook_md_code" not in state:
        return
    try:
        mtime = os.path.getmtime(MARKDOWN_AUTOSAVE_PATH)
    except OSError:
        return

    if mtime <= getattr(state, "notebook_md_mtime", 0.0):
        return  # no change on disk

    # Only reload if user has no unsaved edits:
    cur = state.notebook_md_code.get_value()
    synced = getattr(state, "notebook_md_synced", cur)
    if cur != synced:
        return  # user has diverged locally; don't clobber

    try:
        with open(MARKDOWN_AUTOSAVE_PATH, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return

    state.notebook_md_code.from_backend(text)
    state.notebook_md_last_run = text
    state.notebook_md_synced = text
    state.notebook_md_mtime = mtime


def _refresh_code_from_disk_if_needed() -> None:
    """Reload setup/agent code from disk if files changed and user hasn't edited them."""
    # --- Setup cell ---
    if SETUP_AUTOSAVE_PATH and "setup_code" in state:
        try:
            mtime = os.path.getmtime(SETUP_AUTOSAVE_PATH)
        except OSError:
            mtime = 0.0
        if mtime > getattr(state, "setup_mtime", 0.0):
            current = state.setup_code.get_value()
            synced = getattr(state, "setup_synced", current)
            if current == synced:
                try:
                    with open(SETUP_AUTOSAVE_PATH, "r", encoding="utf-8") as f:
                        text = f.read()
                    state.setup_code.from_backend(text)
                    state.setup_synced = text
                    state.setup_mtime = mtime
                except Exception as e:
                    st.warning(f"Could not reload setup code from disk: {e}")

    # --- Agent cell ---
    if AGENT_AUTOSAVE_PATH and "agent_code" in state:
        try:
            mtime = os.path.getmtime(AGENT_AUTOSAVE_PATH)
        except OSError:
            mtime = 0.0
        if mtime > getattr(state, "agent_mtime", 0.0):
            current = state.agent_code.get_value()
            synced = getattr(state, "agent_synced", current)
            if current == synced:
                try:
                    with open(AGENT_AUTOSAVE_PATH, "r", encoding="utf-8") as f:
                        text = f.read()
                    state.agent_code.from_backend(text)
                    state.agent_synced = text
                    state.agent_mtime = mtime
                except Exception as e:
                    st.warning(f"Could not reload agent code from disk: {e}")




# -------------------------------------------------------------------
# Code mode helpers (SAFE agent + unsafe setup)
# -------------------------------------------------------------------

def _ensure_code_state() -> None:
    """Initialize session_state entries for the dual code mode."""
    # Shared namespace used by BOTH cells:
    # - setup runs with exec()
    # - agent runs with safe_exec() using this same ns
    if "dual_ns" not in state:
        state.dual_ns = {}

    # Setup cell code (unsafe)
    if "setup_code" not in state:
        initial_setup = _autoload_if_needed(SETUP_AUTOSAVE_PATH, DEFAULT_SETUP_TEXT)
        state.setup_code = Code(initial_setup)
        state.setup_synced = initial_setup
        try:
            state.setup_mtime = os.path.getmtime(SETUP_AUTOSAVE_PATH)
        except OSError:
            state.setup_mtime = 0.0

    # Agent cell code (safe)
    if "agent_code" not in state:
        initial_agent = _autoload_if_needed(
            AGENT_AUTOSAVE_PATH,
            DEFAULT_AGENT_TEXT,
        )
        state.agent_code = Code(initial_agent)
        state.agent_synced = initial_agent
        try:
            state.agent_mtime = os.path.getmtime(AGENT_AUTOSAVE_PATH)
        except OSError:
            state.agent_mtime = 0.0

    # Outputs for setup cell
    if "setup_stdout" not in state:
        state.setup_stdout = ""
    if "setup_stderr" not in state:
        state.setup_stderr = ""
    if "setup_exception" not in state:
        state.setup_exception = ""

    # Outputs for agent cell
    if "agent_stdout" not in state:
        state.agent_stdout = ""
    if "agent_stderr" not in state:
        state.agent_stderr = ""
    if "agent_exception" not in state:
        state.agent_exception = ""


def _run_setup() -> None:
    """Run the unsafe setup cell with full Python exec()."""
    code_obj: Code = state.setup_code
    src = code_obj.get_value()
    ns = state.dual_ns

    buf_out = io.StringIO()
    buf_err = io.StringIO()

    try:
        from contextlib import redirect_stdout, redirect_stderr

        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            exec(compile(src, "<notebook_setup>", "exec"), ns)
        exc_str = ""
    except Exception:
        exc_str = traceback.format_exc()

    state.setup_stdout = buf_out.getvalue()[:MAX_OUTPUT_CHARS]
    state.setup_stderr = buf_err.getvalue()[:MAX_OUTPUT_CHARS]
    state.setup_exception = exc_str[:MAX_OUTPUT_CHARS]

    _autosave(SETUP_AUTOSAVE_PATH, src)
    state.setup_synced = src
    try:
        state.setup_mtime = os.path.getmtime(SETUP_AUTOSAVE_PATH)
    except OSError:
        pass


def _run_code_safe() -> None:
    """Run the agent cell using safe_exec() in the shared namespace."""
    code_obj: Code = state.agent_code
    src = code_obj.get_value()
    ns = state.dual_ns  # same shared namespace as setup cell

    buf_out = io.StringIO()
    buf_err = io.StringIO()

    try:
        from contextlib import redirect_stdout, redirect_stderr

        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            safe_exec(src, ns)
        exc_str = ""
    except Exception:
        exc_str = traceback.format_exc()

    state.agent_stdout = buf_out.getvalue()[:MAX_OUTPUT_CHARS]
    state.agent_stderr = buf_err.getvalue()[:MAX_OUTPUT_CHARS]
    state.agent_exception = exc_str[:MAX_OUTPUT_CHARS]

    _autosave(AGENT_AUTOSAVE_PATH, src)
    state.setup_synced = src
    try:
        state.agent_mtime = os.path.getmtime(AGENT_AUTOSAVE_PATH)
    except OSError:
        pass


def show_code_mode() -> None:
    st.subheader("Code notebook")

    _ensure_code_state()
    _refresh_code_from_disk_if_needed()

    setup_code_obj: Code = state.setup_code
    agent_code_obj: Code = state.agent_code

    # ---------------------- Setup cell (unsafe) ----------------------
    st.markdown("**Setup cell (UNSAFE – for imports and environment)**")

    setup_editor = Editor(
        code=setup_code_obj,
        key="notebook_setup_editor",
        lang="python",
        min_lines=MAX_LINES_SETUP,
        max_lines=MAX_LINES_SETUP,
    )
    setup_editor.show()

    if setup_editor.event in ("run", "submit"):
        _run_setup()

    with st.expander("Setup cell output", expanded=True):
        if state.setup_stdout:
            st.code(state.setup_stdout, language="text")
        if state.setup_stderr:
            st.code(state.setup_stderr, language="text")
        if state.setup_exception:
            st.error(state.setup_exception)
        elif not state.setup_stdout and not state.setup_stderr:
            st.caption("No output yet – run the setup cell to see logs here.")

    st.markdown("---")

    # ---------------------- Agent cell (safe) ----------------------
    st.markdown("**Agent cell (SAFE – sandboxed code)**")

    agent_editor = Editor(
        code=agent_code_obj,
        key="notebook_agent_editor",
        lang="python",
        min_lines=MAX_LINES_AGENT,
        max_lines=MAX_LINES_AGENT,
    )
    agent_editor.show()

    if agent_editor.event in ("run", "submit"):
        _run_code_safe()

    if state.agent_stdout or state.agent_stderr or state.agent_exception:
        st.markdown("#### Agent output")
        if state.agent_stdout:
            st.code(state.agent_stdout, language="text")
        if state.agent_stderr:
            st.code(state.agent_stderr, language="text")
        if state.agent_exception:
            st.error(state.agent_exception)
    else:
        st.info(
            "No output yet. "
            "**Run** or **Ctrl+Enter** to see logs here."
        )


# -------------------------------------------------------------------
# Markdown mode helpers
# -------------------------------------------------------------------

def _ensure_markdown_state() -> None:
    if "notebook_md_code" not in state:
        initial = _autoload_if_needed(MARKDOWN_AUTOSAVE_PATH, DEFAULT_MARKDOWN_TEXT)
        state.notebook_md_code = Code(initial)
        state.notebook_md_synced = initial
        try:
            state.notebook_md_mtime = os.path.getmtime(MARKDOWN_AUTOSAVE_PATH)
        except OSError:
            state.notebook_md_mtime = 0.0

    if "notebook_md_last_run" not in state:
        # On first use, last_run = current text
        state.notebook_md_last_run = state.notebook_md_code.get_value()


def show_markdown_mode() -> None:
    import streamlit.components.v1 as components

    st.subheader("Markdown notebook")

    _ensure_markdown_state()
    _refresh_markdown_from_disk_if_needed()
    code_obj: Code = state.notebook_md_code

    col_editor, col_preview = st.columns(2)

    # ---- Left: editor ----
    with col_editor:
        editor = Editor(
            code=code_obj,
            key="notebook_markdown_editor",
            lang="markdown",
            min_lines=MAX_LINES_MARKDOWN,
            max_lines=MAX_LINES_MARKDOWN,
        )
        editor.show()

        if editor.event in ("run", "submit"):
            state.notebook_md_last_run = code_obj.get_value()
            _autosave(MARKDOWN_AUTOSAVE_PATH, state.notebook_md_last_run)
            state.notebook_md_synced = state.notebook_md_last_run
            try:
                state.notebook_md_mtime = os.path.getmtime(MARKDOWN_AUTOSAVE_PATH)
            except OSError:
                pass

        st.caption("Press **Run** or **Ctrl+Enter** to update the preview.")

    # ---- Right: compiled markdown preview in a scrollable box ----
    with col_preview:
        text = state.notebook_md_last_run

        line_height_px = 24
        lines = MAX_LINES_MARKDOWN or 15
        preview_height = lines * line_height_px

        if _md_lib is not None:
            html = _md_to_html(text)
            components.html(
                f"""
                <div style="
                    max-height: {preview_height}px;
                    overflow-y: auto;
                    padding: 0.5rem;
                ">
                    {html}
                </div>
                """,
                height=preview_height + 40,
                scrolling=False,
            )
        else:
            # Fallback: plain markdown, no scroll box
            st.markdown(text)


# -------------------------------------------------------------------
# Multi-agent Notebook orchestration (chat-like, Consul-style)
# -------------------------------------------------------------------

def _run_notebook_streaming(cfg: AppConfig, branch: Branch):
    """
    Notebook turn orchestration.

    - User sends a task.
    - We run R collaborative rounds, where agents may speak according to their schedule.
    - For each round, we inject explicit 'Round i / R' info into the prompt.
    - There is no voting step; agents simply collaborate in the transcript.
    - At the end, the summarizer (if enabled) creates a summary message.
    """
    last_user_idx = last_user_index(branch)
    if last_user_idx is None:
        # Nothing to do yet; user hasn't sent a message.
        return

    # Common turn setup: agents (excluding summarizer/planner), and web context.
    _ignored_prompt_branch, agents, web_ctx = prepare_turn_common(cfg, branch)

    if not agents:
        st.info("No enabled notebook agents. Turn on at least one non-summarizer agent.")
        return

    nb_cfg: dict[str, Any] = getattr(cfg, "notebook", {}) or {}
    total_rounds = int(nb_cfg.get("rounds", 3))

    for round_idx in range(1, total_rounds + 1):
        # Decide which agents speak this round (same scheduling logic as Consul).
        active_agents = [a for a in agents if should_speak(a, round_idx, total_rounds)]
        if not active_agents:
            continue

        holders = [st.empty() for _ in active_agents]

        # Build the context/messages visible this round.
        prompt_branch = prompt_branch_for_round(cfg, branch, last_user_idx)

        # Inject explicit round info as a system message used only for prompting.
        round_msg = Msg(
            role="system",
            sender="Round",
            content=(
                f"[NOTEBOOK ROUND] This is round {round_idx} of {total_rounds} "
                "in a collaborative Notebook session.\n\n"
                "Use this information to pace your work."
            ),
            markdown=False,
        )
        prompt_branch.messages.append(round_msg)

        # Stream each active agent as a bubble, sharing the same prompt_branch.
        for ag, holder in zip(active_agents, holders):
            stream_agent_bubble(
                holder=holder,
                agent=ag,
                prompt_branch=prompt_branch,
                final_branch=branch,
                cfg=cfg,
                web_ctx=web_ctx,
            )

    # Optional summarizer (same as Chat/Consul)
    run_summarizer_if_enabled(cfg, branch)


# -------------------------------------------------------------------
# Entry point for the Notebook mode
# -------------------------------------------------------------------

def notebook_page() -> None:
    """
    Main entrypoint for the Notebook mode.

    - Top: multi-agent, multi-round Notebook chat (like Consul but no voting).
    - Bottom: Notebook editors:
        • Markdown notebook (saved as a file in workspace/).
        • Dual code cells: unsafe setup + safe agent cell.

    All files live under cfg.workspace_dir so agents can use
    workspace_read/workspace_write on the same paths.
    """
    cfg: AppConfig = st.session_state.app_config

    nb_cfg: dict[str, Any] = getattr(cfg, "notebook", {}) or {}

    workspace_dir = getattr(cfg, "workspace_dir", "workspace") or "workspace"
    os.makedirs(workspace_dir, exist_ok=True)

    notes_filename = nb_cfg.get("notes_filename", "notebook_notes.md")
    setup_filename = nb_cfg.get("setup_filename", "notebook_setup.py")
    agent_filename = nb_cfg.get("agent_filename", "notebook_agent.py")

    global MARKDOWN_AUTOSAVE_PATH, SETUP_AUTOSAVE_PATH, AGENT_AUTOSAVE_PATH
    global ENABLE_AUTOSAVE, MAX_OUTPUT_CHARS, AUTOLOAD_ON_START

    MARKDOWN_AUTOSAVE_PATH = os.path.join(workspace_dir, notes_filename)
    SETUP_AUTOSAVE_PATH = os.path.join(workspace_dir, setup_filename)
    AGENT_AUTOSAVE_PATH = os.path.join(workspace_dir, agent_filename)

    ENABLE_AUTOSAVE = bool(nb_cfg.get("autosave", True))
    AUTOLOAD_ON_START = True  # always try to load existing notebook files
    MAX_OUTPUT_CHARS = int(nb_cfg.get("max_output_chars", 20_000))

    # Ensure the files exist on disk with sensible defaults
    for path, default in [
        (MARKDOWN_AUTOSAVE_PATH, DEFAULT_MARKDOWN_TEXT),
        (SETUP_AUTOSAVE_PATH, DEFAULT_SETUP_TEXT),
        (AGENT_AUTOSAVE_PATH, DEFAULT_AGENT_TEXT),
    ]:
        if not os.path.exists(path):
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(default)
            except Exception as e:
                st.warning(f"Could not create notebook file {os.path.basename(path)}: {e}")

    # ---- Top: chat-like multi-agent Notebook mode ----

    rounds = int(nb_cfg.get("rounds", 3))
    caption = (
        "Mode: Notebook · "
        f"Model mode: {'One model for all (' + cfg.global_model + ')' if cfg.same_model_for_all else 'Per-agent models'} · "
        f"Rounds per turn: {rounds}"
    )

    render_mode_scaffold(
        mode="notebook",
        caption=caption,
        selectbox_key="notebook_branch_select",
        input_key="notebook_chat_input",
        input_label="Ask the notebook agents…",
        run_turn_fn=_run_notebook_streaming,
    )

    # ---- Bottom: actual notebook editors (Markdown + Code) ----

    with st.expander("Notebook info", expanded=False):
        st.markdown(
            "Below, you have a simple markdown notebook and a two-cell code notebook within two possibilities:\n"
            "- **Code**\n"
            "   - **Setup** runs UNSAFE (full Python exec, for imports and helpers).\n"
            "   - **Agent** runs in a SAFE sandbox (no imports, limited builtins, step limit).\n"
            "       - It can use names defined in the setup cell (e.g. random, tkinter, helper functions).\n"
            "- **Note**\n"
            "   - **Markdown** is saved to the workspace and compiled to a live preview.\n"
            "---\n\n"
            "- The **Notebook** mode requires `Use Ollama tools` to be `True`.\n" 
            "- The **Notebook** mode just works with *'thinking'* models (e.g. gpt-oss, qwen3).\n" 
            "- All cells are stored as normal files in your workspace so agents can read/write them "
            "using the `workspace_read` and `workspace_write` tools.\n"
             "---\n\n"
             "**Tips:**\n"
             "- Set `Blind First Turn` to `False`.\n" 
             "- Increase the `Max Tokens` and `Context Window` inside of `Generation Parameters` to avoid a `500 server error`.\n" 
             "- The agents can execute the code if the option for `py_repl` is set to `True`.\n" 
             "  - If that's the case, let the agents know they can execute the code with the `py_repl` tool.\n" 

        )

        with st.expander("Notebook files in workspace/", expanded=False):
            st.markdown(f"Notes file : `{notes_filename}`")
            st.markdown(f"Setup file : `{setup_filename}`")
            st.markdown(f"Agent file : `{agent_filename}`")

    choice = st.radio(
        "Notebook view",
        options=["Code", "Note"],  # Code first = default
        key="notebook_view_mode",
        horizontal=True,
    )

    if choice == "Note":
        show_markdown_mode()
    else:
        show_code_mode()
