# core/tools.py (new module)
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from contextlib import redirect_stdout, redirect_stderr
import os, json, random, time, ast, io, sys, traceback
import builtins as _builtins
# Import config and types for context
from .config import AppConfig, AgentConfig, Branch, ToolConfig, RagIndexConfig

@dataclass
class ToolContext:
    """Context available to tools: current agent, app config, and branch (conversation)."""
    agent: AgentConfig
    cfg: AppConfig
    branch: Branch
    # Per-turn cache for tool calls (resets each turn)
    turn_cache: Dict[str, "ToolResult"] = field(default_factory=dict)

@dataclass
class ToolResult:
    output: str = ""
    error: Optional[str] = None
    # Optional fields for special tools (e.g., retrieval details for RAG)
    retrieved: list = field(default_factory=list)    # list of retrieved chunks (for rag_search)
    context: str = ""                                # combined context block (for rag_search)

    def is_success(self) -> bool:
        return self.error is None

# Global tool registry and caches
_tools_registry: Dict[str, Any] = {}
_tool_cache_global: Dict[str, ToolResult] = {}
_tool_cache_agent: Dict[str, Dict[str, ToolResult]] = {}

def register_tool(name: str):
    """Decorator to register a tool function with a given name in the global registry."""
    def decorator(func):
        _tools_registry[name] = func
        return func
    return decorator

def run_tool(ctx: ToolContext, tool_name: str, args: Any) -> ToolResult:
    """Execute a tool by name with given arguments, using caching as appropriate."""
    tools_cfg = getattr(ctx.cfg, "tools", {}) or {}
    tool_cfg: Optional[ToolConfig] = tools_cfg.get(tool_name)

    if not tool_cfg:
        return ToolResult(error=f"Tool '{tool_name}' is disabled or not found.")

    # Single mode inference path
    mode = _infer_mode_from_branch(ctx.branch)

    # Shared gating logic
    if not _tool_allowed_for(ctx.agent, tool_name, tool_cfg, mode):
        # Slightly nicer error messages depending on the reason
        if not tool_cfg.enabled_modes.get(mode, True):
            return ToolResult(error=f"Tool '{tool_name}' is not enabled in {mode.capitalize()} mode.")
        if ctx.agent.allowed_tools and tool_name not in ctx.agent.allowed_tools:
            return ToolResult(error=f"Tool '{tool_name}' not allowed for agent {ctx.agent.name}.")
        # Fallback (shouldn’t really happen)
        return ToolResult(error=f"Tool '{tool_name}' is not allowed in this context.")

    # Prepare cache key (use JSON dump for dict args to ensure hashable string)
    try:
        args_key = json.dumps(args, sort_keys=True) if not isinstance(args, str) else str(args)
    except Exception:
        args_key = str(args)
    cache_key = f"{tool_name}:{args_key}"

    # Decide caching scopes (skip caching for some non-deterministic tools)
    cacheable = tool_name not in {"dice", "workspace_write", "workspace_read", "py_repl", "conv_search"}

    if cacheable:
        # Check per-turn cache
        if cache_key in ctx.turn_cache:
            return ctx.turn_cache[cache_key]
        # Check per-agent cache
        agent_cache = _tool_cache_agent.setdefault(ctx.agent.uid, {})
        if cache_key in agent_cache:
            return agent_cache[cache_key]
        # Check global cache
        if cache_key in _tool_cache_global:
            return _tool_cache_global[cache_key]

    # Run the tool function if available
    func = _tools_registry.get(tool_name)
    if not func:
        return ToolResult(error=f"Tool '{tool_name}' is not implemented.")

    result: ToolResult = func(ctx, args)

    # Store in caches if successful and cacheable
    if cacheable and result and result.error is None:
        ctx.turn_cache[cache_key] = result
        agent_cache = _tool_cache_agent.setdefault(ctx.agent.uid, {})
        agent_cache[cache_key] = result
        if tool_name in {"rag_search"}:  # heavy tools can go global
            _tool_cache_global[cache_key] = result

    return result


# ------------------- Tool selection helpers (for prompts & Ollama tools) ------------------- #

def _infer_mode_from_branch(branch: Branch) -> str:
    """
    Best-effort detection of current mode for a given branch.

    Prefers Streamlit session_state["mode"] when available, otherwise falls
    back to branch id prefixes like 'main-chat', 'chat-2', 'consul-1', etc.
    """
    mode = None
    try:
        import streamlit as st  # type: ignore
        mode = st.session_state.get("mode", None)
    except Exception:
        mode = None

    if mode:
        return str(mode)

    bid = getattr(branch, "id", "") or ""
    if bid.startswith("main-"):
        return bid.split("-", 1)[1] or "chat"
    if "-" in bid:
        prefix = bid.split("-", 1)[0]
        if prefix in ("chat", "consul", "notebook"):
            return prefix
    return "chat"


def available_tools_for_agent(agent: AgentConfig, cfg: AppConfig, branch: Branch) -> Dict[str, ToolConfig]:
    """
    Return the subset of tools that this agent may use in the current mode.

    Mirrors the checks in run_tool (mode allow-list and per-agent allowed_tools)
    so that prompt/tool wiring is consistent.
    """
    tools = getattr(cfg, "tools", {}) or {}
    mode = _infer_mode_from_branch(branch)
    allowed: Dict[str, ToolConfig] = {}

    for name, tcfg in tools.items():
        if _tool_allowed_for(agent, name, tcfg, mode):
            allowed[name] = tcfg

    return allowed


def ollama_tool_defs_for_agent(agent: AgentConfig, cfg: AppConfig, branch: Branch) -> list[dict]:
    """
    Build the list of Ollama tool definitions for this agent, based on config.

    We pull schemas from ToolConfig.settings["ollama_schema"] so config.py
    remains the single source of truth.
    """
    allowed = available_tools_for_agent(agent, cfg, branch)
    tools: list[dict] = []
    for name, tcfg in allowed.items():
        schema = (tcfg.settings or {}).get("ollama_schema")
        if schema:
            tools.append(schema)
    return tools

def _tool_allowed_for(
    agent: AgentConfig,
    tool_name: str,
    tcfg: ToolConfig,
    mode: str,
) -> bool:
    """
    Single source of truth for tool gating:

      • per-mode flags (ToolConfig.enabled_modes)
      • per-agent allow list (agent.allowed_tools)
    """
    # Mode gate
    if not tcfg.enabled_modes.get(mode, True):
        return False

    # Agent allow-list gate
    if agent.allowed_tools and tool_name not in agent.allowed_tools:
        return False

    return True



# ------------------- Tool Implementations ------------------- #

@register_tool("calc")
def _tool_calc(ctx: ToolContext, args: Any) -> ToolResult:
    """Simple calculator tool: evaluate a math expression safely."""
    # Support both:
    # - a raw string: "11232109 * 32190319"
    # - a dict from Ollama tools: {"expression": "..."}
    if isinstance(args, dict) and "expression" in args:
        expr = str(args["expression"])
    elif isinstance(args, str):
        expr = args
    else:
        expr = str(args)

    expr = expr.strip()
    if not expr:
        return ToolResult(error="Empty expression for calc tool.")

    try:
        # Very restricted eval: only digits, basic operators, parentheses, dot, whitespace.
        import re as _re

        if not _re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", expr):
            return ToolResult(error=f"Disallowed characters in expression: {expr!r}")

        allowed_names = {"__builtins__": {}}
        result = eval(expr, allowed_names, {})
        return ToolResult(output=str(result))
    except Exception as e:
        return ToolResult(error=f"Calculation error: {e}")

@register_tool("dice")
def _tool_dice(ctx: ToolContext, args: Any) -> ToolResult:
    """
    Dice roll tool: roll an N-sided die (default 6) or multiple dice.

    Accepts:
      - dict from Ollama tools:
          {"notation": "2d6"}
          {"sides": 20, "count": 2}
        (also tolerates {"expression": "..."} for compatibility with the
         generic string-arguments fallback in the runtime)
      - raw string: "2d6", "d20", "6"
      - integer/float: 6  (interpreted as d6)
    """
    sides = 6
    count = 1
    notation = ""

    # 1) Dict arguments from schema / Ollama tools
    if isinstance(args, dict):
        # Prefer explicit "notation", then "expression" for compatibility
        notation_val = args.get("notation") or args.get("expression") or ""
        if isinstance(notation_val, str):
            notation = notation_val.strip().lower()

        # If no usable notation, fall back to structured fields
        if not notation:
            if "sides" in args:
                try:
                    sides = int(args["sides"])
                except Exception:
                    pass
            if "count" in args:
                try:
                    count = int(args["count"])
                except Exception:
                    pass

    # 2) Raw string arguments
    elif isinstance(args, str):
        notation = args.strip().lower()

    # 3) Bare numeric arguments → number of sides
    elif isinstance(args, (int, float)):
        sides = int(args)

    # If we got a notation string, parse it (e.g. "2d6", "d20", "6")
    if notation:
        try:
            if "d" in notation:
                left, right = notation.split("d", 1)
                # "d20" → left == "" → default count = 1
                if left.strip():
                    count = int(left)
                if right.strip():
                    sides = int(right)
            else:
                # Just "6" → d6
                sides = int(notation)
        except Exception:
            # If parsing fails, fall back to defaults (1d6)
            sides = 6
            count = 1

    # Clamp to sensible limits
    sides = max(2, min(sides, 1000))
    count = max(1, min(count, 10))

    # Roll
    rolls = [random.randint(1, sides) for _ in range(count)]

    # Format output (transparent & human-friendly)
    if count == 1:
        return ToolResult(output=f"{rolls[0]} (d{sides})")

    total = sum(rolls)
    rolls_str = ", ".join(str(r) for r in rolls)
    return ToolResult(output=f"{rolls_str} (total {total}, {count}d{sides})")

# ------------------- py_repl / Notebook runner helpers ------------------- #

# Modules that are considered unsafe for agents to import via py_repl
_PY_REPL_BANNED_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "shlex",
    "shutil",
    "pathlib",
    "inspect",
    "socket",
    "requests",
    "urllib",
    "http",
    "ftplib",
    "telnetlib",
    "ssl",
    "ctypes",
    "multiprocessing",
    "threading",
    "signal",
    "builtins",
    "importlib",
    "pdb",
}

# Safe builtins for executing notebook_agent.py
_PY_REPL_SAFE_BUILTINS: dict[str, object] = {}
for _name, _obj in _builtins.__dict__.items():
    if _name in {"open", "__import__", "eval", "exec", "input"}:
        continue
    _PY_REPL_SAFE_BUILTINS[_name] = _obj

_PY_REPL_MAX_STEPS = 100_000
_PY_REPL_DEFAULT_MAX_OUTPUT = 20_000


def _py_repl_validate_imports(
    source: str,
    filename: str,
    *,
    forbid_any_import: bool,
) -> str | None:
    """
    Inspect import statements in a source file.

    - If forbid_any_import is True, any import is rejected.
    - Otherwise, imports of modules in _PY_REPL_BANNED_IMPORTS are rejected.
    Returns an error string or None if everything looks acceptable.
    """
    try:
        tree = ast.parse(source, filename=filename, mode="exec")
    except SyntaxError as e:
        return f"Syntax error in {filename}: {e}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if forbid_any_import:
                return (
                    f"Imports are not allowed in {filename} when executed via py_repl. "
                    "Ask the user to move all imports into the notebook setup cell."
                )

            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".", 1)[0] for alias in node.names]
            else:  # ImportFrom
                if node.module:
                    modules = [node.module.split(".", 1)[0]]

            for mod in modules:
                if mod in _PY_REPL_BANNED_IMPORTS:
                    return (
                        f"Import of '{mod}' is not allowed in {filename} when executed via py_repl. "
                        "Ask the user to remove it from the notebook."
                    )

    return None


def _py_repl_safe_exec_agent(
    source: str,
    ns: dict,
    *,
    filename: str,
    max_output_chars: int,
) -> dict[str, str]:
    """
    Execute notebook_agent.py source in a restricted environment:

    - No imports or direct __builtins__ access.
    - Limited builtins.
    - Execution step limit with sys.settrace.
    """
    try:
        tree = ast.parse(source, filename=filename, mode="exec")
    except SyntaxError:
        raise

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise RuntimeError(
                "Imports are not allowed in the agent notebook file when executed via py_repl."
            )
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "__builtins__":
                raise RuntimeError(
                    "Direct access to __builtins__ is restricted in the py_repl sandbox."
                )

    code_obj = compile(tree, filename, "exec")
    target_filename = code_obj.co_filename

    old_builtins = ns.get("__builtins__", _builtins)
    ns["__builtins__"] = _PY_REPL_SAFE_BUILTINS

    steps = 0

    def tracefunc(frame, event, arg):
        nonlocal steps
        if event == "line" and frame.f_code.co_filename == target_filename:
            steps += 1
            if steps > _PY_REPL_MAX_STEPS:
                raise TimeoutError(
                    f"Maximum execution steps exceeded ({_PY_REPL_MAX_STEPS}). "
                    "Your code may contain an infinite loop."
                )
        return tracefunc

    buf_out = io.StringIO()
    buf_err = io.StringIO()

    old_trace = sys.gettrace()
    sys.settrace(tracefunc)
    try:
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            exec(code_obj, ns)
        exc_str = ""
    except Exception:
        exc_str = traceback.format_exc()
    finally:
        sys.settrace(old_trace)
        ns["__builtins__"] = old_builtins

    return {
        "stdout": buf_out.getvalue()[:max_output_chars],
        "stderr": buf_err.getvalue()[:max_output_chars],
        "exception": exc_str[:max_output_chars],
    }

@register_tool("py_repl")
def _tool_py_repl(ctx: ToolContext, args: Any) -> ToolResult:
    """
    Notebook runner tool: execute the user's current notebook setup and agent files.

    Behaviour:
      - Only available in Notebook mode.
      - Reads `notebook_setup.py` and `notebook_agent.py` from the workspace directory,
        using filenames from cfg.notebook if available.
      - Validates imports:
          • setup file: imports allowed, but banned modules (os, sys, subprocess, …) are rejected.
          • agent file: imports are not allowed at all.
      - Executes:
          • setup file with normal builtins (user-provided imports).
          • agent file in a restricted sandbox (no imports, limited builtins, step limit).
    """
    # # 1) Enforce Notebook mode
    # mode = _infer_mode_from_branch(ctx.branch)
    # if mode != "notebook":
    #     return ToolResult(error="py_repl is only available in Notebook mode.")

    # 2) Determine max_output_chars from args or cfg.notebook
    max_output_chars = None
    if isinstance(args, dict):
        try:
            if "max_output_chars" in args and args["max_output_chars"] is not None:
                max_output_chars = int(args["max_output_chars"])
        except Exception:
            max_output_chars = None

    nb_cfg = getattr(ctx.cfg, "notebook", {}) or {}
    if max_output_chars is None or max_output_chars <= 0:
        max_output_chars = int(nb_cfg.get("max_output_chars", _PY_REPL_DEFAULT_MAX_OUTPUT))
    max_output_chars = max(100, min(max_output_chars, 50_000))

    # 3) Resolve workspace + notebook filenames
    workspace = ctx.cfg.workspace_dir or "workspace"
    os.makedirs(workspace, exist_ok=True)

    setup_name = nb_cfg.get("setup_filename", "notebook_setup.py")
    agent_name = nb_cfg.get("agent_filename", "notebook_agent.py")

    base = os.path.abspath(workspace)
    setup_path = os.path.abspath(os.path.join(base, setup_name))
    agent_path = os.path.abspath(os.path.join(base, agent_name))

    if not setup_path.startswith(base) or not agent_path.startswith(base):
        return ToolResult(error="Notebook files must live inside the workspace directory.")

    # 4) Read both files
    try:
        with open(setup_path, "r", encoding="utf-8") as f:
            setup_src = f.read()
    except FileNotFoundError:
        return ToolResult(error=f"Setup notebook file not found: {setup_name!r}")
    except Exception as e:
        return ToolResult(error=f"Error reading setup notebook file: {e}")

    try:
        with open(agent_path, "r", encoding="utf-8") as f:
            agent_src = f.read()
    except FileNotFoundError:
        return ToolResult(error=f"Agent notebook file not found: {agent_name!r}")
    except Exception as e:
        return ToolResult(error=f"Error reading agent notebook file: {e}")

    # 5) Validate imports per your safety rules
    #    - setup: imports allowed, but unsafe modules are banned
    err = _py_repl_validate_imports(
        setup_src,
        setup_name,
        forbid_any_import=False,
    )
    if err:
        return ToolResult(error=err)

    #    - agent: imports are not allowed at all
    err = _py_repl_validate_imports(
        agent_src,
        agent_name,
        forbid_any_import=True,
    )
    if err:
        return ToolResult(error=err)

    # 6) Execute setup (unsafe, but user-authored) and then agent (sandboxed)
    ns: dict[str, Any] = {"__name__": "__notebook__", "__file__": agent_name}

    # --- Run setup ---
    setup_out = io.StringIO()
    setup_err = io.StringIO()
    setup_exc = ""

    try:
        setup_code_obj = compile(setup_src, setup_name, "exec")
        with redirect_stdout(setup_out), redirect_stderr(setup_err):
            exec(setup_code_obj, ns)
    except Exception:
        setup_exc = traceback.format_exc()

    setup_stdout = setup_out.getvalue()[:max_output_chars]
    setup_stderr = setup_err.getvalue()[:max_output_chars]

    # --- Run agent in sandbox ---
    try:
        agent_result = _py_repl_safe_exec_agent(
            agent_src,
            ns,
            filename=agent_name,
            max_output_chars=max_output_chars,
        )
    except Exception as e:
        # If the sandbox itself blew up before we could format, show that
        tb = traceback.format_exc()
        return ToolResult(
            error=(
                "py_repl encountered an error while executing the agent notebook file.\n\n"
                f"Error: {e}\n\nTraceback:\n{tb}"
            )
        )

    # 7) Assemble a transparent, human-readable result
    parts: list[str] = []
    parts.append(
        "py_repl executed Notebook files in workspace:\n"
        f"- {setup_name}\n- {agent_name}\n\n"
    )

    if setup_stdout or setup_stderr or setup_exc:
        parts.append("=== notebook_setup.py ===\n")
        if setup_stdout:
            parts.append("stdout:\n")
            parts.append(setup_stdout)
            parts.append("\n")
        if setup_stderr:
            parts.append("stderr:\n")
            parts.append(setup_stderr)
            parts.append("\n")
        if setup_exc:
            parts.append("exception:\n")
            parts.append(setup_exc)
            parts.append("\n")
        parts.append("\n")

    parts.append("=== notebook_agent.py ===\n")
    if agent_result["stdout"]:
        parts.append("stdout:\n")
        parts.append(agent_result["stdout"])
        parts.append("\n")
    if agent_result["stderr"]:
        parts.append("stderr:\n")
        parts.append(agent_result["stderr"])
        parts.append("\n")
    if agent_result["exception"]:
        parts.append("exception:\n")
        parts.append(agent_result["exception"])
        parts.append("\n")

    if not (
        setup_stdout
        or setup_stderr
        or setup_exc
        or agent_result["stdout"]
        or agent_result["stderr"]
        or agent_result["exception"]
    ):
        parts.append("(no output)\n")

    text = "".join(parts)
    if len(text) > max_output_chars:
        text = text[:max_output_chars] + "\n...\n(output truncated by py_repl)"

    return ToolResult(output=text)


@register_tool("workspace_read")
def _tool_workspace_read(ctx: ToolContext, args: Any) -> ToolResult:
    """
    Read the contents of a text file from the workspace directory.

    Accepts:
      - dict from Ollama tools: {"file": "path/to/file"}
      - raw string: "path/to/file"
    """
    filename: Optional[str] = None

    if isinstance(args, dict):
        # Prefer the schema field name; tolerate a few synonyms
        filename = args.get("file") or args.get("path") or args.get("filename")
    else:
        filename = str(args)

    filename = (filename or "").strip()
    if not filename:
        return ToolResult(error="No filename provided for workspace_read.")

    workspace = ctx.cfg.workspace_dir or "workspace"
    os.makedirs(workspace, exist_ok=True)

    # Prevent path traversal outside the workspace
    base = os.path.abspath(workspace)
    filepath = os.path.abspath(os.path.join(base, filename))
    if not filepath.startswith(base):
        return ToolResult(error="Access outside the workspace directory is not allowed.")
    
    # --- Notebook-setup is read-only for agents in Notebook mode ---
    mode = _infer_mode_from_branch(ctx.branch)
    if mode == "notebook":
        nb_cfg = getattr(ctx.cfg, "notebook", {}) or {}
        setup_name = nb_cfg.get("setup_filename", "notebook_setup.py")

        # Compute the relative path inside workspace for comparison
        rel = os.path.relpath(filepath, base).replace("\\", "/")
        if rel == setup_name or rel.endswith("/" + setup_name):
            if not nb_cfg.get("allow_agent_write_setup", False):
                return ToolResult(
                    error=(
                        f"Agents are not allowed to modify the setup notebook file "
                        f"('{setup_name}'). Ask the user to edit it directly in the "
                        "Notebook UI instead."
                    )
                )
    # ------

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return ToolResult(output=content)
    except FileNotFoundError:
        return ToolResult(error=f"File not found: {filename!r}")
    except Exception as e:
        return ToolResult(error=f"Read error: {e}")


@register_tool("workspace_write")
def _tool_workspace_write(ctx: ToolContext, args: Any) -> ToolResult:
    """
    Write content to a file in the workspace directory.

    Schema-driven usage:
      {"file": "path/to/file", "content": "text..."}
    """
    if not isinstance(args, dict):
        return ToolResult(error="Args must be an object with 'file' and 'content' fields.")

    filename = (args.get("file") or args.get("path") or args.get("filename") or "").strip()
    content = args.get("content", "")

    if not filename:
        return ToolResult(error="Args must include a non-empty 'file' field.")

    workspace = ctx.cfg.workspace_dir or "workspace"
    os.makedirs(workspace, exist_ok=True)

    # Prevent path traversal outside the workspace
    base = os.path.abspath(workspace)
    filepath = os.path.abspath(os.path.join(base, filename))
    if not filepath.startswith(base):
        return ToolResult(error="Access outside the workspace directory is not allowed.")

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(str(content))
        return ToolResult(output=f"Wrote {len(str(content))} characters to {filename!r}")
    except Exception as e:
        return ToolResult(error=f"Write error: {e}")


@register_tool("conv_search")
def _tool_conv_search(ctx: ToolContext, args: Any) -> ToolResult:
    """
    Search within the current conversation (this branch) for messages
    containing a given query string.

    Accepts:
      - dict from Ollama tools:
          {"query": "paladin", "max_results": 5, "roles": ["user", "agent"]}
      - raw string: "paladin" (interpreted as query, using defaults)
    """
    # --- Parse arguments ---
    query = ""
    max_results = 5
    roles_filter: set[str] | None = None

    if isinstance(args, dict):
        query = str(args.get("query", "") or "").strip()
        try:
            if "max_results" in args and args["max_results"] is not None:
                max_results = int(args["max_results"])
        except Exception:
            max_results = 5

        # Optional roles filter
        raw_roles = args.get("roles")
        if isinstance(raw_roles, (list, tuple)):
            normalized = {str(r).strip().lower() for r in raw_roles if str(r).strip()}
            allowed = {"user", "agent", "system", "tool"}
            filt = normalized & allowed
            roles_filter = filt or None
    else:
        query = str(args or "").strip()

    if not query:
        return ToolResult(error="conv_search: missing 'query' text to search for.")

    max_results = max(1, min(max_results, 50))

    # --- Perform the search over the branch messages ---
    branch = ctx.branch
    messages = getattr(branch, "messages", []) or []

    q_lower = query.lower()
    matches = []
    for idx, msg in enumerate(messages):
        role = getattr(msg, "role", "") or ""
        sender = getattr(msg, "sender", "") or ""
        content = getattr(msg, "content", "") or ""

        if roles_filter is not None and role not in roles_filter:
            continue
        if not content:
            continue

        if q_lower in content.lower():
            # Build a small snippet
            snippet = content.strip().replace("\n", " ")
            if len(snippet) > 280:
                snippet = snippet[:277].rstrip() + "..."
            matches.append((idx, role, sender, snippet))
            if len(matches) >= max_results:
                break

    # --- Build a human-readable result ---
    if not matches:
        return ToolResult(
            output=(
                f"Conversation search for {query!r}: no matches found in this branch."
            )
        )

    lines: list[str] = []
    lines.append(
        f"Conversation search for {query!r} – found {len(matches)} match(es):\n"
    )

    for idx, role, sender, snippet in matches:
        # Example: [#5] (user / Alice): I rolled a 20 on the attack...
        tag_sender = f"{sender}" if sender else "(unknown sender)"
        lines.append(
            f"- [#{idx}] ({role} / {tag_sender}): {snippet}"
        )

    text = "\n".join(lines)
    return ToolResult(output=text)


@register_tool("world_state")
def _tool_world_state(ctx: ToolContext, args: Any) -> ToolResult:
    """Provide current world or environment state (e.g., current time)."""
    # Example: return current date and time
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    state_info = f"Current time: {now}"
    return ToolResult(output=state_info)

@register_tool("rag_search")
def _tool_rag_search(ctx: ToolContext, args: Any) -> ToolResult:
    """Retrieval-Augmented Generation (RAG) search: query a knowledge index."""
    query = None
    index_name = None
    if isinstance(args, dict):
        query = args.get("query") or args.get("q")
        index_name = args.get("index") or args.get("name")
    else:
        query = str(args)
    query = (query or "").strip()
    if not query:
        return ToolResult(error="No query provided for RAG search.")
    # Choose index: use specified or default to first available
    if not index_name:
        if ctx.cfg.rag_indexes:
            index_name = next(iter(ctx.cfg.rag_indexes.keys()))
        else:
            index_name = None
    rag_index = ctx.cfg.rag_indexes.get(index_name) if index_name else None
    if not rag_index:
        return ToolResult(error="No RAG index available or index not found.")
    # Perform a simple search (placeholder for actual vector DB search)
    # For demonstration, we'll pretend the index has a list of documents in rag_index.source (if it's a file with lines)
    retrieved = []
    try:
        # If rag_index.source is a path to a text file, search for query in it
        if os.path.isfile(rag_index.source):
            with open(rag_index.source, "r", encoding="utf-8") as f:
                docs = f.readlines()
        else:
            docs = []
        for doc in docs:
            if query.lower() in doc.lower():
                snippet = doc.strip()
                retrieved.append({"text": snippet, "score": 1.0, "source": rag_index.name})
                if len(retrieved) >= ctx.cfg.tools["rag_search"].settings.get("top_k", 3):
                    break
    except Exception as e:
        return ToolResult(error=f"RAG search error: {e}")
    # If no docs found, return output accordingly
    if not retrieved:
        return ToolResult(output="(no relevant documents found)")
    # Compile a context block from retrieved chunks (e.g., bullet list of snippets)
    lines = []
    for item in retrieved:
        txt = item.get("text", "")
        src = item.get("source", "")
        lines.append(f"- {txt} (source: {src})")
    context_block = "\n".join(lines)
    result = ToolResult(output="Knowledge retrieved.", retrieved=retrieved, context=context_block)
    return result
