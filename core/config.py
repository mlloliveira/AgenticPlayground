# core/config.py
import json, uuid, time, copy
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any

@dataclass
class GenParams:
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 2048 #512
    num_ctx: int = 16384 #4096
    seed: Optional[int] = None
    extra_options: Dict[str, Any] = field(default_factory=dict)  # Arbitrary extra options passed straight through to Ollama's `options` dict.
    def clamped(self):
        t = min(max(self.temperature, 0.0), 2.0)
        tp = min(max(self.top_p, 0.0), 1.0)
        tk = min(max(int(self.top_k), 0), 2048)
        mx = min(max(int(self.max_tokens), 1), 32768)
        nc = min(max(int(self.num_ctx), 256), 131072)
        sd = None if self.seed in (None, "", "None") else int(self.seed)
        ex = dict(self.extra_options or {})
        return GenParams(
            temperature=t,
            top_p=tp,
            top_k=tk,
            max_tokens=mx,
            num_ctx=nc,
            seed=sd,
            extra_options=ex,
        )
    
@dataclass
class AgentSchedule:
    policy: str = "always"           # "always" | "first" | "last" | "every_n" | "range" | "never"
    every_n: int = 1                 # used only if policy == "every_n"
    start_round: int = 1             # used for "range" / "every_n" anchor
    end_round: Optional[int] = None  # used only if policy == "range"

### Tools

@dataclass
class ToolConfig:
    name: str
    #enabled: bool = True # Deprecated: use enabled_modes instead.
    enabled_modes: Dict[str, bool] = field(default_factory=lambda: {"chat": False, "consul": False, "notebook": False})
    # `settings` can hold tool-specific options (e.g. for RAG: index name, etc.)
    settings: Dict[str, Any] = field(default_factory=dict)
    label: str = ""       # Human-friendly display name.
    description: str = "" # Short description shown in the sidebar / context inspector.

@dataclass
class RagIndexConfig:
    name: str
    source: str              # e.g. path or identifier for the index data
    description: str = ""
    embeddings_model: Optional[str] = None

###

@dataclass
class AgentConfig:
    # Stable identity to keep widget state pinned even if you insert/delete agents
    uid: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = "Agent"
    system_prompt: str = "You are an agent."
    model: str = "llama3:8b"
    enabled: bool = True
    params_override: Dict[str, Any] = field(default_factory=dict)
    #think_api: bool = False
    allow_web: bool = False
    is_summarizer: bool = False  # <— replaces name-based detection
    # Added 07/Nov/25
    role_tag: str = "general"      # e.g., "debater", "judge", "narrator", "character", "planner"
    schedule: AgentSchedule = field(default_factory=AgentSchedule)
    # Added 25/Nov/25 - List of tool names this agent is allowed to use (empty means no restriction/all tools allowed)
    allowed_tools: List[str] = field(default_factory=list)

@dataclass
class AppConfig:
    same_model_for_all: bool = True
    global_model: str = "gemma3:12b"#"llama3:8b"
    global_params: GenParams = field(default_factory=GenParams)
    summarizer_enabled: bool = True
    summarizer_visible: bool = False
    colored_bubbles: bool = True
    show_prompt_preview: bool = True
    show_thinking: bool = True
    blind_first_turn: bool = True
    show_time: bool = True # Added 10/Dez/25 
    web_tool: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "max_chars": 8000, "ddg_results": 5, "all_agents": False})
    consul: Dict[str, Any] = field(default_factory=lambda: {"rounds": 3, "voting": "majority", "alpha": 2.0, "auto_web_brief": False})
    feature_flags: Dict[str, bool] = field(default_factory=lambda: {"warn_on_weird_sampling": True,
                                                                    "ollama_tools": False, # <— Global toggle for /api/chat tools
                                                                    })
    agents: List[AgentConfig] = field(default_factory=list)
    markdown_all: bool = True   # when True, render all messages as Markdown cards
    show_denorm_tokens: bool = True # When True, request & display denormalized output tokens (via Ollama logprobs)
    # Configurable sidebar layout (block IDs, in order)
    sidebar_layout: List[str] = field(
        default_factory=lambda: DEFAULT_SIDEBAR_LAYOUT.copy()
    )
    sidebar_expanded: Dict[str, bool] = field(default_factory=dict)
    #ViT preview settings (single source of truth)
    vision: Dict[str, Any] = field(
        default_factory=lambda: {
            "preview_size": 224,      # conceptual ViT input size
            "patch_size": 16,         # conceptual patch size
            "show_patch_grid": True,  # toggle visualization
        }
    )
    # Global tools configuration and available RAG indexes
    tools: Dict[str, ToolConfig] = field(default_factory=lambda: {name: copy.deepcopy(tc) for name, tc in DEFAULT_TOOLS.items()})
    rag_indexes: Dict[str, RagIndexConfig] = field(default_factory=dict)
    workspace_dir: str = "workspace"

    # Notebook mode
    notebook: Dict[str, Any] = field(
        default_factory=lambda: {
            "rounds": 3,
            "autosave": True,
            "max_output_chars": 20_000,
            "setup_filename": "notebook_setup.py",
            "agent_filename": "notebook_agent.py",
            "notes_filename": "notebook_notes.md",
            "allow_agent_write_setup": False, #Allow agents to write in the setup file
        }
    )

    def to_json(self): return json.dumps(asdict(self), indent=2)
    @staticmethod
    def from_json(s: str) -> "AppConfig":
        obj = json.loads(s)
        obj["global_params"] = GenParams(**obj.get("global_params", {}))
        # Let dataclass defaults fill missing fields (uid, is_summarizer, etc.)
        #obj["agents"] = [AgentConfig(**a) for a in obj.get("agents", [])]
        obj["agents"] = [agent_from_dict(a) for a in obj.get("agents", [])] # Use tolerant loader for agents to ignore old fields like "think_api"
        return AppConfig(**obj)


@dataclass
class Msg:
    role: str     # "user" | "agent" | "summary"
    sender: str
    content: str
    ts: float = field(default_factory=time.time)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_s: float = 0.0
    thinking: Optional[str] = None
    debug: Dict[str, Any] = field(default_factory=dict)
    sources: List[Dict[str, str]] = field(default_factory=list)
    images: List[str] = field(default_factory=list)  # paths to attached images
    denorm_tokens: List[str] = field(default_factory=list) # denormalized output tokens (from Ollama logprobs); only for assistant / summary
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    markdown: bool = False       # render body as Markdown if True
    kind: str = "chat"           # "chat" (default) or "final" (special card styling)

# Added on 07/Nov:
@dataclass
class Preset:
    name: str
    description: str = ""
    global_model: Optional[str] = None
    global_params: Dict[str, Any] = field(default_factory=dict)
    agents: List[AgentConfig] = field(default_factory=list)
    sidebar_layout: List[str] = field(default_factory=list)
    sidebar_expanded: Dict[str, bool] = field(default_factory=dict)

def preset_to_dict(p: Preset) -> Dict[str, Any]:
    return {
        "name": p.name,
        "description": p.description,
        "global_model": p.global_model,
        "global_params": p.global_params,
        "agents": [agent_to_dict(a) for a in p.agents],
        "sidebar_layout": p.sidebar_layout,
        "sidebar_expanded": p.sidebar_expanded,
    }

def preset_from_dict(d: Dict[str, Any]) -> Preset:
    return Preset(
        name=d.get("name", "Unnamed"),
        description=d.get("description", ""),
        global_model=d.get("global_model"),
        global_params=d.get("global_params", {}),
        agents=[agent_from_dict(x) for x in d.get("agents", [])],
        sidebar_layout=d.get("sidebar_layout") or [],
        sidebar_expanded=d.get("sidebar_expanded") or {},
    )


@dataclass
class Branch:
    id: str
    label: str
    messages: List[Msg] = field(default_factory=list)
    summary: str = ""


def agent_to_dict(a: AgentConfig) -> Dict[str, Any]:
    d = asdict(a)  # safe for nested dataclasses
    return d

def agent_from_dict(d: Dict[str, Any]) -> AgentConfig:
    # tolerate missing new fields in old presets
    sched = d.get("schedule", {}) or {}
    d["schedule"] = AgentSchedule(**sched) if not isinstance(sched, AgentSchedule) else sched
    d.setdefault("role_tag", "general")
    return AgentConfig(**{k: v for k, v in d.items() if k in AgentConfig.__dataclass_fields__})

#--- Default Agents per mode ---#

# Defaults (Chat)
DEFAULT_CHAT_AGENTS = [
    AgentConfig(name="Agent A",
        system_prompt="You are Agent A, a pragmatic strategist. Be helpful, but concise and practical.",
        role_tag="debater", is_summarizer=False),
    AgentConfig(name="Agent B",
        system_prompt="You are Agent B, a critical analyst. Be helpful, but challenge assumptions respectfully.",
        role_tag="debater", is_summarizer=False),
    AgentConfig(name="Summarizer",
        system_prompt="You are a neutral summarizer. Only summarize; do not add opinions.",
        role_tag="summarizer", is_summarizer=True),
]

#Defaults (Consul)
CONSUL_BASE_PROMPT = " You are a member of the Consul. Propose an answer to the user's prompt. Argue with the other Agents either agreeing or disagreeing with their proposal. " \
"A valid counter-argument is accompanied by an alternative answer to the user's prompt. Cite evidence when possible, " \
"and be convincing. At the end of the debate you will VOTE for the best final answer among the participants."

DEFAULT_CONSUL_AGENTS = [
    AgentConfig(name="Agent Alpha",
        system_prompt="You are Agent Alpha, a pragmatic strategist. Be concise and practical." + CONSUL_BASE_PROMPT,
        role_tag="debater"),
    AgentConfig(name="Agent Beta",
        system_prompt="You are Agent Beta, a critical analyst. Be concise and challenge assumptions." + CONSUL_BASE_PROMPT,
        role_tag="debater"),
    AgentConfig(name="Agent Gamma",
        system_prompt="You are Agent Gamma, an intellectual. Be concise, smart and thoughtful." + CONSUL_BASE_PROMPT,
        role_tag="debater"),
]

# --- Notebook base prompts ---

NOTEBOOK_BASE_PROMPT = """
You are in *Notebook* mode.

The user works with three files in a sandbox workspace:
- `notebook_notes.md`   – markdown notes and commentary.
- `notebook_setup.py`   – UNSAFE setup code for imports and helper functions (user-owned).
- `notebook_agent.py`   – SAFE agent code that uses helpers defined in the setup file.

You can access these files only through the tools:
- workspace_read(file)
- workspace_write(file, content)

Rules:
- Never guess what a file contains. Call workspace_read first.
- When you change a file, use workspace_write with the full new content of that file (not a diff).
- Code in `notebook_agent.py` must NOT contain `import` statements; rely on names defined in `notebook_setup.py`.
- Normally you do NOT modify `notebook_setup.py`. If new imports or helpers are needed, explain clearly what the user should add there.
- Keep your chat messages short and concrete, and always mention which files you read or updated.
- Other agents may also help; you only control your own turn. Focus on a small, useful improvement each time you speak.
"""


DEFAULT_NOTEBOOK_AGENTS = [
    AgentConfig(
        name="Agent 1Planner",
        system_prompt=(
            "You are Agent 1Planner. Your job is to understand the user's goal and propose "
            "a small, concrete plan for the notebook.\n\n"
            "On each turn:\n"
            "1. Call workspace_read on `notebook_notes.md`, `notebook_setup.py`, "
            "   and `notebook_agent.py` to see the current state.\n"
            "2. Restate the user's goal in 1–3 short sentences.\n"
            "3. Propose a numbered list of simple steps, saying which file each step will touch.\n"
            "4. If the code needs new imports or helper functions, say exactly what should go into "
            "`notebook_setup.py` (but do not write that file yourself).\n\n"
            "Keep your message short. Do not try to do all the writing or coding yourself; "
            "focus on giving clear tasks that a writer agent can execute.\n\n"
        ) + NOTEBOOK_BASE_PROMPT,
        role_tag="debater",
    ),
    AgentConfig(
        name="Agent 2Writter",
        system_prompt=(
            "You are Agent 2Writter. Your job is to actually write and update the notebook files.\n\n"
            "On each turn:\n"
            "1. Decide which files are relevant (usually `notebook_notes.md` and/or `notebook_agent.py`).\n"
            "2. Call workspace_read on each relevant file to see the current content.\n"
            "3. Prepare the full new content for each file that should change.\n"
            "4. Call workspace_write for each changed file with the full updated content.\n"
            "5. In your chat message, briefly describe what you changed in each file.\n\n"
            "If the user asked for any change to notes or code, do NOT finish your turn "
            "without at least one workspace_write call.\n"
            "Normally you do NOT edit `notebook_setup.py`; instead, ask the user to add or adjust "
            "imports/helpers there if needed.\n\n"
        ) + NOTEBOOK_BASE_PROMPT,
        role_tag="debater",
    ),
    AgentConfig(
        name="Agent 3Editor",
        system_prompt=(
            "You are Agent 3Editor. Your job is to review and lightly improve the existing notebook content.\n\n"
            "On each turn:\n"
            "1. Call workspace_read on `notebook_notes.md` and `notebook_agent.py` to see the current content.\n"
            "2. Check for clarity, correctness, and alignment with the user's goal.\n"
            "3. Either:\n"
            "   - Suggest improvements in your chat message (pointing to specific parts of the files), or\n"
            "   - For small, clearly helpful fixes, apply them directly with workspace_write.\n"
            "4. Keep your feedback and changes focused; do not invent entirely new directions.\n\n"
            "Your messages should be short and specific about what is good, what is bad, and what you changed.\n\n"
        ) + NOTEBOOK_BASE_PROMPT,
        role_tag="debater",
    ),
]


# --- Sidebar default configuration ---

DEFAULT_SIDEBAR_LAYOUT: List[str] = [
    "MODELS",
    "SAVE_RESET",
    "PRESETS",
    "WEB",
    "UI",
    "CONSUL",
    "AGENTS",
    "GLOBAL_PARAMS",
    "USAGE",
]

DEFAULT_SIDEBAR_EXPANDED: Dict[str, bool] = {
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

# --- Default tool configurations  ---

DEFAULT_TOOLS: Dict[str, ToolConfig] = {
    "calc": ToolConfig(
        name="calc",
        enabled_modes = {"chat": True},
        label="Calculator",
        description="Evaluate a mathematical expression and return the exact numeric result.",
        settings={
            # Schema used when calling Ollama /api/chat tools
            "ollama_schema": {
                "type": "function",
                "function": {
                    "name": "calc",
                    "description": "Evaluate a mathematical expression and return the exact numeric result.",
                    "parameters": {
                        "type": "object",
                        "required": ["expression"],
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": (
                                    "A purely arithmetic expression, e.g. "
                                    "'2 + 2', '11232109 * 32190319', '(1+2)*3'."
                                ),
                            }
                        },
                    },
                },
            },
        },
    ),
    "dice": ToolConfig(
        name="dice",
        enabled_modes={"chat": True},
        label="Dice / RNG",
        description="Roll pseudo-random dice for games or simulations.",
        settings={
            # Schema used when calling Ollama /api/chat tools
            "ollama_schema": {
                "type": "function",
                "function": {
                    "name": "dice",
                    "description": (
                        "Roll pseudo-random dice for games or simulations. "
                        "Use standard dice notation like 'd6', '2d8', '5d20', or '10d100' "
                        "or specify sides/count explicitly. Can also be used to generate random numbers."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "notation": {
                                "type": "string",
                                "description": (
                                    "Dice notation like 'd6', '2d6', 'd20', '3d100'. "
                                    "If provided, this takes precedence over other fields."
                                ),
                            },
                            "sides": {
                                "type": "integer",
                                "minimum": 2,
                                "maximum": 1000,
                                "description": (
                                    "Number of sides on each die (2–1000). "
                                    "Used when notation is not given."
                                ),
                            },
                            "count": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 10,
                                "description": (
                                    "Number of dice to roll (1–10). "
                                    "Used when notation is not given."
                                ),
                            },
                        },
                        # All fields optional; default is 1d6
                    },
                },
            },
        },
    ),

    "py_repl": ToolConfig(
        name="py_repl",
        #enabled_modes = {"notebook": True},
        label="Python REPL",
        description="Execute small Python snippets in an isolated workspace folder in a restricted sandbox.",
        settings={
            "ollama_schema": {
                "type": "function",
                "function": {
                    "name": "py_repl",
                    "description": (
                        "Execute the user's current notebook code by running `notebook_setup.py` "
                        "and then `notebook_agent.py` from the workspace directory. "
                        "Imports of unsafe modules (like `os` or `sys`) are rejected. "
                        "Only these two files may be executed."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "max_output_chars": {
                                "type": "integer",
                                "minimum": 100,
                                "maximum": 50000,
                                "description": (
                                    "Optional limit on combined stdout/stderr/traceback returned. "
                                    "If omitted, a per-app default is used."
                                ),
                            },
                        },
                    },
                },
            },
        },
    ),
    "workspace_read": ToolConfig(
        name="workspace_read",
        enabled_modes = {"chat": True, "notebook": True},
        label="Workspace Read",
        description="Read the contents of a text file from the sandbox workspace.",
        settings={
            "ollama_schema": {
                "type": "function",
                "function": {
                    "name": "workspace_read",
                    "description": (
                        "Read the contents of a text file from the sandbox workspace directory. "
                        "Use this instead of guessing what is in files."
                    ),
                    "parameters": {
                        "type": "object",
                        "required": ["file"],
                        "properties": {
                            "file": {
                                "type": "string",
                                "description": (
                                    "Path to the file, relative to the workspace root. "
                                    "For example: 'notebook_notes.md', 'notebook_setup.py'."
                                ),
                            },
                        },
                    },
                },
            },
        },
    ),
    "workspace_write": ToolConfig(
        name="workspace_write",
        enabled_modes = {"chat": True, "notebook": True},
        label="Workspace Write",
        description="Write or overwrite a text file in the sandbox workspace.",
        settings={
            "ollama_schema": {
                "type": "function",
                "function": {
                    "name": "workspace_write",
                    "description": (
                        "Write or overwrite the contents of a text file in the sandbox workspace "
                        "directory. Use this to persist notes, code, or notebook cells."
                    ),
                    "parameters": {
                        "type": "object",
                        "required": ["file", "content"],
                        "properties": {
                            "file": {
                                "type": "string",
                                "description": (
                                    "Path to the file, relative to the workspace root. "
                                    "For example: 'notebook_notes.md', 'notebook_agent.py'."
                                ),
                            },
                            "content": {
                                "type": "string",
                                "description": (
                                    "The full text content to write into the file. "
                                    "This will replace any existing contents."
                                ),
                            },
                        },
                    },
                },
            },
        },
    ),
    "conv_search": ToolConfig(
        name="conv_search",
        enabled_modes = {"chat": True},
        label="Conversation Search",
        description="Search within the current conversation instead of guessing. Useful to find where something was said earlier in this branch.",
        settings={
            "ollama_schema": {
                "type": "function",
                "function": {
                    "name": "conv_search",
                    "description": (
                        "Search through the current conversation (the active branch) for messages "
                        "containing a given query string. Use this instead of guessing what was "
                        "said earlier."
                    ),
                    "parameters": {
                        "type": "object",
                        "required": ["query"],
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "Text to search for in conversation messages. "
                                    "Matching is case-insensitive substring search."
                                ),
                            },
                            "max_results": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 50,
                                "description": (
                                    "Maximum number of matching messages to return. "
                                    "Defaults to 5 if omitted."
                                ),
                            },
                            "roles": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["user", "agent", "system", "tool"],
                                },
                                "description": (
                                    "Optional filter: only consider messages whose role is in this list. "
                                    "If omitted, all roles are searched."
                                ),
                            },
                        },
                    },
                },
            },
        },
    ),
    # "world_state": ToolConfig(
    #     name="world_state",
    #     #enabled_modes = {"chat": False, "consul": False},
    #     label="World State",
    #     description="Expose simple world / environment state (e.g. current time).",
    # ),
    "rag_search": ToolConfig(
        name="rag_search",
        #enabled_modes = {"chat": False, "consul": False},
        label="RAG Search",
        description="Retrieve relevant snippets from a configured RAG index.",
        settings={"index": "", "top_k": 3},
    ),
}
