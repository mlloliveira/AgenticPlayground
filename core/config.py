# core/config.py
import json, uuid, time
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any

@dataclass
class GenParams:
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 512
    num_ctx: int = 4096
    seed: Optional[int] = None
    def clamped(self):
        t = min(max(self.temperature, 0.0), 2.0)
        tp = min(max(self.top_p, 0.0), 1.0)
        tk = min(max(int(self.top_k), 0), 2048)
        mx = min(max(int(self.max_tokens), 1), 32768)
        nc = min(max(int(self.num_ctx), 256), 131072)
        sd = None if self.seed in (None, "", "None") else int(self.seed)
        return GenParams(t, tp, tk, mx, nc, sd)
    
@dataclass
class AgentSchedule:
    policy: str = "always"           # "always" | "first" | "last" | "every_n" | "range" | "never"
    every_n: int = 1                 # used only if policy == "every_n"
    start_round: int = 1             # used for "range" / "every_n" anchor
    end_round: Optional[int] = None  # used only if policy == "range"

@dataclass
class AgentConfig:
    # Stable identity to keep widget state pinned even if you insert/delete agents
    uid: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = "Agent"
    system_prompt: str = "You are an agent."
    model: str = "llama3:8b"
    enabled: bool = True
    params_override: Dict[str, Any] = field(default_factory=dict)
    think_api: bool = False
    allow_web: bool = False
    is_summarizer: bool = False  # <— replaces name-based detection
    # Added 07/Nov
    role_tag: str = "general"      # e.g., "debater", "judge", "narrator", "character", "planner"
    schedule: AgentSchedule = field(default_factory=AgentSchedule)

@dataclass
class AppConfig:
    same_model_for_all: bool = True
    global_model: str = "llama3:8b"
    global_params: GenParams = field(default_factory=GenParams)
    summarizer_enabled: bool = True
    summarizer_visible: bool = False
    colored_bubbles: bool = True
    show_prompt_preview: bool = True
    show_thinking: bool = True
    blind_first_turn: bool = True
    web_tool: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "max_chars": 8000, "ddg_results": 5, "all_agents": False})
    consul: Dict[str, Any] = field(default_factory=lambda: {"rounds": 3, "voting": "majority", "alpha": 2.0, "auto_web_brief": False})
    feature_flags: Dict[str, bool] = field(default_factory=lambda: {"warn_on_weird_sampling": True})
    agents: List[AgentConfig] = field(default_factory=list)
    markdown_all: bool = True   # when True, render all messages as Markdown cards
    show_denorm_tokens: bool = True # When True, request & display denormalized output tokens (via Ollama logprobs)
    # Configurable sidebar layout (block IDs, in order)
    sidebar_layout: List[str] = field(default_factory=lambda: [
        "MODELS",
        "SAVE_RESET",
        "PRESETS",
        "WEB",
        "UI",
        "AGENTS",
        "GLOBAL_PARAMS",
        "CONSUL",
        "USAGE",
    ])
    sidebar_expanded: Dict[str, bool] = field(default_factory=dict)

    def to_json(self): return json.dumps(asdict(self), indent=2)
    @staticmethod
    def from_json(s: str) -> "AppConfig":
        obj = json.loads(s)
        obj["global_params"] = GenParams(**obj.get("global_params", {}))
        # Let dataclass defaults fill missing fields (uid, is_summarizer, etc.)
        obj["agents"] = [AgentConfig(**a) for a in obj.get("agents", [])]
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
