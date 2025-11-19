# core/runtime_web.py
from __future__ import annotations

import re, json
from typing import List, Dict, Optional, Tuple

from core.config import AppConfig, Branch, AgentConfig
from core.state import is_planner
from core.ollama_client import generate_once
from core.web_tools import web_search, summarise_sources
from core.agents import effective_params, effective_model

# Recognize manual queries like:  "#web: what is RLHF?"
WEB_DIRECTIVE = re.compile(r"(?im)^\s*#web:?[\s]+(.+)$")


# ---------- small helpers (self-contained) ----------

def _clip(txt: str, max_chars: int = 1600) -> str:
    """Trim long summaries to keep the planner prompt compact."""
    txt = (txt or "").strip()
    if len(txt) <= max_chars:
        return txt
    cut = txt.rfind(".", 0, max_chars)
    if cut == -1:
        cut = max_chars
    return txt[:cut].strip()

def _normalize_query(q: str) -> str:
    """Make a sentence more search-friendly (remove punctuation & squeeze spaces)."""
    import re as _re
    q = _re.sub(r'[.,;:!?\"()\[\]{}]', '', q)
    q = _re.sub(r'\s+', ' ', q).strip()
    return q

def _first_sentence(s: str) -> str:
    for sep in ["?", "!", "."]:
        if sep in s:
            return s.split(sep, 1)[0].strip()
    return s.strip()

def _extract_json_obj(text: str) -> Optional[dict]:
    """
    Robustly extract a JSON object from model output.
    - Strips ```json fences
    - Falls back to first {...} block if present
    """
    if not text:
        return None

    fenced = re.match(r"\s*```(?:json)?\s*(.*?)\s*```\s*$", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        blob = text[start:end+1]
        try:
            obj = json.loads(blob)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


# ---------- planner (silent) ----------

def _run_planner(cfg: AppConfig, branch: Branch) -> list[str]:
    """Run the Web Planner silently and return list[str] queries or [] on failure."""
    if not cfg.web_tool.get("enabled", False):
        return []

    planner = next((a for a in cfg.agents if a.enabled and is_planner(a)), None)
    if not planner:
        return []

    # last user message
    last_user = next((i for i in range(len(branch.messages)-1, -1, -1)
                      if branch.messages[i].role == "user"), None)
    if last_user is None:
        return []
    user_text = (branch.messages[last_user].content or "").strip()
    if not user_text:
        return []

    params = effective_params(planner, cfg)
    model = effective_model(planner, cfg)

    # Give the planner context via the branch summary (if available), plus the latest user message
    summary = _clip(branch.summary or "", 1600)
    if summary:
        planner_prompt = f"Conversation summary (for context):\n{summary}\n\nUser question:\n{user_text}"
    else:
        planner_prompt = user_text

    resp = generate_once(model=model, system=planner.system_prompt, prompt=planner_prompt, params=params)
    raw = resp.get("content", "") if isinstance(resp, dict) else str(resp)
    obj = _extract_json_obj(raw)
    queries = []
    if isinstance(obj, dict) and isinstance(obj.get("queries"), list):
        for q in obj["queries"][:5]:
            qn = _normalize_query(str(q))
            if qn and qn.lower() not in [x.lower() for x in queries]:
                queries.append(qn)
    return queries


# ---------- public: build the web brief for the latest user message ----------

def web_brief_for_branch(
    cfg: AppConfig,
    branch: Branch,
) -> Tuple[str, List[Dict[str, str]], List[str], str, Optional[str]]:
    """
    Build a brief from the web for the latest user message in `branch`.

    Priority:
      1) Manual '#web:' lines (first sentence of each)
      2) Else, planner-generated queries (if internet is ON and planner exists)
    Then:
      - Run DDG search for each query (core.web_tools.web_search)
      - Summarize hits into a brief (core.web_tools.summarise_sources)
    Returns:
      (brief_text, sources_std, queries, origin, planner_system_prompt)
      where origin ∈ {"manual", "planner", ""}.
    """
    if not cfg.web_tool.get("enabled", False):
        return "", [], [], "", None

    # last user message in this branch
    last_user_idx = None
    for i in range(len(branch.messages) - 1, -1, -1):
        if branch.messages[i].role == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        return "", [], [], "", None

    user_text = (branch.messages[last_user_idx].content or "").strip()
    if not user_text:
        return "", [], [], "", None

    # 1) manual #web: (first sentence only)
    manual: List[str] = []
    for line in user_text.splitlines():
        m = WEB_DIRECTIVE.match(line)
        if m:
            s = _first_sentence(m.group(1))
            manual.append(_normalize_query(s))

    queries: List[str] = []
    for q in manual:
        if q and q.lower() not in [x.lower() for x in queries]:
            queries.append(q)

    origin = "manual" if queries else ""
    planner_sp = None

    # 2) planner fallback if nothing manual
    if not queries and cfg.web_tool.get("enabled", False):
        # Only if a planner agent exists
        if any(a.enabled and is_planner(a) for a in cfg.agents):
            queries = _run_planner(cfg, branch)
            if queries:
                origin = "planner"
                planner_sp = next((a.system_prompt for a in cfg.agents if is_planner(a)), None)

    if not queries:
        return "", [], [], "", None

    # 3) search + brief
    hits_raw: List[Dict[str, str]] = []
    for q in queries:
        hits_raw.extend(web_search(q, k=int(cfg.web_tool.get("ddg_results", 5))))

    # normalize to {title,url,snippet}
    sources_std: List[Dict[str, str]] = []
    for h in hits_raw:
        title = h.get("title") or "(no title)"
        url = h.get("url") or h.get("href") or ""
        snippet = h.get("snippet") or h.get("body") or ""
        row = {"title": title, "url": url, "snippet": snippet}
        if "_ddg_backend" in h:
            row["_ddg_backend"] = h["_ddg_backend"]
        sources_std.append(row)

    brief = summarise_sources(hits_raw, max_chars=int(cfg.web_tool.get("max_chars", 8000)))
    return brief, sources_std, queries, origin, planner_sp
