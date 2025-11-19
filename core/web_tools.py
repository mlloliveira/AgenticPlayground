# core/web_tools.py
from __future__ import annotations
from typing import List, Dict, Tuple
from urllib.parse import urlparse
from ddgs import DDGS  # NEW package name

# One shared client (avoid rate-limit errs by reusing the same session)
_DDG: DDGS | None = None

DEFAULT_REGION = "us-en"        # English results
DEFAULT_SAFESEARCH = "moderate" # "off"|"moderate"|"strict"

def _client() -> DDGS:
    global _DDG
    if _DDG is None:
        _DDG = DDGS(timeout=20)
    return _DDG

def _normalize(results: list, k: int) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for r in results[:k]:
        title = r.get("title") or r.get("heading") or ""
        url = r.get("href") or r.get("url") or r.get("link") or ""
        snippet = r.get("body") or r.get("snippet") or r.get("excerpt") or ""
        out.append({"title": title, "url": url, "snippet": snippet})
    return out

def _looks_incoherent(q: str, url: str) -> bool:
    """Drop obviously mismatched domains (simple heuristics)."""
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    ql = q.lower()
    if "whatsapp" in host and "whatsapp" not in ql:
        return True
    if "support.google.com" in host and "android" in host and "android" not in ql:
        return True
    return False

def web_search(query: str, k: int = 5, region: str = DEFAULT_REGION,
               timelimit: str | None = None) -> List[Dict[str, str]]:
    """
    Robust DDG text search with backend rotation:
      try 'api' → 'html' → 'lite'. Returns [{title,url,snippet}...]
    """
    backends = ["api", "html", "lite"]
    last_err = None
    for be in backends:
        try:
            ddg = _client()
            raw = ddg.text(
                query,
                region=region,
                safesearch=DEFAULT_SAFESEARCH,
                timelimit=timelimit,
                backend=be,
                max_results=max(10, k),
            )
            items = _normalize(raw, max(10, k))
            items = [x for x in items if x.get("url") and not _looks_incoherent(query, x["url"])]
            if items:
                items = items[:k]
                items[0]["_ddg_backend"] = be  # expose which backend worked
                return items
        except Exception as e:
            last_err = str(e)
            continue
    return [{"title": "(web error)", "url": "", "snippet": last_err or "no results"}]

def summarise_sources(sources: List[Dict[str, str]], max_chars: int = 8000) -> str:
    if not sources:
        return ""
    lines: List[str] = []
    for s in sources:
        title = s.get("title") or "(no title)"
        url = s.get("url") or ""
        snippet = s.get("snippet") or ""
        lines.append(f"- **{title}** — {snippet}\n  {url}")
    brief = "\n".join(lines)
    return brief[:max_chars]

def web_smoke_test() -> Tuple[str, List[Dict[str, str]]]:
    q = "trolley problem utilitarian critique"
    hits = web_search(q, k=3)
    be = hits[0].get("_ddg_backend", "?") if hits else "?"
    return be, hits
