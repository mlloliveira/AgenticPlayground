# core/ollama_client.py
import os, json, time, requests
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, Iterable, Union
from .config import GenParams

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def _opts_from_params(params):
    # Accept GenParams or dict
    def _get(k, default=None):
        return params.get(k, default) if isinstance(params, dict) else getattr(params, k, default)

    opts: Dict[str, Any] = {
        "temperature": _get("temperature", 0.7),
        "top_p":       _get("top_p", 0.9),
        "top_k":       _get("top_k", 50),
        "num_ctx":     _get("num_ctx", 4096),
        "num_predict": _get("max_tokens", 512),
    }

    seed = _get("seed", None)
    if seed is not None:
        opts["seed"] = seed

    # Advanced / custom options, e.g. min_p, mirostat, raw, think, etc.
    extra = _get("extra_options", {}) or {}
    if isinstance(extra, dict):
        for k, v in extra.items():
            if v is not None:
                opts[k] = v

    return opts

def _encode_images(images: Optional[Iterable[Union[str, bytes]]]) -> list[str]:
    """
    Best-effort encoder for the Ollama vision 'images' field.

    Accepts:
      • file paths (str) → read bytes and base64 encode
      • raw bytes        → base64 encode
      • other strings    → passed through (assumed already base64)

    Invalid/unreadable entries are skipped.
    """
    out: list[str] = []
    if not images:
        return out

    for img in images:
        try:
            if isinstance(img, bytes):
                data = img
            elif isinstance(img, str):
                p = Path(img)
                if p.exists() and p.is_file():
                    data = p.read_bytes()
                else:
                    # assume already base64 (or some special value) – pass through
                    out.append(img)
                    continue
            else:
                continue

            import base64
            out.append(base64.b64encode(data).decode("ascii"))
        except Exception:
            # best-effort; simply skip anything that fails
            continue
    return out


def generate_stream(
    model: str,
    prompt: str,
    params: GenParams,
    system: Optional[str] = None,
    keep_alive: str = "5m",
    #think_api: bool = False,
    logprobs: bool = False,
    images: Optional[Iterable[Union[str, bytes]]] = None,
) -> Iterator[Dict[str, Any]]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "keep_alive": keep_alive,
        "options": _opts_from_params(params),
    }
    if system:
        payload["system"] = system
    if params.seed is not None:
        payload["options"]["seed"] = params.seed
    if logprobs:
        # ask Ollama to include denormalized tokens in each chunk
        payload["logprobs"] = True
    if images:
        encoded = _encode_images(images)
        if encoded:
            payload["images"] = encoded

    with requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json=payload,
        stream=True,
        timeout=600,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = json.loads(line)
                yield data
                if data.get("done"):
                    break
            except Exception:
                continue


def generate_once(model: str, system: str, prompt: str, params, logprobs: bool = False, images: Optional[Iterable[Union[str, bytes]]] = None):
    """
    Non-streaming single turn using the SAME endpoint as streaming (/api/generate),
    with 'stream': False. Uses OLLAMA_HOST to match generate_stream.
    """
    url = f"{OLLAMA_HOST}/api/generate"
    opts = _opts_from_params(params)

    # carry through seed if present
    if isinstance(params, dict):
        sd = params.get("seed", None)
    else:
        sd = getattr(params, "seed", None)
    if sd is not None:
        opts["seed"] = sd

    payload = {
        "model": model,
        "prompt": prompt or "",
        "stream": False,
        "options": opts,
    }
    if system:
        payload["system"] = system
    if logprobs:
        payload["logprobs"] = True
    if images:
        encoded = _encode_images(images)
        if encoded:
            payload["images"] = encoded

    try:
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        # Ollama /api/generate returns {"response": "..."} for non-stream calls
        content = data.get("response", "") or (data.get("message") or {}).get("content", "")
        return {"content": content, **data}
    except requests.RequestException as e:
        # Planner/summarizer will gracefully skip if empty
        return {"content": "", "error": str(e)}
    
def chat_once_with_tools(
    model: str,
    messages: list[dict],
    tools: list[dict],
    params: GenParams,
    think: bool = True,
    logprobs: bool = False,
) -> dict:
    """
    Single HTTP call to Ollama /api/chat with optional tools.

    Returns the parsed JSON dict from Ollama. On HTTP/network error,
    returns {"error": "<message>"} so callers can surface a friendly warning.
    """
    url = f"{OLLAMA_HOST.rstrip('/')}/api/chat"
    opts = _opts_from_params(params)

    if isinstance(params, dict):
        sd = params.get("seed", None)
    else:
        sd = getattr(params, "seed", None)
    if sd is not None:
        opts["seed"] = sd
    
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "tools": tools or [],
        "think": think,
        "options": opts,
    }

    if logprobs:
        payload["logprobs"] = True

    try:
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        return data
    except requests.RequestException as e:
        return {"error": str(e)}


def list_running_models() -> list[dict]:
    """Return [{'name': 'llama3:8b', 'size': ..., 'expires_at': ...}, ...]"""
    r = requests.get(f"{OLLAMA_HOST}/api/ps", timeout=3)
    r.raise_for_status()
    return r.json().get("models", [])

def unload_model(model: str) -> dict:
    """
    Ask Ollama to unload immediately by sending an empty prompt with keep_alive=0.
    """
    payload = {"model": model, "prompt": "", "keep_alive": 0, "stream": False}
    r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=10)
    r.raise_for_status()
    return r.json() if r.headers.get("content-type","").startswith("application/json") else {}


