# core/ollama_client.py
import os, json, time, requests
from typing import Iterator, Dict, Any, Optional
from .config import GenParams

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def _opts_from_params(params):
    # Accept GenParams or dict
    def _get(k, default=None):
        return params.get(k, default) if isinstance(params, dict) else getattr(params, k, default)
    opts = {
        "temperature": _get("temperature", 0.7),
        "top_p":       _get("top_p", 0.9),
        "top_k":       _get("top_k", 50),
        "num_ctx":     _get("num_ctx", 4096),
        "num_predict": _get("max_tokens", 512),
    }
    seed = _get("seed", None)
    if seed is not None:
        opts["seed"] = seed
    return opts

def generate_stream(
    model: str,
    prompt: str,
    params: GenParams,
    system: Optional[str] = None,
    keep_alive: str = "5m",
    think_api: bool = False,
    logprobs: bool = False,
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
    if think_api:
        payload["think"] = False
    if params.seed is not None:
        payload["options"]["seed"] = params.seed
    if logprobs:
        # ask Ollama to include denormalized tokens in each chunk
        payload["logprobs"] = True

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


def generate_once(model: str, system: str, prompt: str, params, logprobs: bool = False):
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


