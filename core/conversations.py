# core/conversations.py

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import streamlit as st
from cryptography.fernet import Fernet, InvalidToken

from core.config import Branch, Msg

PROJECT_APP_DIR = Path(__file__).resolve().parents[1]
CONV_SAVE_DIR = PROJECT_APP_DIR / "conversations"
CONV_SAVE_DIR.mkdir(parents=True, exist_ok=True)

FERNET_KEY_FILE = PROJECT_APP_DIR / "conversations.key"


# ---------- encryption helpers ----------

def _get_fernet() -> Optional[Fernet]:
    """
    Load or lazily create a Fernet key.
    If anything goes wrong, return None (we'll fall back to plaintext).
    """
    try:
        if not FERNET_KEY_FILE.exists():
            key = Fernet.generate_key()
            FERNET_KEY_FILE.write_bytes(key)
        else:
            key = FERNET_KEY_FILE.read_bytes()
        return Fernet(key)
    except Exception:
        return None


def _conv_path(name: str) -> Path:
    """Sanitize a human name into a filesystem-safe filename."""
    safe = "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip()
    if not safe:
        safe = "conversation"
    return CONV_SAVE_DIR / f"{safe}.bin"


# ---------- public API ----------

def list_saved_conversations() -> List[str]:
    """Return sorted list of saved conversation names (stem of file names)."""
    return sorted(p.stem for p in CONV_SAVE_DIR.glob("*.bin"))


def _serialize_msg(m: Msg) -> Dict[str, Any]:
    return {
        "role": m.role,
        "sender": m.sender,
        "content": m.content,
        "ts": float(getattr(m, "ts", 0.0)),
        "prompt_tokens": int(getattr(m, "prompt_tokens", 0)),
        "completion_tokens": int(getattr(m, "completion_tokens", 0)),
        "duration_s": float(getattr(m, "duration_s", 0.0)),
        "thinking": getattr(m, "thinking", None),
        "debug": getattr(m, "debug", {}) or {},
        "sources": getattr(m, "sources", []) or [],
        "id": getattr(m, "id", ""),
        "markdown": bool(getattr(m, "markdown", False)),
        "kind": getattr(m, "kind", "chat"),
    }


def _deserialize_msg(d: Dict[str, Any]) -> Msg:
    # Be robust against old/extra keys
    allowed = set(Msg.__dataclass_fields__.keys())
    clean = {k: v for k, v in d.items() if k in allowed}
    return Msg(**clean)


def save_current_branch(name: str) -> tuple[bool, str]:
    """
    Serialize the CURRENT ACTIVE BRANCH to disk.
    Payload includes:
      - mode
      - branch label + messages + summary
      - usage totals
    """
    if "branches" not in st.session_state or "active_branch_id" not in st.session_state:
        return False, "No active branch to save."

    mode = st.session_state.get("mode", "chat")
    brs = st.session_state.branches
    active_id = st.session_state.active_branch_id
    branch = brs.get(active_id)
    if branch is None:
        return False, "Active branch not found."

    payload = {
        "mode": mode,
        "branch": {
            "label": branch.label,
            "messages": [_serialize_msg(m) for m in branch.messages],
            "summary": branch.summary,
        },
        "totals": st.session_state.get("totals", {}),
    }

    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    f = _get_fernet()
    blob = f.encrypt(raw) if f else raw

    path = _conv_path(name)
    path.write_bytes(blob)
    return True, f"Saved conversation as “{path.stem}”."


def load_conversation_payload(name: str) -> Optional[Tuple[str, Branch, Dict[str, int]]]:
    """
    Read a saved conversation and return:
      (mode, branch, totals)

    NOTE: The returned Branch.id is NOT trusted; the caller decides how to
    scope it (we force it to main-<mode> in state.py).
    """
    path = _conv_path(name)
    if not path.exists():
        return None

    blob = path.read_bytes()
    f = _get_fernet()
    if f:
        try:
            raw = f.decrypt(blob)
        except InvalidToken:
            # maybe file was saved before encryption; fall back
            raw = blob
    else:
        raw = blob

    data = json.loads(raw.decode("utf-8"))

    mode = data.get("mode", "chat")
    bdata = data.get("branch", {}) or {}
    #label = bdata.get("label") or ""
    msgs = [ _deserialize_msg(m) for m in bdata.get("messages", []) ]
    summary = bdata.get("summary", "")

    branch = Branch(id="loaded-temp", label="", messages=msgs, summary=summary)  # Ignore stored label since we call it "main" when loading it
    totals = data.get("totals", {}) or {}

    return mode, branch, totals


def delete_conversation(name: str) -> None:
    path = _conv_path(name)
    if path.exists():
        path.unlink()
