# core/agents.py
from __future__ import annotations
from typing import Optional
from .config import AppConfig, AgentConfig

__all__ = ["effective_params", "effective_model"]

def effective_params(agent: AgentConfig, cfg: AppConfig):
    """
    Compute generation params by starting from global defaults and applying
    per-agent overrides (if any). Returns a clamped params object.
    """
    gp = cfg.global_params.clamped()
    ov = agent.params_override or {}
    if "temperature" in ov: gp.temperature = float(ov["temperature"])
    if "top_p" in ov: gp.top_p = float(ov["top_p"])
    if "top_k" in ov: gp.top_k = int(ov["top_k"])
    if "max_tokens" in ov: gp.max_tokens = int(ov["max_tokens"])
    if "num_ctx" in ov: gp.num_ctx = int(ov["num_ctx"])
    if "seed" in ov: gp.seed = None if ov["seed"] in ("", None) else int(ov["seed"])
    return gp.clamped()

def effective_model(agent: AgentConfig, cfg: AppConfig) -> str:
    """
    Pick the model respecting the global “same model for all” flag.
    """
    return cfg.global_model if cfg.same_model_for_all else agent.model
