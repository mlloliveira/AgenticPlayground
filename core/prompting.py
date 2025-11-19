# core/prompting.py
import re
from .config import AgentConfig, Branch, AppConfig
from time import gmtime, strftime

THINK_PATTERNS = [
    (r"(?is)<think>([\s\S]*?)</think>", True),
    (r"(?is)\[THINK\]([\s\S]*?)\[/THINK\]", False),
    (r"(?is)Thoughts:\s*(.+?)(?:\n\s*Answer:|\Z)", False),
    (r"(?is)Reasoning:\s*(.+?)(?:\n\s*(?:Final|Answer):|\Z)", False),
]

def parse_thinking_keep_tags(text: str):
    for pat, _ in THINK_PATTERNS:
        m = re.search(pat, text)
        if m:
            raw = m.group(1).strip()
            thinking = f"<think>\n{raw}\n</think>"
            answer = (text[:m.start()] + text[m.end():]).strip()
            answer = re.sub(r"(?is)<think>[\s\S]*?</think>", "", answer).strip()
            return thinking, answer
    return None, text

def build_prompt_for_agent(agent: AgentConfig, branch: Branch, cfg: AppConfig, extra_preface: str = "") -> str:
    show_time=False
    lines = []
    if extra_preface.strip():
        lines.append("[WEB RESULTS]"); lines.append(extra_preface.strip()); lines.append("")
    if branch.summary.strip():
        lines.append("[SUMMARY SO FAR]"); lines.append(branch.summary.strip()); lines.append("")
    lines.append("[CONVERSATION - most recent last]")
    for m in branch.messages:
        lines.append(f"{m.sender}: {m.content.strip()}")
    if show_time:
        now = strftime("%Y-%m-%d", gmtime())
        lines.append("")
        lines.append("[CURRENT DAY]")
        lines.append(f"Today is: {now}. Time format: YYYY-MM-DD. ")
    lines.append("")
    lines.append("[TASK]")
    lines.append(f"You are {agent.name}. Write your next message for the conversation above. "
                 f"Do not include speaker tags or system notes—just your message.")
    return "\n".join(lines)

def build_prompt_for_summarizer(branch: Branch) -> str:
    lines = ["[CONVERSATION]"]
    for m in branch.messages:
        lines.append(f"{m.sender}: {m.content.strip()}")
    lines.append(""); lines.append("[TASK] Provide a concise summary of the conversation so far.")
    return "\n".join(lines)
