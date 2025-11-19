# ui/styles.py
GLOBAL_CSS = """
<style>
.block-container { max-width: 1200px; padding-top: 0.6rem; }
.chat-wrap { display: flex; flex-direction: column; gap: 10px; margin: 0 auto; max-width: 100%; }

/* Base bubble */
.msg {
  position: relative;
  border-radius: 14px;
  padding: 10px 12px;
  margin: 2px 0;
  max-width: 100%;
  align-self: flex-start;
  background: #f5f5f5; /* default grey for agents */
  border-left: 6px solid transparent; /* side accent; set per color class */
}

.msg .who  { font-weight: 700; margin-bottom: 6px; }
.msg .body { line-height: 1.5; }
.msg .meta { font-size: 12px; opacity: 0.7; margin-top: 6px; }

/* Make markdown content behave nicely inside bubbles and final cards */
.md-body, .msg .body { overflow-x:auto; }
.md-body img, .msg .body img { max-width:100%; height:auto; }
.md-body table, .msg .body table { max-width:100%; width:100%; border-collapse: collapse; }
.md-body th, .md-body td, .msg .body th, .msg .body td { border-bottom: 1px solid #eee; padding: 4px 6px; }
.md-body pre, .msg .body pre { white-space: pre-wrap; word-wrap: break-word; }
.md-body code, .msg .body code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }

/* User bubble: green background, right-aligned, no accent stripe */
.msg.user {
  background: #d6f3d9;
  align-self: flex-end;
  text-align: right;
  border-left-color: transparent;
}

/* Summary bubble (kept subtle) */
.msg.summary {
  background: #f0f0f0;
  font-style: italic;
}

/* Agent bubble (grey bg, with left accent set by c0..c6 classes) */
.msg.agent { background: #f5f5f5; }

/* (Legacy) Remove old a/b/c accent classes if present */
.msg.agent.a, .msg.agent.b, .msg.agent.c { border-left-color: transparent; }
</style>
"""

FINAL_CARD_CSS = """
<style>
.final-card {
  border-left: 6px solid var(--accent, #f2a900);
  background: #faf7f0;
  padding: 14px 16px;
  border-radius: 12px;
  margin: 10px 0 14px 0;
  box-shadow: 0 1px 1px rgba(0,0,0,0.04);
}
.final-card .final-title { font-weight: 700; margin-bottom: 6px; color: #6b4e00; }
.final-card .final-meta { font-size: 12px; opacity: 0.8; margin-bottom: 10px; }
</style>
"""

BUBBLE_7_COLORS_CSS = """
<style>
/* 7 deterministic agent colors (c0..c6). Name color + left accent. */
.msg.agent.c0 .who  { color:#059669; } .msg.agent.c0 { border-left-color:#059669; } /* green-600 */
.msg.agent.c1 .who  { color:#D97706; } .msg.agent.c1 { border-left-color:#D97706; } /* amber-600 */
.msg.agent.c2 .who  { color:#DC2626; } .msg.agent.c2 { border-left-color:#DC2626; } /* red-600 */
.msg.agent.c3 .who  { color:#2563EB; } .msg.agent.c3 { border-left-color:#2563EB; } /* blue-600 */
.msg.agent.c4 .who  { color:#7C3AED; } .msg.agent.c4 { border-left-color:#7C3AED; } /* violet-600 */
.msg.agent.c5 .who  { color:#0D9488; } .msg.agent.c5 { border-left-color:#0D9488; } /* teal-600 */
.msg.agent.c6 .who  { color:#B45309; } .msg.agent.c6 { border-left-color:#B45309; } /* orange-600 */
</style>
"""

# Append once (avoid double-adding EXTRA/COLORS blocks)
GLOBAL_CSS = GLOBAL_CSS + BUBBLE_7_COLORS_CSS

EDIT_CSS = """
<style>
.msg-wrap { position: relative; }
.msg-wrap .edit-btn { 
  position: absolute; 
  top: -6px; 
  right: -6px; 
  opacity: 0; 
  transition: opacity .12s ease; 
}
.msg-wrap:hover .edit-btn { opacity: 1; }

/* Make the Streamlit button tiny/neutral */
.msg-wrap .edit-btn button[kind="secondary"] {
  padding: 2px 6px !important;
  border-radius: 8px !important;
  font-size: 12px !important;
  line-height: 1 !important;
}
</style>
"""
GLOBAL_CSS = GLOBAL_CSS + EDIT_CSS
