# Agentic LLM Playground

An experimental, **transparent** and **modular** agentic LLM playground built on top of [Streamlit](https://streamlit.io) and [Ollama](https://ollama.com).

The goal of this project is not to hide the magic, but to **expose how LLM agents think, plan and talk to each other.**. It is designed as a sandbox for experimenting with different agent setups, prompts and modes.

> ⚠️ This project is a work in progress. The core structure is ~70% there; features and modes will evolve over time.

---

## Core ideas

- **Transparency first**  
  - Show system prompts, full prepared prompts, tool usage, web searches and even *token-level* output.
  - Expose internal reasoning traces (e.g. `<think>...</think>`) for models that support reasoning.
  - Make it easy to inspect *why* an agent replied the way it did.

- **Modular by design**  
  - Agents, modes and UI blocks are all defined in config and core modules.
  - New modes (e.g. judge, storytelling/narrator, RAG experiments) can (and will) be added.
  - The sidebar layout is user-configurable via a settings page.

- **Playground, not product**  
  - You are encouraged to fork, tweak agents, play with prompts, and create your own experimental setups.
  - The UI is intentionally “power‑user friendly” rather than ultra-minimal.
  - Streamlit provides the frontend so I can focus more on the backend.

---

## Current features

### Modes

- **Chat**  
  A standard chat with one or more agents, but with full visibility into system prompts, context and parameters.

- **Consul**  
  A “council of agents” mode where multiple agents debate / propose answers and a final response is formed.  
  Useful for experimenting with:
  - voting / consensus,
  - diverse prompts per agent,
  - different models in the same run.

> More modes (e.g. judge, storyteller/narrator, RAG) are planned but not yet implemented.

### Branching, editing & experimentation

- **Branching conversations**
  - Each conversation consists of branches (e.g. `main`, `Chat #1`, `Chat #2`).
  - You can fork at any point and explore different continuations from the same history.

- **Inline editing**
  - Edit user messages (and agent messages) in-place.
  - Editing a past message creates a new branch so you can compare outcomes.
  - Designed for “what if I’d asked it this way instead?” workflows.

- **Save & load conversations**
  - Save the current branch to an encrypted file (using `cryptography.Fernet`).
  - Load a saved branch as the new `main` conversation branch.

### Web & planner tools

- **Web search**
  - Uses the `ddgs` (DuckDuckGo Search) library for web search.
  - Results are summarized and can be injected into the context when agents have web enabled.

- **Web planner & summarizer (special agents)**
  - Planner and summarizer agents run in the background to structure searches and maintain a running summary.
  - These are intentionally the only “special” agents; everything else should be composable by users.

### Transparency tools

- **Context inspector**
  - For each agent message, you can open a **Context** panel that shows:
    - System prompt
    - Prepared prompt body
    - Generation parameters
    - Web status & parsed queries
    - Planner information (if used)
    - The thinking trace (for reasoning models)
    - The denorm output tokens

- **Thinking trace**
  - If enabled, and the model / backend supports it, the internal reasoning is captured as `msg.thinking`.
  - Shown in the Context panel as a **Thinking** section.

- **Denormalized output tokens**
  - When enabled, new assistant messages store the **denormalized** output tokens returned by Ollama’s `logprobs` API.
  - These are shown under Context as:
    ```
    ‹I› ‹ hope› ‹ this› ‹ explanation› ‹ is› ‹ clear› ‹.›
    ```
  - Tokens are model-family agnostic and human-readable (spaces and newlines are preserved).
  - Requires **Ollama ≥ 0.12.11** (logprobs support).

### Sidebar & All Settings

- **Mode selector + All Settings**
  - A persistent mode bar at the top of the sidebar lets you switch between **Chat**, **Consul**, and **All Settings**.
  - **All Settings** is a dedicated page where you can:
    - Reorder sidebar sections (Models, Web, Presets, Global Params, Agents, UI & Tools, Consul, Save & Reset, Usage, …).
    - Choose which sections are visible/hide in the normal sidebar.
    - Edit parameters even for sections that are hidden from the sidebar.

- **Usage & reset tools**
  - Shows token usage per session (prompt tokens, completion tokens, context size, etc.).
  - Buttons to:
    - Reset the entire chat (all branches)
    - Reset only the active branch
    - Save / load conversations

---

## Tech stack

- **Python** (3.10+ recommended)
- **[Streamlit](https://streamlit.io/)** – UI framework
- **[Ollama](https://ollama.com/)** – local model runner (HTTP API)
- **[requests](https://pypi.org/project/requests/)** – HTTP client
- **[ddgs](https://pypi.org/project/ddgs/)** – DuckDuckGo search
- **[markdown](https://pypi.org/project/Markdown/)** – Markdown rendering
- **[cryptography](https://pypi.org/project/cryptography/)** – encrypted conversation saves (Fernet)

---

## Installation

### 1. Prerequisites

- Python **3.10+**
- [Ollama](https://ollama.com/) installed and running (`ollama serve`)
- At least one model pulled (e.g. `gemma3`, `qwen3`, `gpt-oss`, etc.)

Optional env var:

```bash
export OLLAMA_HOST="http://localhost:11434"
```

If you don’t set it, the default is `http://localhost:11434`.

### 2. Clone & create a virtual env

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

### 3. Install dependencies

A minimal `requirements.txt` might look like:

```text
streamlit==1.51.0
requests==2.31.0
ddgs==9.7.1
markdown==3.10
cryptography==42.0.5
```

You can pin versions using the provided `version.py` helper (see below).

Install with:

```bash
pip install -r requirements.txt
```

### 4. Run the app

From the project root (where `app.py` lives):

```bash
streamlit run app.py
```

The UI should open in your browser (usually at `http://localhost:8501`).

---

## Project layout (high level)

The repository is roughly organized as:

```text
app.py                  # Streamlit entry point

core/
  config.py             # AppConfig, AgentConfig, Branch, Msg, presets
  state.py              # Global state, sidebar, All Settings, branches, usage
  ollama_client.py      # Thin HTTP client for Ollama (generate, ps, unload)
  agents.py             # Default agent definitions (Chat & Consul)
  web_tools.py          # DuckDuckGo search + summarization helpers
  runtime_web.py        # Web planner / summarizer logic
  prompting.py          # Prompt construction & <think> parsing
  conversations.py      # Save/load encrypted conversation branches

views/
  chat.py               # Chat mode page
  consul.py             # Consul mode page
  runtime.py            # Shared runtime helpers (run_agents_for_turn, etc.)
  settings.py           # All Settings page (sidebar layout, etc.)

ui/
  styles.py             # Global CSS and styling constants
  render.py             # Rendering of messages, bubbles, context, tokens
```

You can customize agents, prompts and behavior primarily via:

- `core/config.py` (agent definitions & parameters)
- `core/agents.py`
- `core/prompting.py`
- `core/state.py` (sidebar & layout)
- `app/views/*` (mode-specific UI)

---

## Roadmap / ideas

Some directions this playground is planned (or suitable) for:

- **Judge modes**
  - Independent judge agent that scores or critiques answers from other agents.
- **Storytelling / narrator mode**
  - Multi-agent collaborative story generation, with a Narrator agent controlling pacing.
- **RAG experiments**
  - Local embedding + retrieval stack plugged into agents as a tool.
- **More fine-grained transparency**
  - New ideas for transparency and things to play are always coming along.

Contributions, issues and discussions are very welcome – this project is meant to be played with.

---

## License

This project is licensed under the [MIT License](https://mit-license.org/).
