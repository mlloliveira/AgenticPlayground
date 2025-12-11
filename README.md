# Agentic LLM Playground

An experimental, **transparent** and **modular** agentic LLM playground built on top of [Streamlit](https://streamlit.io) and [Ollama](https://ollama.com).

The goal of this project is not to hide the magic, but to **expose how LLM agents think, plan and talk to each other.** It is designed as a sandbox for experimenting with different agent setups, prompts and modes.

> ⚠️ This project is a work in progress. The core structure is ~80% there; features and modes will evolve over time.

---

## Core ideas

- **Transparency first**  
  - Show system prompts, full prepared prompts, tool usage, web searches and even *token-level* output.
  - Expose internal reasoning traces (e.g. `<think>...</think>`) for models that support reasoning.
  - Make it easy to inspect *why* an agent replied the way it did.
  - For vision-capable models, show when and how images are sent, and provide a conceptual ViT-style view of image patching.

- **Modular by design**  
  - Agents, modes and UI blocks are all defined in config and core modules.
  - New modes (e.g. judge, storytelling/narrator, RAG experiments) can (and will) be added.
  - The sidebar layout is user-configurable via a settings page.

- **Playground, not product**  
  - You are encouraged to fork, tweak agents, play with prompts, and create your own experimental setups.
  - The UI is intentionally “power-user friendly” rather than ultra-minimal.
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

- **Notebook**  
  A mini, Streamlit-based notebook mode with:
  - a Python code cell (REPL-style, with output),
  - a markdown notepad,
  - a safe execution sandbox.
  - a basic multi-agent, multi-round chat around the notebook, so agents can discuss, plan and execute changes.

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

### Vision / image input (vLLM support)


- **Per-turn image control**
  - Use the sidebar to attach an image to the next user message.
  - The attached image is reused for later turns until you clear it.


- **Model-agnostic & transparent**
  - You can use any Ollama model; vision behavior depends on the model itself.
  - Conceptual ViT-style patch view

### Tools

Tools are the main way agents interact with anything beyond pure text. Only works with "thinking" models.

- **Calculator tool (`calc`)**
  - Safe, structured calculator for numeric work.

- **Dice / RNG tool (`dice`)**
  - Roll virtual dice (e.g. for games, simulations, random choices).

- **Read tool (`workspace_read`)**
  - Lets agents read files such as snippets or notes.

- **Write tool (`workspace_write`)**
  - Let agents write/update files such snippets or notes.

- **Notebook runner (`py_repl`)**
  - Let agents executes the current notebook python files inside a restricted environment (no imports, limited builtins, step limit).

- **Conversation search (`conv_search`)**
  - Lets agents search within the current branch for earlier messages containing a query string.

### Transparency 

- **Context inspector**
  - For each agent message, you can open a **Context** panel that shows:
    - System prompt
    - Prepared prompt body
    - Generation parameters
    - Web status & parsed queries
    - Planner information (if used)
    - The thinking trace (for reasoning models)
    - The denorm output tokens
    - Images passed to the model on that turn 

- **Conceptual ViT-style patch view**
  - When a message uses image input, the Context panel shows a `Vision (ViT-style patches — conceptual)` section.
  - The image is downsampled (by default to 224×224) and overlaid with a 16×16 patch grid (14×14 ≈ 196 patches).
  - This is an approximation of how ViT-like models tokenize images into patches / “image tokens”, meant to build intuition rather than reflect exact model internals.

- **Thinking trace**
  - If enabled, and the model / backend supports it, the internal reasoning is captured as `msg.thinking`.
  - Shown in the Context panel as a **Thinking** section.
  - When a reasoning model returns only thinking and no explicit answer, the playground prepends a warning to make this explicit.

- **Tools**
  - If a tool is used, it shows which tool and how it was used (tool name, arguments, and result).

- **Denormalized output tokens**
  - When enabled, new assistant messages store the **denormalized** output tokens returned by Ollama’s `logprobs` API.
  - These are shown under Context as:
    ```text
    ‹I› ‹ hope› ‹ this› ‹ explanation› ‹ is› ‹ clear› ‹.›
    ```
  - Tokens are model-family agnostic and human-readable (spaces and newlines are preserved).
  - Requires **Ollama ≥ 0.12.11** (logprobs support).

### Sidebar & All Settings

- **Mode selector + All Settings**
  - A persistent mode bar at the top of the sidebar lets you switch between **Chat**, **Consul**, and **All Settings**.
  - **All Settings** is a dedicated page where you can:
    - Reorder sidebar sections (Models, Web, Presets, Global Params, Agents, UI & Tools, Consul, Save & Reset, Usage, …).
    - Choose which sections are visible/hidden in the normal sidebar.
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
- **[Pillow](https://pypi.org/project/pillow/)** – image loading and conceptual ViT-style grid previews for vision models
- **[Streamlit code editor](https://github.com/bouzidanas/streamlit-code-editor)** – a code editor component for streamlit.io apps

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
streamlit_code_editor
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
  agents.py             # Default agent definitions (Chat & Consul)
  config.py             # AppConfig, AgentConfig, Branch, Msg, presets
  conversations.py      # Save/load encrypted conversation branches
  ollama_client.py      # Thin HTTP client for Ollama (generate, ps, unload)
  prompting.py          # Prompt construction & <think> parsing
  runtime_web.py        # Web planner / summarizer logic
  state.py              # Global state, sidebar, All Settings, branches, usage
  tools.py              # Tools such as math, dice, read and write.          
  vision.py             # Vision/vLLM helpers
  web_tools.py          # DuckDuckGo search + summarization helpers

views/
  chat.py               # Chat mode page
  consul.py             # Consul mode page
  notebook.py           # Notebook mode page
  notebook_imports.py   # Helpers for the notebook mode
  runtime.py            # Shared runtime helpers (run_agents_for_turn, etc.)
  settings.py           # All Settings page (sidebar layout, etc.)

ui/
  styles.py             # Global CSS and styling constants
  render.py             # Rendering of messages, bubbles, context, tokens
```


---

## Roadmap / ideas

Some directions this playground is planned (or suitable) for:

- **More tools**
  - File-scoped tools, plotting tools, simple database / vector store tools.
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
