# This file is based on code from https://github.com/B4PT0R/streamlit_notebook
# Original license: MIT
# Modified by https://github.com/mlloliveira 

import random
import string
import streamlit as st
from code_editor import code_editor #https://github.com/bouzidanas/streamlit-code-editor

# Simple alias so we mirror a "notebook-style" state helper
state = st.session_state


def _short_id(length: int = 16) -> str:
    """Generate a short random id (used for editor keys)."""
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choices(alphabet, k=length))


class editor_output_parser:
    """
    Minimal parser for the streamlit-code-editor output.

    It deduplicates events based on the returned `id` field and always returns
    a pair (event, text_content).
    """

    def __init__(self, initial_code: str = ""):
        self.last_id = None
        self.last_code = initial_code

    def __call__(self, output):
        if output is None:
            # No new interaction: keep last contents, no event
            return None, self.last_code

        # New code text from the editor
        self.last_code = output.get("text", self.last_code)

        event = None
        out_id = output.get("id")
        out_type = output.get("type", "")

        # Only emit an event when the id changes (new click / keypress)
        if out_id is not None and out_id != self.last_id:
            self.last_id = out_id
            if out_type:
                event = out_type

        return event, self.last_code


class Code:
    """
    Tiny helper that mirrors the original Code object from cell_ui.py.

    It wraps a string and lets the UI update it without clobbering
    backend-set values in the same run.
    """

    def __init__(self, value: str = ""):
        self._value = value
        self.new_code_flag = False

    def get_value(self) -> str:
        return self._value

    def from_ui(self, value: str) -> None:
        # If backend just set a new value, ignore the first UI echo of it
        if self.new_code_flag:
            self.new_code_flag = False
        else:
            self._value = value

    def from_backend(self, value: str) -> None:
        self._value = value
        self.new_code_flag = True


class Editor:
    """
    Very small wrapper around streamlit-code-editor.

    Features:
    - Line numbers
    - Configurable visible lines before scrolling (min_lines / max_lines)
    - A single "Run" button (bottom-right) that fires event "run"
    - Ctrl+Enter emits event "submit" (handled by the underlying component)

    After calling .show(), check `editor.event` for "run" or "submit".
    The current text is always available via `editor.code.get_value()`.
    """

    def __init__(
        self,
        code: Code | None = None,
        key: str | None = None,
        lang: str = "text",
        min_lines: int = 15,
        max_lines: int = 15,
    ):
        self.code: Code = code or Code()
        self.key: str = key or _short_id()
        self.lang: str = lang
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.event: str | None = None
        self._parser = editor_output_parser(self.code.get_value())

    # --- internal helpers -------------------------------------------------

    def _buttons(self):
        """Single Run button, styled similarly to the original notebook."""
        return [
            {
                "name": "Run",
                "feather": "Play",
                "iconSize": "20px",
                "primary": True,
                "hasText": False,
                "alwaysOn": True,
                "showWithIcon": True,
                "commands": [["response", "run"]],
                "style": {
                    "bottom": "0px",
                    "right": "0px",
                    "fontSize": "14px",
                },
            }
        ]

    def get_params(self) -> dict:
        """Build kwargs passed to code_editor()."""
        params = dict(
            lang=self.lang,
            key=self.key,
            buttons=self._buttons(),
            options={
                "showLineNumbers": True,
            },
            props={
                "enableBasicAutocompletion": False,
                "enableLiveAutocompletion": False,
                "enableSnippets": False,
                "style": {
                    "borderRadius": "0px 0px 0px 0px",
                },
            },
            # Same min/max so the editor has fixed height and scrolls after that
            height=[self.min_lines, self.max_lines],
        )
        return params
    
    def get_output(self, output):
        """
        Retrieve the latest editor output, preferring the component's
        stored state in st.session_state when present.

        This mirrors the original project's behavior and prevents
        losing the editor contents when the app reruns or when
        switching modes.
        """
        if self.key in state:
            return state[self.key]
        else:
            return output

    # --- public API -------------------------------------------------------

    def show(self) -> None:
        """
        Render the editor and process any new event.

        After this call:
        - self.code.get_value() holds the current text
        - self.event is either "run", "submit" or None
        """
        raw_output = code_editor(self.code.get_value(), **self.get_params())
        output = self.get_output(raw_output)
        event, content = self._parser(output)
        self.code.from_ui(content)
        self.event = event


__all__ = ["state", "Code", "Editor"]
