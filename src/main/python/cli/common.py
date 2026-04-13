"""Shared utilities for BEQ CLI scripts.

Extracted from beq_profile_cli.py so that beq.py (unified CLI) and
beq_profile_cli.py (standalone profile generator) share the same
interactive components, config persistence, and startup logic.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Venv and path setup — must run before any project imports
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[4]  # cli/ -> python/ -> main/ -> src/ -> repo root


def ensure_venv() -> None:
    """Re-exec under poetry if we're not already in the venv."""
    if not os.environ.get("VIRTUAL_ENV"):
        os.execvp("poetry", ["poetry", "run", "python", str(Path(sys.argv[0]).resolve()), *sys.argv[1:]])


def setup_python_path() -> None:
    """Add src/main/python and src/test/python to sys.path."""
    for _p in (REPO_ROOT, REPO_ROOT / "src" / "main" / "python", REPO_ROOT / "src" / "test" / "python"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))


# ---------------------------------------------------------------------------
# Rich console singleton
# ---------------------------------------------------------------------------

from rich.console import Console  # noqa: E402 — after path setup is defined

console = Console()

MEDIA_EXTENSIONS = {".mkv", ".iso", ".mp4", ".m2ts", ".ts", ".avi"}


# ---------------------------------------------------------------------------
# Filterable select — built on prompt_toolkit
# ---------------------------------------------------------------------------


def filterable_select(
    message: str,
    choices: list[tuple[str, object]],
    default: str | None = None,
) -> object | None:
    """Show a list that filters as the user types.

    Args:
        message: prompt text shown above the list
        choices: list of (label, value) pairs. If value is ``None``,
                 the entry is rendered as a non-selectable section header.
        default: label to pre-select

    Returns the value of the selected choice, or None if cancelled.
    """
    from prompt_toolkit import Application
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, HSplit, Window, FormattedTextControl

    # Separate selectable items from section headers.
    selectable = [(label, val) for label, val in choices if val is not None]

    filter_text = ""
    selected_idx = 0
    result: list[object | None] = [None]

    def _filtered() -> list[tuple[str, object]]:
        if not filter_text:
            return selectable
        q = filter_text.lower()
        return [(label, val) for label, val in selectable if q in label.lower()]

    def _get_display() -> FormattedText:
        items = _filtered()
        lines: list[tuple[str, str]] = []
        # Header.
        lines.append(("bold", f"? {message}"))
        if filter_text:
            lines.append(("", "  (filter: "))
            lines.append(("fg:yellow bold", filter_text))
            lines.append(("", ", Esc to clear)"))
        else:
            lines.append(("fg:ansigray", "  (type to filter, Esc to go back)"))
        lines.append(("", "\n"))
        if not items:
            lines.append(("fg:red", "  No matches.\n"))
            return FormattedText(lines)

        if filter_text:
            # When filtering, show flat list (no headers).
            for i, (label, _val) in enumerate(items):
                if i == selected_idx:
                    lines.append(("fg:cyan bold", f"  > {label}\n"))
                else:
                    lines.append(("", f"    {label}\n"))
        else:
            # Show full list with section headers.
            sel_idx = 0
            for label, val in choices:
                if val is None:
                    # Section header.
                    lines.append(("", "\n"))
                    lines.append(("fg:yellow bold", f"  {label}\n"))
                else:
                    if sel_idx == selected_idx:
                        lines.append(("fg:cyan bold", f"    > {label}\n"))
                    else:
                        lines.append(("", f"      {label}\n"))
                    sel_idx += 1
        return FormattedText(lines)

    kb = KeyBindings()

    @kb.add("up")
    def _up(event):
        nonlocal selected_idx
        if selected_idx > 0:
            selected_idx -= 1

    @kb.add("down")
    def _down(event):
        nonlocal selected_idx
        items = _filtered()
        if selected_idx < len(items) - 1:
            selected_idx += 1

    @kb.add("enter")
    def _enter(event):
        items = _filtered()
        if items:
            result[0] = items[selected_idx][1]
        event.app.exit()

    @kb.add("c-c")
    @kb.add("c-d")
    def _cancel(event):
        event.app.exit()

    @kb.add("backspace")
    def _backspace(event):
        nonlocal filter_text, selected_idx
        if filter_text:
            filter_text = filter_text[:-1]
            selected_idx = 0

    @kb.add("escape")
    def _escape(event):
        nonlocal filter_text, selected_idx
        if filter_text:
            # Clear filter first.
            filter_text = ""
            selected_idx = 0
        else:
            # No filter active — go back / cancel.
            event.app.exit()

    @kb.add("<any>")
    def _type(event):
        nonlocal filter_text, selected_idx
        char = event.data
        if char.isprintable() and len(char) == 1:
            filter_text += char
            selected_idx = 0

    # Set initial selection to default.
    if default:
        for i, (label, _) in enumerate(choices):
            if label == default:
                selected_idx = i
                break

    control = FormattedTextControl(_get_display)
    app: Application = Application(
        layout=Layout(HSplit([Window(control)])),
        key_bindings=kb,
        full_screen=False,
    )
    app.run()
    return result[0]


# ---------------------------------------------------------------------------
# Config — reads/writes the shared ~/.config/beqdesigner/settings.json
# ---------------------------------------------------------------------------


@dataclass
class CliConfig:
    author: str = "auto"
    output_dir: str = "profiles"
    last_media_dir: str = ""
    last_media_file: str = ""


def load_config() -> CliConfig:
    """Load CLI preferences from shared settings.json."""
    # Lazy import to avoid circular deps before setup_python_path().
    from spike._auto_beq_helpers import load_settings
    settings = load_settings()
    return CliConfig(
        author=settings.get("cli_author", "auto"),
        output_dir=settings.get("cli_output_dir", "profiles"),
        last_media_dir=settings.get("cli_last_media_dir", ""),
        last_media_file=settings.get("cli_last_media_file", ""),
    )


def save_config(config: CliConfig) -> None:
    """Persist CLI preferences to shared settings.json."""
    from spike._auto_beq_helpers import save_settings
    save_settings({
        "cli_author": config.author,
        "cli_output_dir": config.output_dir,
        "cli_last_media_dir": config.last_media_dir,
        "cli_last_media_file": config.last_media_file,
    })


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------


def get_version_info() -> tuple[str, str, str]:
    """Return (version, branch, commit) for the current repo."""
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        ).stdout.strip()
    except Exception:
        branch, commit = "unknown", "unknown"
    try:
        from importlib.metadata import version as _pkg_version
        version = _pkg_version("beqdesigner")
    except Exception:
        version = "dev"
    return version, branch, commit


def show_banner(title: str, config: CliConfig, log_file: Path | None = None) -> str:
    """Print a startup banner and return the plain-text version for logging.

    Returns the banner text with Rich markup stripped (for log files).
    """
    from rich.panel import Panel
    from spike._auto_beq_helpers import audio_cache_dir

    version, branch, commit = get_version_info()
    try:
        cache_dir = str(audio_cache_dir())
    except RuntimeError:
        cache_dir = "[red]NOT CONFIGURED[/red]"

    lines = [
        f"[bold]Version:[/bold]   {version} ({branch} @ {commit})",
        f"[bold]Output:[/bold]    {config.output_dir}",
        f"[bold]WAV cache:[/bold] {cache_dir}",
    ]
    if log_file:
        lines.append(f"[bold]Log file:[/bold]  {log_file}")

    banner = "\n".join(lines)
    console.print()
    console.print(Panel(banner, title=title, border_style="blue"))

    # Return plain text for log files.
    return re.sub(r"\[/?[^\]]+\]", "", banner)


def setup_log_file(output_dir: str, log_name: str = "beq.log") -> Path:
    """Create a log file handler and return the log file path."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / log_name
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-5s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(file_handler)
    return log_file
