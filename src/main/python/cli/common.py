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
# Filterable select — InquirerPy fuzzy prompt
# ---------------------------------------------------------------------------


def menu_select(
    message: str,
    choices: list[tuple[str, object]],
    default: str | None = None,
) -> object | None:
    """Show a menu with arrow-key selection and Rich-style formatting.

    Section headers (value=None) are rendered as non-selectable labels.
    Arrow keys navigate, Enter selects, Esc goes back.
    """
    from prompt_toolkit import Application
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, HSplit, Window, FormattedTextControl

    selectable = [(label, val) for label, val in choices if val is not None]
    selected_idx = 0
    result: list[object | None] = [None]

    # Find default index.
    if default:
        for i, (_, val) in enumerate(selectable):
            if val == default:
                selected_idx = i
                break

    def _get_display() -> FormattedText:
        lines: list[tuple[str, str]] = []
        lines.append(("bold", f"\n  {message}\n"))

        sel_idx = 0
        for label, val in choices:
            if val is None:
                # Section header.
                lines.append(("", "\n"))
                lines.append(("fg:yellow bold", f"  {label}\n"))
            else:
                parts = label.split("\n", 1)
                title = parts[0].strip()
                if sel_idx == selected_idx:
                    lines.append(("fg:cyan bold", f"  ❯ {title}\n"))
                else:
                    lines.append(("", f"    {title}\n"))
                if len(parts) > 1:
                    style = "fg:ansigray"
                    for line in parts[1].strip().split("\n"):
                        lines.append((style, f"      {line.strip()}\n"))
                sel_idx += 1

        lines.append(("", "\n"))
        lines.append(("fg:ansigray", "  ↑↓ navigate, Enter select, Esc back\n"))
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
        if selected_idx < len(selectable) - 1:
            selected_idx += 1

    @kb.add("enter")
    def _enter(event):
        result[0] = selectable[selected_idx][1]
        event.app.exit()

    @kb.add("escape")
    @kb.add("left")
    def _back(event):
        event.app.exit()

    @kb.add("c-c")
    @kb.add("c-d")
    def _cancel(event):
        event.app.exit()

    control = FormattedTextControl(_get_display)
    app: Application = Application(
        layout=Layout(HSplit([Window(control)])),
        key_bindings=kb,
        full_screen=False,
        erase_when_done=True,
    )
    app.run()
    return result[0]


def fuzzy_select(
    message: str,
    choices: list[tuple[str, object]],
    default: object | None = None,
) -> object | None:
    """Show a filterable list for browsing (directories, files). Type to search.

    Args:
        message: prompt text shown above the list
        choices: list of (label, value) pairs
        default: default value to pre-select

    Returns the value of the selected choice, or None if cancelled (Esc).
    """
    from InquirerPy import inquirer

    iq_choices = [{"name": label, "value": val} for label, val in choices]

    return inquirer.fuzzy(
        message=message,
        choices=iq_choices,
        default=default,
        mandatory=False,
        keybindings={"skip": [{"key": "escape"}]},
        long_instruction="(Esc to go back)",
    ).execute()


# ---------------------------------------------------------------------------
# Config — reads/writes the shared ~/.config/beqdesigner/settings.json
# ---------------------------------------------------------------------------


@dataclass
class CliConfig:
    output_dir: str = "profiles"
    verbose: bool = False
    last_media_dir: str = ""
    last_media_file: str = ""


def load_config() -> CliConfig:
    """Load CLI preferences from shared settings.json."""
    # Lazy import to avoid circular deps before setup_python_path().
    from spike._auto_beq_helpers import load_settings
    settings = load_settings()
    return CliConfig(
        output_dir=settings.get("cli_output_dir", "profiles"),
        verbose=settings.get("cli_verbose", False),
        last_media_dir=settings.get("cli_last_media_dir", ""),
        last_media_file=settings.get("cli_last_media_file", ""),
    )


def save_config(config: CliConfig) -> None:
    """Persist CLI preferences to shared settings.json."""
    from spike._auto_beq_helpers import save_settings
    save_settings({
        "cli_output_dir": config.output_dir,
        "cli_verbose": config.verbose,
        "cli_last_media_dir": config.last_media_dir,
        "cli_last_media_file": config.last_media_file,
    })


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------


def get_version_info() -> tuple[str, str, str]:
    """Return (version, branch, commit) for the current repo.

    Resolution order:
    1. build/version.json (written by run.sh before Docker build)
    2. Live git commands (when running outside Docker)
    """
    branch = ""
    commit = ""

    # 1. Check version.json (baked into Docker image by run.sh).
    version_file = REPO_ROOT / "build" / "version.json"
    if version_file.exists():
        try:
            import json as _json
            data = _json.loads(version_file.read_text())
            branch = data.get("branch", "")
            commit = data.get("commit", "")
        except Exception:
            pass

    # 2. Fall back to live git.
    if not branch or branch == "unknown":
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
    from spike._auto_beq_helpers import wav_cache_dir

    from spike._auto_beq_helpers import beq_shared_dir

    version, branch, commit = get_version_info()

    # Shared directory — required, everything derives from it.
    try:
        beq_shared_dir = beq_shared_dir()
        shared_str = str(beq_shared_dir)
    except Exception:
        beq_shared_dir = None
        shared_str = "[red]NOT CONFIGURED — set BEQ_SHARED_DIR env var[/red]"

    # WAV cache — derived from shared dir.
    try:
        cache_dir = str(wav_cache_dir())
    except RuntimeError as exc:
        logging.getLogger("beq_cli").warning("wav_cache_dir not configured: %s", exc)
        cache_dir = "[red]NOT CONFIGURED[/red]"

    # Detect which model will be used (without loading it).
    import os as _os
    advisor = _os.environ.get("AUTO_BEQ_ADVISOR", "measurement")
    if advisor == "torch_differentiable":
        model_info = "E85 differentiable-DSP (torch)"
    elif _os.environ.get("AUTO_BEQ_MODEL_PATH"):
        model_info = f"custom ({_os.environ['AUTO_BEQ_MODEL_PATH']})"
    elif beq_shared_dir:
        prod = beq_shared_dir / "production_model.joblib"
        if prod.exists():
            model_info = f"E82 production ({prod.name})"
        else:
            model_info = "[yellow]no production model — will train inline (slow)[/yellow]"
    else:
        model_info = "[yellow]unknown — shared dir not configured[/yellow]"

    out_abs = str(Path(config.output_dir).resolve())
    # Pad labels to align values (longest label is "Shared dir:" = 11 chars).
    lines = [
        f"[bold]Version:   [/bold] {version} ({branch} @ {commit})",
        f"[bold]Shared dir:[/bold] {shared_str}",
        f"[bold]WAV cache: [/bold] {cache_dir}",
        f"[bold]Model:     [/bold] {model_info}",
        f"[bold]Output:    [/bold] [link=file://{out_abs}]{config.output_dir}[/link]",
    ]
    if log_file:
        log_abs = str(log_file.resolve())
        lines.append(f"[bold]Log file:  [/bold] [link=file://{log_abs}]{log_file}[/link]")

    banner = "\n".join(lines)
    console.print()
    console.print(Panel(banner, title=title, border_style="blue"))

    # Return plain text for log files.
    return re.sub(r"\[/?[^\]]+\]", "", banner)


def setup_log_file(output_dir: str, log_name: str = "beq.log") -> Path:
    """Create a log file handler and return the log file path.

    Idempotent - if a FileHandler for the same path already exists on
    the root logger, it is not added again.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / log_name

    # Avoid duplicate file handlers for the same path.
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file.resolve()):
            return log_file

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-5s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(file_handler)
    return log_file
