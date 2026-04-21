#!/usr/bin/env python3
"""Interactive CLI for generating BEQ bass-correction profiles.

Run via the unified CLI:

    bin/beq-designer profile                      # interactive menus
    bin/beq-designer profile "Avatar (2009).mkv"  # single file shortcut
    bin/beq-designer profile ./media-dir/         # batch shortcut
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from InquirerPy import inquirer
import typer
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from cli.common import (
    CliConfig,
    MEDIA_EXTENSIONS,
    console,
    fuzzy_select,
    load_config,
    save_config,
    setup_log_file,
    show_banner,
)
from cli.generate import generate_profile


# ---------------------------------------------------------------------------
# Model check
# ---------------------------------------------------------------------------


# Shared implementation — no duplication.
from model.wav_discovery import check_production_model as _check_production_model


# ---------------------------------------------------------------------------
# Interactive prompts
# ---------------------------------------------------------------------------


def _prompt_mode() -> str:
    """Ask what the user wants to do."""
    return inquirer.select(
        message="What would you like to do?",
        choices=[
            {"name": "Generate profile for a single file", "value": "single"},
            {"name": "Generate profiles for a directory (batch)", "value": "batch"},
            {"name": "Configure preferences", "value": "config"},
        ],
    ).execute()


def _discover_media(directory: Path, recursive: bool = True) -> list[Path]:
    """Find media files in a directory, optionally recursing into subdirs.

    Uses os.scandir for speed — avoids stat-ing every file when only the
    name (extension) is needed.
    """
    results: list[Path] = []
    try:
        with os.scandir(directory) as it:
            for entry in it:
                if entry.is_file(follow_symlinks=False) and Path(entry.name).suffix.lower() in MEDIA_EXTENSIONS:
                    results.append(Path(entry.path))
                elif recursive and entry.is_dir(follow_symlinks=False):
                    results.extend(_discover_media(Path(entry.path), recursive=True))
    except PermissionError:
        pass
    results.sort()
    return results


_LARGE_DIR_THRESHOLD = 30


def _prompt_directory(prompt_text: str, config: CliConfig) -> Path | None:
    """Prompt for a directory path with tab-completion.

    If the selected directory contains more than 30 subdirectories, asks for
    confirmation to avoid accidentally scanning a huge media library.
    """
    default = config.last_media_dir if config.last_media_dir else ""
    if default:
        console.print(f"[dim]Last used: {default}[/dim]")
    shown_tip = False
    while True:
        if not shown_tip:
            console.print("[dim]Tip: press Tab to autocomplete paths, type to filter[/dim]")
            shown_tip = True
        path_str = inquirer.filepath(
            message=prompt_text,
            default=default,
            only_directories=True,
        ).execute()
        if not path_str:
            return None
        selected = Path(path_str).resolve()
        subdirs = [d for d in selected.iterdir() if d.is_dir()] if selected.is_dir() else []
        if len(subdirs) > _LARGE_DIR_THRESHOLD:
            proceed = inquirer.confirm(
                message=f"This directory contains {len(subdirs)} subdirectories. Are you sure?",
                default=False,
            ).execute()
            if not proceed:
                default = path_str
                continue
        return selected


def _prompt_media_file_in(media_dir: Path, config: CliConfig) -> Path | None:
    """Navigate into a directory to select a media file.

    Only looks one level deep at a time — shows immediate subdirectories
    and media files. If there are subdirectories but no media files at
    this level, presents the subdirectories to drill into. Keeps drilling
    until the user reaches a directory with media files.
    """
    current = media_dir

    shown_tip = False

    while True:
        # Quick non-recursive scan — only this directory level.
        direct_files = _discover_media(current, recursive=False)
        subdirs = sorted(d for d in current.iterdir() if d.is_dir())

        if direct_files:
            # We have media files — show them, plus any subdirectories that
            # themselves contain media (e.g. season dirs alongside a trailer).
            # Subdirs without media (e.g. .trickplay/, backdrops/) are hidden.
            items: list[tuple[str, tuple[str, Path]]] = []
            for d in subdirs:
                if _discover_media(d, recursive=False):
                    items.append((f"{d.name}/", ("dir", d)))
            for f in direct_files:
                items.append((f.name, ("file", f)))

            # Pre-select last used file.
            default_choice = None
            last = config.last_media_file
            if last:
                last_path = Path(last)
                if str(last_path.parent) == str(current):
                    default_choice = last_path.name

            if not shown_tip:
                console.print("[dim]Tip: type to filter, arrow keys to navigate, Esc to clear filter[/dim]")
                shown_tip = True
            result = fuzzy_select(
                f"Select file or folder ({current.name}):",
                choices=items,
                default=default_choice,
            )
            if result is None:
                return None

            kind, value = result
            if kind == "dir":
                current = value
                continue
            return value

        elif subdirs:
            # No media here — just show subdirectories instantly, no scanning.
            items = [(f"{d.name}/", d) for d in subdirs]

            # Pre-select subdir containing the last used file.
            default_choice = None
            last = config.last_media_file
            if last:
                for d in subdirs:
                    if last.startswith(str(d)):
                        default_choice = f"{d.name}/"
                        break

            if not shown_tip:
                console.print("[dim]Tip: type to filter, arrow keys to navigate, Esc to clear filter[/dim]")
                shown_tip = True
            selected = fuzzy_select(
                f"Select folder ({current.name}):",
                choices=items,
                default=default_choice,
            )
            if selected is None:
                return None
            current = selected

        else:
            exts = ", ".join(sorted(MEDIA_EXTENSIONS))
            console.print(f"[red]No media files ({exts}) or subdirectories in {current}[/red]")
            return None


def _prompt_file_select(files: list[Path], base_dir: Path, config: CliConfig) -> Path | None:
    """Show a select list of files, pre-selecting the last used one."""
    if not files:
        return None

    last = config.last_media_file
    default_val = None
    iq_choices = []
    for f in files:
        rel = str(f.relative_to(base_dir))
        iq_choices.append({"name": rel, "value": f})
        if last and str(f) == last:
            default_val = f

    selected = inquirer.fuzzy(
        message="Select media file:",
        choices=iq_choices,
        default=default_val,
        mandatory=False,
    ).execute()
    return selected


def _prompt_media_file(config: CliConfig) -> Path | None:
    """Prompt for a directory, then select a media file from it."""
    media_dir = _prompt_directory("Media directory:", config)
    if media_dir is None:
        return None
    config.last_media_dir = str(media_dir)
    save_config(config)
    return _prompt_media_file_in(media_dir, config)


def _prompt_select_files(media_dir: Path, output_dir: Path) -> list[Path]:
    """Discover media files (recursively) and let the user select which to process.

    Handles flat directories and TV show structures (Show/Season N/episodes).
    """
    files = _discover_media(media_dir)
    if not files:
        exts = ", ".join(sorted(MEDIA_EXTENSIONS))
        console.print(f"[red]No media files ({exts}) found in {media_dir}[/red]")
        return []

    # Pre-select files that don't already have a profile in the output dir.
    iq_choices = []
    for f in files:
        profile_path = Path(output_dir) / f"{f.stem}_beq.json"
        already_done = profile_path.exists()
        label = str(f.relative_to(media_dir))
        if already_done:
            label += "  (profile exists)"
        iq_choices.append({"name": label, "value": f, "enabled": not already_done})

    selected = inquirer.checkbox(
        message=f"Found {len(files)} media files — select which to process:",
        choices=iq_choices,
    ).execute()
    return selected or []


def _prompt_author(config: CliConfig) -> str:
    """Prompt for author style, offering saved default."""
    return inquirer.text(
        message="Author style:",
        default=config.author,
    ).execute() or config.author


def _prompt_output_dir(config: CliConfig) -> str:
    """Prompt for output directory."""
    return inquirer.filepath(
        message="Output directory:",
        default=config.output_dir,
        only_directories=True,
    ).execute() or config.output_dir


def _configure_preferences(config: CliConfig) -> CliConfig:
    """Interactive preferences editor."""
    console.print("\n[bold]Current preferences:[/bold]")
    console.print(f"  Output dir:    {config.output_dir}")
    console.print(f"  Verbose logs:  {'on' if config.verbose else 'off'}")
    console.print()

    config.output_dir = _prompt_output_dir(config)
    config.verbose = inquirer.confirm(
        message="Show verbose logs during profile generation?",
        default=config.verbose,
    ).execute()
    save_config(config)
    console.print("[green]Preferences saved.[/green]\n")
    return config


# ---------------------------------------------------------------------------
# Progress tracking via log interception
# ---------------------------------------------------------------------------

_STAGE_TRIGGERS: list[tuple[str, str, int]] = [
    # (log trigger, display label, weight)
    # Audio extraction — one of these fires (30%)
    ("extracting LFE", "Extracting audio", 30),
    ("cached LFE WAV found", "Audio cached", 30),
    # Spectrum analysis (10%)
    ("WAV:", "Analysing spectrum", 10),
    # Model loading — one of these fires (30%)
    ("loaded E85 torch", "Loading model", 30),
    ("loaded production model", "Loading model", 30),
    ("training inline fallback", "Training model (slow — no production model)", 30),
    # Prediction + output (30%)
    ("predicting filters", "Predicting filters", 10),
    ("spectrograph saved", "Generating plots", 10),
    ("profile saved", "Saving profile", 10),
]


_MAX_LOG_LINES = 12  # how many log lines to keep in the verbose panel


class ProgressLoggingHandler(logging.Handler):
    """Intercept generate_profile log messages to drive a rich Progress bar.

    In verbose mode, also collects recent log lines for the live log panel.
    """

    def __init__(self, progress: Progress, task_id, *, verbose: bool = False, level: int = logging.NOTSET):
        super().__init__(level)
        self._progress = progress
        self._task_id = task_id
        self._seen: set[str] = set()
        self._verbose = verbose
        self.log_lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()

        if self._verbose:
            self.log_lines.append(msg)
            if len(self.log_lines) > _MAX_LOG_LINES:
                self.log_lines.pop(0)

        for trigger, label, weight in _STAGE_TRIGGERS:
            if trigger in msg and trigger not in self._seen:
                self._seen.add(trigger)
                self._progress.update(self._task_id, description=label, advance=weight)
                break


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------


def _render_results(profile: dict, output_path: Path) -> None:
    """Print a markdown-formatted summary of the generated profile."""
    from rich.markdown import Markdown

    filters = profile.get("filters", [])
    title = profile.get("title", "Unknown")
    year = profile.get("year", "")
    audio = profile.get("audioTypes", [""])[0] if profile.get("audioTypes") else ""
    mv = profile.get("mv", "0")
    note = profile.get("note", "")

    # Build markdown summary.
    lines = [
        f"## {title} ({year})",
        "",
        f"**Audio:** {audio}  ",
        f"**Master volume adjust:** {mv} dB  ",
        f"**Model:** {note}  ",
        "",
        "### Predicted Filters",
        "",
        "| Type | Freq (Hz) | Gain (dB) | Q |",
        "|------|-----------|-----------|---|",
    ]
    for f in filters:
        lines.append(
            f"| {f.get('type', '?')} | {f.get('freq', 0):.0f} | {f.get('gain', 0):+.1f} | {f.get('q', 0):.2f} |"
        )

    lines.append("")

    # Output paths.
    abs_path = output_path.resolve()
    lines.append("### Output")
    lines.append("")
    lines.append(f"- **Profile:** `{output_path}`")

    img_dir = output_path.parent
    spect = list(img_dir.glob(f"*spectrograph*.png"))
    if spect:
        lines.append(f"- **Spectrograph:** `{spect[0].name}`")

    md_text = "\n".join(lines)

    console.print()
    console.print(Markdown(md_text))

    # Clickable links (separate from markdown - Rich markdown doesn't support file:// links).
    # Escape path text because media filenames often contain [brackets]
    # (e.g. [Bluray-2160p][FLAC 2.0]) which Rich interprets as markup tags.
    from rich.text import Text
    console.print()
    link_text = Text(f"  {output_path}")
    link_text.stylize(f"link file://{abs_path}")
    console.print(link_text)
    if spect:
        spect_abs = spect[0].resolve()
        spect_text = Text(f"  {spect[0].name}")
        spect_text.stylize(f"link file://{spect_abs}")
        console.print(spect_text)


# ---------------------------------------------------------------------------
# Pipeline wrappers
# ---------------------------------------------------------------------------


def _run_single(
    media_path: Path, author: str, output_path: Path, output_dir: Path, *, verbose: bool = False,
) -> dict | None:
    """Run generate_profile for a single file with rich progress display."""
    console.print(f"\n[bold]Processing:[/bold] {media_path.name}")

    # Attach progress handler to both loggers (pipeline + shared helpers).
    loggers = [logging.getLogger("generate_beq_profile"), logging.getLogger("auto_beq_spike")]

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )
    task = progress.add_task("Starting...", total=100)
    handler = ProgressLoggingHandler(progress, task, verbose=verbose)
    for lgr in loggers:
        lgr.addHandler(handler)

    if verbose:
        from rich.console import Group
        from rich.live import Live
        from rich.panel import Panel
        from rich.text import Text

        def _make_display():
            log_text = Text("\n".join(handler.log_lines) or "Waiting for output...")
            log_panel = Panel(log_text, title="Log", border_style="dim")
            return Group(progress, log_panel)

        try:
            with Live(_make_display(), console=console, refresh_per_second=4, transient=False) as live:
                live.update(_make_display())
                # Run pipeline in a thread so Live can refresh.
                import threading
                result_holder: list = []
                error_holder: list = []

                def _run():
                    try:
                        result_holder.append(generate_profile(
                            media_path, author=author, output_path=output_path, output_dir=output_dir,
                        ))
                    except Exception as exc:
                        error_holder.append(exc)

                t = threading.Thread(target=_run, daemon=True)
                t.start()
                while t.is_alive():
                    live.update(_make_display())
                    t.join(timeout=0.25)
                live.update(_make_display())

            if error_holder:
                raise error_holder[0]
            profile = result_holder[0]
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted.[/dim]")
            raise
        except Exception as exc:
            console.print(f"\n[red]Error:[/red] {exc}")
            console.print_exception(show_locals=False)
            return None
        finally:
            for lgr in loggers:
                lgr.removeHandler(handler)
    else:
        try:
            with progress:
                profile = generate_profile(
                    media_path,
                    author=author,
                    output_path=output_path,
                    output_dir=output_dir,
                )
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted.[/dim]")
            raise
        except Exception as exc:
            console.print(f"\n[red]Error:[/red] {exc}")
            console.print_exception(show_locals=False)
            return None
        finally:
            for lgr in loggers:
                lgr.removeHandler(handler)

    _render_results(profile, output_path)
    return profile


def _run_batch(
    files: list[Path],
    author: str,
    output_dir: Path,
    *,
    verbose: bool = False,
) -> None:
    """Process multiple media files with per-file progress."""
    console.print(f"\n[bold]Batch processing {len(files)} files[/bold]\n")

    succeeded = 0
    failed = 0
    for i, media_path in enumerate(files, 1):
        console.rule(f"[bold][{i}/{len(files)}] {media_path.name}[/bold]")
        out_path = output_dir / f"{media_path.stem}_beq.json"
        result = _run_single(media_path, author, out_path, output_dir, verbose=verbose)
        if result:
            succeeded += 1
        else:
            failed += 1

    console.print()
    console.rule("[bold]Batch complete[/bold]")
    console.print(f"  [green]Succeeded:[/green] {succeeded}")
    if failed:
        console.print(f"  [red]Failed:[/red]    {failed}")


# ---------------------------------------------------------------------------
# Typer entry point
# ---------------------------------------------------------------------------


def generate(
    media: Optional[Path] = typer.Argument(  # noqa: UP007 — typer needs Optional
        None,
        help="Media file or directory. If omitted, interactive menus are shown.",
        resolve_path=True,
    ),
    author: Optional[str] = typer.Option(  # noqa: UP007
        None,
        help="Author style for filter prediction. Overrides saved config.",
    ),
    output: Optional[Path] = typer.Option(  # noqa: UP007
        None, "-o", "--output",
        help="Output JSON path (single file mode).",
        resolve_path=True,
    ),
    output_dir: Optional[Path] = typer.Option(  # noqa: UP007
        None, "--output-dir",
        help="Output directory (batch mode).",
        resolve_path=True,
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable debug logging."),
) -> None:
    """Generate BEQ profiles from media files.

    Run with no arguments for interactive menus, or pass a media path directly.
    """
    # Swallow default log output — the ProgressHandler drives the UI.
    # Log file is set up by main_callback; if called standalone, set it up here.
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, handlers=[logging.NullHandler()])

    config = load_config()

    # Banner (log file already set up by main_callback if running via bin/beq-designer).
    plain_banner = show_banner("BEQ Profile Generator", config)
    logging.getLogger("beq_profile_cli").info(plain_banner)

    # Warn if no production model — this will be slow and inaccurate.
    _has_model = _check_production_model()
    if not _has_model:
        from rich.panel import Panel
        console.print(Panel(
            "[yellow bold]No production model found.[/yellow bold]\n\n"
            "Profile generation will fall back to inline training, which is\n"
            "slower and less accurate than the production model.\n\n"
            "[bold]To fix:[/bold] Go to [cyan]Tools > Train model[/cyan] from the main menu,\n"
            "or run: [cyan]bin/beq-designer dev train[/cyan]\n\n"
            "[dim]If you downloaded a release build, the model should have been\n"
            "included. Check that your WAV cache is populated first.[/dim]",
            title="Missing Model",
            border_style="yellow",
        ))

    # Verbose: CLI flag -v OR saved preference.
    effective_verbose = verbose or config.verbose
    effective_author = author or "auto"

    # --- Interactive mode (no positional arg) ---
    if media is None:
        mode = _prompt_mode()
        if mode is None:
            raise typer.Exit()

        if mode == "config":
            _configure_preferences(config)
            raise typer.Exit()

        if mode == "single":
            media = _prompt_media_file(config)
            if media is None:
                raise typer.Exit()
            effective_output_dir = Path(output_dir or config.output_dir)
            effective_output_dir.mkdir(parents=True, exist_ok=True)
            effective_output = output or (effective_output_dir / f"{media.stem}_beq.json")
            config.last_media_file = str(media)
            save_config(config)
            _run_single(media, effective_author, effective_output, effective_output_dir, verbose=effective_verbose)

        elif mode == "batch":
            media_dir = _prompt_directory("Media directory:", config)
            if media_dir is None:
                raise typer.Exit()
            effective_output_dir = Path(output_dir or config.output_dir)
            effective_output_dir.mkdir(parents=True, exist_ok=True)
            files = _prompt_select_files(media_dir, effective_output_dir)
            if not files:
                raise typer.Exit()
            config.last_media_dir = str(media_dir)
            save_config(config)
            _run_batch(files, effective_author, effective_output_dir, verbose=effective_verbose)

        raise typer.Exit()

    # --- Shortcut mode (positional arg provided) ---
    if media.is_dir():
        # Directory supplied — pre-fill directory prompt, then file selection.
        config.last_media_dir = str(media)
        save_config(config)
        selected = _prompt_media_file(config)
        if selected is None:
            raise typer.Exit()
        effective_output_dir = Path(output_dir or config.output_dir)
        effective_output_dir.mkdir(parents=True, exist_ok=True)
        effective_output = output or (effective_output_dir / f"{selected.stem}_beq.json")
        config.last_media_file = str(selected)
        save_config(config)
        _run_single(selected, effective_author, effective_output, effective_output_dir, verbose=effective_verbose)

    elif media.is_file():
        # Single file mode.
        effective_output_dir = Path(output_dir or config.output_dir)
        effective_output_dir.mkdir(parents=True, exist_ok=True)
        effective_output = output or (effective_output_dir / f"{media.stem}_beq.json")
        config.last_media_dir = str(media.parent)
        config.last_media_file = str(media)
        save_config(config)
        _run_single(media, effective_author, effective_output, effective_output_dir, verbose=effective_verbose)

    else:
        console.print(f"[red]Path does not exist:[/red] {media}")
        raise typer.Exit(1)


def main():
    typer.run(generate)


if __name__ == "__main__":
    main()
