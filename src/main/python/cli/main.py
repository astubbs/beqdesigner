#!/usr/bin/env python3
"""BEQ Designer — Bass EQ correction profile designer.

Run with no arguments for an interactive menu, or use subcommands directly:

    bin/beq-designer                              # interactive menu
    bin/beq-designer profile "movie.mkv"          # generate profile
    bin/beq-designer extract --media-root /media  # extract audio cache
    bin/beq-designer cache-status                 # cache health
    bin/beq-designer --help                       # list all subcommands
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

import typer

from cli.common import (
    REPO_ROOT,
    CliConfig,
    console,
    filterable_select,
    load_config,
    save_config,
    setup_log_file,
    show_banner,
)

# ---------------------------------------------------------------------------
# Typer app — dual mode: interactive (no args) or subcommands
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="beq-designer",
    help="BEQ Designer — Bass EQ correction profile generation, cache management, and analysis.",
    invoke_without_command=True,
    no_args_is_help=False,
    rich_markup_mode="rich",
)

sweep_app = typer.Typer(help="Sweep operations — discover media, run pipeline, generate reports.")
app.add_typer(sweep_app, name="sweep")


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------

_MAIN_MENU = [
    ("Generate profile       Analyse a movie/episode and create BEQ correction filters", "profile"),
    ("Advanced               Analysis, cache management, and developer tools", "advanced"),
    ("Settings               Output directory, default paths", "config"),
    ("Quit", "quit"),
]

_ADVANCED_MENU = [
    ("ANALYSIS — these steps run automatically during profile generation;"
     " use them here to investigate output or re-run individually", None),
    ("NN accuracy report     Compare the model's predicted filters vs hand-tuned BEQ entries", "nn-report"),
    ("Discover media         Scan media library folders and match against the BEQ catalogue", "sweep-discover"),
    ("Test predictions       Run the model against all discovered media and measure accuracy", "sweep-run"),
    ("Experiment results     View and compare results from different test runs", "sweep-report"),

    ("CACHE — audio extraction happens automatically; use these to pre-populate"
     " or troubleshoot the cache", None),
    ("Pre-extract audio      Extract bass/LFE from media ahead of time to speed up analysis", "extract"),
    ("Check cache health     See how many files are cached and ready for analysis", "cache-status"),
    ("Repair cache           Scan for corrupt audio files and optionally re-extract them", "verify"),

    ("Back", "back"),
]


def _interactive_menu_loop(config: CliConfig, verbose: bool) -> None:
    """Show the main menu in a loop until the user quits."""
    while True:
        try:
            action = filterable_select("What would you like to do?", _MAIN_MENU)
        except KeyboardInterrupt:
            break

        if action is None or action == "quit":
            break

        if action == "advanced":
            _advanced_menu_loop(config, verbose)
            continue

        try:
            _dispatch(action, config, verbose)
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted — returning to menu.[/dim]")
        except SystemExit:
            pass  # typer.Exit() from subcommands

    console.print("[dim]Goodbye.[/dim]")


def _advanced_menu_loop(config: CliConfig, verbose: bool) -> None:
    """Show the advanced submenu until the user goes back."""
    while True:
        try:
            action = filterable_select("Advanced options:", _ADVANCED_MENU)
        except KeyboardInterrupt:
            break

        if action is None or action == "back":
            break

        try:
            _dispatch(action, config, verbose)
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted — returning to advanced menu.[/dim]")
        except SystemExit:
            pass


def _dispatch(action: str, config: CliConfig, verbose: bool) -> None:
    """Route a menu choice to the appropriate handler."""
    handlers = {
        "profile": _do_profile,
        "extract": _do_extract,
        "cache-status": _do_cache_status,
        "verify": _do_verify,
        "nn-report": _do_nn_report,
        "sweep-discover": _do_sweep_discover,
        "sweep-run": _do_sweep_run,
        "sweep-report": _do_sweep_report,
        "config": _do_config,
    }
    handler = handlers.get(action)
    if handler:
        handler(config, verbose)


# ---------------------------------------------------------------------------
# Handlers — interactive prompts then delegate to existing scripts
# ---------------------------------------------------------------------------


def _do_profile(config: CliConfig, verbose: bool) -> None:
    """Generate BEQ profile — delegates to beq_profile_cli."""
    from cli.profile import generate
    generate(media=None, author=None, output=None, output_dir=None, verbose=verbose)


def _do_extract(config: CliConfig, verbose: bool) -> None:
    """Extract LFE cache — delegates to extract_lfe."""
    import questionary
    from spike._auto_beq_helpers import audio_cache_dir

    try:
        default_cache = str(audio_cache_dir())
    except RuntimeError:
        default_cache = ""

    media_roots = []
    console.print("[bold]Add media library roots[/bold] (press Enter with empty path to finish):")
    while True:
        root = questionary.path("Media root (empty to finish):", default="").ask()
        if not root:
            break
        p = Path(root).resolve()
        if p.is_dir():
            media_roots.append(str(p))
        else:
            console.print(f"[red]Not a directory: {p}[/red]")

    if not media_roots:
        console.print("[yellow]No media roots provided.[/yellow]")
        return

    limit = questionary.text("Max titles to extract (0 = unlimited):", default="0").ask()

    argv = []
    for r in media_roots:
        argv.extend(["--media-root", r])
    if default_cache:
        argv.extend(["--beq-dir", default_cache])
    if limit and limit != "0":
        argv.extend(["--limit", limit])
    if verbose:
        argv.append("-v")

    from cli.extract import main as extract_main
    extract_main(argv)


def _do_cache_status(config: CliConfig, verbose: bool) -> None:
    """WAV cache status — delegates to wav_cache_status."""
    from cli.cache_status import main as status_main
    status_main()


def _do_verify(config: CliConfig, verbose: bool) -> None:
    """Verify cache integrity — delegates to verify_wav_cache."""
    import questionary
    from spike._auto_beq_helpers import wav_cache_dir

    try:
        default_cache = str(wav_cache_dir())
    except Exception:
        default_cache = ""

    cache_root = questionary.path("WAV cache root:", default=default_cache).ask()
    if not cache_root:
        return
    delete = questionary.confirm("Delete corrupt files?", default=False).ask()

    argv = [cache_root]
    if delete:
        argv.append("--delete")
    if verbose:
        argv.append("-v")

    from cli.verify_cache import main as verify_main
    verify_main(argv)


def _do_nn_report(config: CliConfig, verbose: bool) -> None:
    """NN comparison report — delegates to nn_comparison_report."""
    import questionary

    output = questionary.path("Save report to (empty for stdout):", default="").ask()
    argv = []
    if output:
        argv.extend(["--output", output])

    from cli.nn_report import main as nn_main
    nn_main(argv)


def _do_sweep_discover(config: CliConfig, verbose: bool) -> None:
    """Discover media — delegates to sweep_discover module."""
    from spike.sweep_discover import main as discover_main
    discover_main()


def _do_sweep_run(config: CliConfig, verbose: bool) -> None:
    """Run sweep pipeline — runs pytest in a subprocess."""
    import questionary

    limit = questionary.text("Max files to process:", default="10").ask()
    parallel = questionary.confirm("Run in parallel?", default=False).ask()

    test_module = "src/test/python/spike/test_auto_beq_library_sweep.py"
    test_func = "test_library_sweep_parallel" if parallel else "test_library_sweep"

    env = {
        **os.environ,
        "AUTO_BEQ_SWEEP_LIMIT": limit or "10",
        "AUTO_BEQ_ADVISOR": "measurement",
    }
    subprocess.run(
        ["poetry", "run", "pytest", f"{test_module}::{test_func}", "-v", "-s"],
        env=env, cwd=str(REPO_ROOT),
    )


def _do_sweep_report(config: CliConfig, verbose: bool) -> None:
    """Sweep report — delegates to sweep_report."""
    from cli.sweep_report import main as report_main
    report_main()


def _do_config(config: CliConfig, verbose: bool) -> None:
    """Configure preferences."""
    from cli.profile import _configure_preferences
    _configure_preferences(config)


# ---------------------------------------------------------------------------
# Typer subcommands — for CI / automation / scripting
# ---------------------------------------------------------------------------


@app.command()
def profile(
    media: Optional[Path] = typer.Argument(None, help="Media file or directory.", resolve_path=True),  # noqa: UP007
    author: Optional[str] = typer.Option(None, help="Author style for filter prediction."),  # noqa: UP007
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output JSON path.", resolve_path=True),  # noqa: UP007
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Output directory (batch).", resolve_path=True),  # noqa: UP007
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Debug logging."),
) -> None:
    """Generate BEQ correction profiles for media files."""
    from cli.profile import generate
    generate(media=media, author=author, output=output, output_dir=output_dir, verbose=verbose)


@app.command()
def extract(
    media_root: Optional[list[Path]] = typer.Option(None, "--media-root", help="Media library root(s)."),  # noqa: UP007
    beq_dir: Optional[Path] = typer.Option(None, "--beq-dir", help="BEQ working directory."),  # noqa: UP007
    limit: int = typer.Option(0, "--limit", help="Max titles (0 = unlimited)."),
    verify_only: bool = typer.Option(False, "--verify", help="Verify existing cache only."),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Extract LFE audio from media library to WAV cache."""
    argv: list[str] = []
    if media_root:
        for r in media_root:
            argv.extend(["--media-root", str(r)])
    if beq_dir:
        argv.extend(["--beq-dir", str(beq_dir)])
    if limit:
        argv.extend(["--limit", str(limit)])
    if verify_only:
        argv.append("--verify")
    if verbose:
        argv.append("-v")
    from cli.extract import main as extract_main
    extract_main(argv)


@app.command(name="cache-status")
def cache_status(
    cache_dir: Optional[Path] = typer.Argument(None, help="WAV cache directory."),  # noqa: UP007
) -> None:
    """Show WAV cache counts, titles, and author breakdown."""
    argv = [str(cache_dir)] if cache_dir else []
    from cli.cache_status import main as status_main
    status_main(argv)


@app.command()
def verify(
    cache_root: Optional[Path] = typer.Argument(None, help="WAV cache root directory."),  # noqa: UP007
    delete: bool = typer.Option(False, "--delete", help="Delete corrupt files."),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Verify WAV cache integrity and optionally fix corrupt files."""
    argv: list[str] = []
    if cache_root:
        argv.append(str(cache_root))
    if delete:
        argv.append("--delete")
    if verbose:
        argv.append("-v")
    from cli.verify_cache import main as verify_main
    verify_main(argv)


@app.command(name="nn-report")
def nn_report(
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Save report to file."),  # noqa: UP007
) -> None:
    """Compare NN-predicted vs hand-coded BEQ filters."""
    argv: list[str] = []
    if output:
        argv.extend(["--output", str(output)])
    from cli.nn_report import main as nn_main
    nn_main(argv)


@sweep_app.command(name="discover")
def sweep_discover(
    library: Optional[list[Path]] = typer.Option(None, "--library", help="Media library root(s)."),  # noqa: UP007
) -> None:
    """Scan media library and match against BEQ catalogue."""
    argv: list[str] = []
    if library:
        for lib in library:
            argv.extend(["--library", str(lib)])
    from spike.sweep_discover import main as discover_main
    discover_main(argv if argv else None)


@sweep_app.command(name="run")
def sweep_run(
    limit: int = typer.Option(10, "--limit", help="Max files to process."),
    parallel: bool = typer.Option(False, "--parallel", help="Run in parallel across Ollama hosts."),
) -> None:
    """Run auto-BEQ pipeline on discovered media."""
    test_module = "src/test/python/spike/test_auto_beq_library_sweep.py"
    test_func = "test_library_sweep_parallel" if parallel else "test_library_sweep"
    env = {
        **os.environ,
        "AUTO_BEQ_SWEEP_LIMIT": str(limit),
        "AUTO_BEQ_ADVISOR": "measurement",
    }
    subprocess.run(
        ["poetry", "run", "pytest", f"{test_module}::{test_func}", "-v", "-s"],
        env=env, cwd=str(REPO_ROOT),
    )


@sweep_app.command(name="report")
def sweep_report_cmd() -> None:
    """Generate experiment comparison report from sweep results."""
    from cli.sweep_report import main as report_main
    report_main()


@app.command()
def config() -> None:
    """Edit CLI preferences (output directory, media paths)."""
    cfg = load_config()
    from cli.profile import _configure_preferences
    _configure_preferences(cfg)


# ---------------------------------------------------------------------------
# Dev subcommands — replaces shell wrappers
# ---------------------------------------------------------------------------

dev_app = typer.Typer(help="Developer tools — tests, sweeps, advisor comparison.")
app.add_typer(dev_app, name="dev")


@dev_app.command(name="test")
def dev_test(
    file: Optional[str] = typer.Option(None, "--file", help="Specific test file or file::test_name."),  # noqa: UP007
    verbose: bool = typer.Option(False, "--verbose", help="Show test output (no capture)."),
    advisor: str = typer.Option("measurement", "--advisor", help="Advisor implementation to use."),
) -> None:
    """Run spike tests (unit + integration)."""
    test_selector = file or "src/test/python/spike/"
    cmd = ["poetry", "run", "pytest", test_selector, "-v"]
    if verbose:
        cmd.append("-s")
    env = {**os.environ, "AUTO_BEQ_ADVISOR": advisor}
    subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))


@dev_app.command(name="sweep")
def dev_sweep(
    limit: int = typer.Option(10, "--limit", help="Max files to process."),
    parallel: bool = typer.Option(False, "--parallel", help="Run in parallel across Ollama hosts."),
    advisor: str = typer.Option("measurement", "--advisor", help="Advisor implementation."),
) -> None:
    """Run auto-BEQ sweep pipeline on discovered media."""
    test_module = "src/test/python/spike/test_auto_beq_library_sweep.py"
    test_func = "test_library_sweep_parallel" if parallel else "test_library_sweep"
    env = {
        **os.environ,
        "AUTO_BEQ_SWEEP_LIMIT": str(limit),
        "AUTO_BEQ_ADVISOR": advisor,
    }
    subprocess.run(
        ["poetry", "run", "pytest", f"{test_module}::{test_func}", "-v", "-s"],
        env=env, cwd=str(REPO_ROOT),
    )


@dev_app.command(name="compare-advisors")
def dev_compare_advisors(
    limit: int = typer.Option(0, "--limit", help="Max titles per advisor (0 = all)."),
) -> None:
    """Compare all advisor implementations side-by-side on sweep corpus."""
    advisors = ["measurement", "topology", "slope_extension"]
    for adv in advisors:
        console.rule(f"[bold]Advisor: {adv}[/bold]")
        test_module = "src/test/python/spike/test_auto_beq_library_sweep.py"
        env = {
            **os.environ,
            "AUTO_BEQ_ADVISOR": adv,
            "AUTO_BEQ_SWEEP_LIMIT": str(limit) if limit else "",
        }
        subprocess.run(
            ["poetry", "run", "pytest", f"{test_module}::test_library_sweep", "-v", "-s"],
            env=env, cwd=str(REPO_ROOT),
        )


@dev_app.command(name="playground")
def dev_playground(
    title: str = typer.Option(..., "--title", help="Title in catalogue snapshot."),
    filter_count: Optional[int] = typer.Option(None, "--filter-count", help="Disambiguate entries."),  # noqa: UP007
    plot: bool = typer.Option(False, "--plot", help="Show matplotlib plot."),
) -> None:
    """Test filter proposals on a single catalogue title (spike playground)."""
    from cli.spike_playground import main as spike_main
    argv = ["--entry-title", title]
    if filter_count is not None:
        argv.extend(["--filter-count", str(filter_count)])
    if plot:
        argv.append("--plot")
    spike_main(argv)


# ---------------------------------------------------------------------------
# Main callback — interactive menu when no subcommand given
# ---------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Debug logging."),
) -> None:
    """BEQ Designer — run with no subcommand for interactive menus."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        handlers=[logging.NullHandler()],
    )

    cfg = load_config()
    log_file = setup_log_file(cfg.output_dir)
    plain_banner = show_banner("BEQ Designer CLI", cfg, log_file=log_file)
    logging.getLogger("beq_cli").info(plain_banner)

    if ctx.invoked_subcommand is None:
        _interactive_menu_loop(cfg, verbose)


def main():
    app()


if __name__ == "__main__":
    main()
