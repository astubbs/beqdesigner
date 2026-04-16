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
import sys
from pathlib import Path
from typing import Optional

import typer

log = logging.getLogger("beq_cli")

from cli.common import (
    REPO_ROOT,
    CliConfig,
    console,
    menu_select,
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
    ("Tools                  Model training, analysis, cache management", "tools"),
    ("Settings               Output directory, default paths", "config"),
    ("Quit", "quit"),
]

_cached_model_status: str | None = None


def _model_status_line() -> str:
    """Quick status: is production model present. Cached after first check."""
    global _cached_model_status
    if _cached_model_status is not None:
        return _cached_model_status
    # Import from shared helpers — NOT cli.profile, which pulls in the
    # full Qt/GUI stack via generate → auto_beq → iir → xy → preferences.
    from spike._auto_beq_helpers import check_production_model
    _cached_model_status = (
        "✓ production model found" if check_production_model()
        else "✗ no model — train one first"
    )
    return _cached_model_status


def _build_tools_menu() -> list[tuple[str, object]]:
    """Build the Tools menu dynamically with live status."""
    model_status = _model_status_line()

    # Quick WAV cache check — just existence, no counting.
    try:
        from spike._auto_beq_helpers import wav_cache_dir
        cache = wav_cache_dir()
        cache_status = "✓ configured" if cache.exists() else "not found"
    except Exception:
        cache_status = "not configured"

    return [
        (f"STEP 1: EXTRACT AUDIO ({cache_status})", None),

        ("Pre-extract audio\n"
         "  Extract bass/LFE from media files ahead of time so profile\n"
         "  generation is instant. Best run on the machine closest to\n"
         "  your media drives (NAS, or via Docker).", "extract"),
        ("Check cache health\n"
         "  How many files are cached, how many match the BEQ catalogue,\n"
         "  and whether you have enough for training.", "cache-status"),
        ("Repair cache\n"
         "  Scan for corrupt audio files and optionally delete them\n"
         "  so they get re-extracted on next use.", "verify"),

        (f"STEP 2: TRAIN MODEL ({model_status})", None),

        ("Train model (recommended)\n"
         "  Builds the production XGBoost model (E82) from your WAV cache.\n"
         "  Takes ~1 minute. Required once before generating profiles.", "dev-train"),
        ("Train torch model\n"
         "  Builds the differentiable-DSP model (E85). More accurate but\n"
         "  requires PyTorch and takes longer. Experimental.", "dev-train-torch"),

        ("STEP 3: ANALYSE & TEST (requires trained model)", None),

        ("NN accuracy report\n"
         "  Compare the model's predictions against hand-tuned BEQ\n"
         "  catalogue entries across your WAV cache.", "nn-report"),
        ("Discover media\n"
         "  Scan media library folders and match titles against the\n"
         "  BEQ catalogue for training and testing.", "sweep-discover"),
        ("Test predictions\n"
         "  Run the model against all discovered media and measure\n"
         "  prediction accuracy.", "sweep-run"),
        ("Experiment results\n"
         "  Compare results from different experiment runs.", "sweep-report"),

        ("REPORTS — data about your cache and the BEQ catalogue", None),

        ("Acquisition recommendations\n"
         "  Which BEQ catalogue titles are missing from your library?\n"
         "  Suggests titles to improve training coverage.", "report-acquisitions"),
        ("Cache bias report\n"
         "  Compare your cache distribution to the full catalogue —\n"
         "  find gaps in genre, era, or author coverage.", "report-cache-bias"),
        ("Author patterns\n"
         "  Per-author analysis of how different BEQ authors apply\n"
         "  correction filters.", "report-author-patterns"),

        ("Back", "back"),
    ]


def _interactive_menu_loop(config: CliConfig, verbose: bool) -> None:
    """Show the main menu in a loop until the user quits."""
    while True:
        try:
            action = menu_select("What would you like to do?", _MAIN_MENU)
        except KeyboardInterrupt:
            break

        if action is None or action == "quit":
            break

        if action == "tools":
            _tools_menu_loop(config, verbose)
            continue

        try:
            _dispatch(action, config, verbose)
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted — returning to menu.[/dim]")
        except RuntimeError as e:
            console.print(f"\n[red]Error:[/red] {e}")
        except SystemExit as e:
            if e.code and e.code != 0:
                console.print(f"[red]Command exited with code {e.code}[/red]")
        except Exception:
            console.print_exception(show_locals=False)

    console.print("[dim]Goodbye.[/dim]")


def _tools_menu_loop(config: CliConfig, verbose: bool) -> None:
    """Show the tools submenu until the user goes back."""
    # WAV count and model status already cached by _validate_config_paths().

    while True:
        try:
            action = menu_select("Tools:", _build_tools_menu())
        except KeyboardInterrupt:
            break

        if action is None or action == "back":
            break

        try:
            _dispatch(action, config, verbose)
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted — returning to tools menu.[/dim]")
        except RuntimeError as e:
            console.print(f"\n[red]Error:[/red] {e}")
        except SystemExit as e:
            if e.code and e.code != 0:
                console.print(f"[red]Command exited with code {e.code}[/red]")
        except Exception:
            console.print_exception(show_locals=False)


def _dispatch(action: str, config: CliConfig, verbose: bool) -> None:
    """Route a menu choice to the same typer command function used by CLI."""
    handlers = {
        "profile": lambda: profile(media=None, author=None, output=None, output_dir=None, verbose=verbose),
        "extract": lambda: extract(media_root=None, beq_dir_opt=None, limit=0, verify_only=False, verbose=False),
        "cache-status": lambda: cache_status(cache_dir=None),
        "verify": lambda: verify(cache_root=None, delete=False, verbose=False),
        "nn-report": lambda: nn_report(output=None),
        "sweep-discover": lambda: sweep_discover(library=None),
        "sweep-run": lambda: sweep_run(limit=10, parallel=False),
        "sweep-report": lambda: sweep_report_cmd(),
        "config": lambda: config_cmd(),
        "dev-train": lambda: dev_train(),
        "dev-train-torch": lambda: dev_train_torch(),
        "report-acquisitions": lambda: report_acquisitions(count=50, output=None),
        "report-cache-bias": lambda: report_cache_bias(output=None),
        "report-author-patterns": lambda: report_author_patterns(output=None),
    }
    handler = handlers.get(action)
    if handler:
        handler()


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _render_markdown_report(report_main_func, argv=None) -> None:
    """Run a report script, capture its markdown stdout, render with Rich."""
    import inspect
    import io
    import contextlib
    from rich.markdown import Markdown

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sig = inspect.signature(report_main_func)
        if sig.parameters:
            report_main_func(argv)
        else:
            report_main_func()

    md_text = buf.getvalue()
    if md_text.strip():
        console.print()
        console.print(Markdown(md_text))
    else:
        console.print("[dim]No output.[/dim]")


_cached_beq_dir: Path | None | bool = False  # False = not yet checked


def _auto_beq_dir() -> Path | None:
    """Get beq_dir from shared config, or None if not configured. Cached."""
    global _cached_beq_dir
    if _cached_beq_dir is not False:
        return _cached_beq_dir
    try:
        from spike._auto_beq_helpers import beq_dir as _bd
        _cached_beq_dir = _bd()
    except Exception:
        _cached_beq_dir = None
    return _cached_beq_dir


_cached_wav_cache: Path | None | bool = False


def _auto_wav_cache() -> Path | None:
    """Get wav_cache_dir from shared config, or None if not configured. Cached."""
    global _cached_wav_cache
    if _cached_wav_cache is not False:
        return _cached_wav_cache
    try:
        from spike._auto_beq_helpers import wav_cache_dir
        _cached_wav_cache = wav_cache_dir()
    except Exception:
        _cached_wav_cache = None
    return _cached_wav_cache


# ---------------------------------------------------------------------------
# Typer subcommands — THE single code path for both menu and CLI
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
    beq_dir_opt: Optional[Path] = typer.Option(None, "--beq-dir", help="BEQ working directory."),  # noqa: UP007
    limit: int = typer.Option(0, "--limit", help="Max titles (0 = unlimited)."),
    verify_only: bool = typer.Option(False, "--verify", help="Verify existing cache only."),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Extract LFE audio from media library to WAV cache."""
    argv: list[str] = ["-v"]  # always verbose — CLI should never be silent
    if media_root:
        for r in media_root:
            argv.extend(["--media-root", str(r)])
    # Auto-populate beq-dir from shared config if not explicitly provided.
    effective_beq = beq_dir_opt or _auto_beq_dir()
    if effective_beq:
        argv.extend(["--beq-dir", str(effective_beq)])
    if limit:
        argv.extend(["--limit", str(limit)])
    if verify_only:
        argv.append("--verify")
    from cli.extract import main as extract_main
    extract_main(argv)


@app.command(name="cache-status")
def cache_status(
    cache_dir: Optional[Path] = typer.Argument(None, help="WAV cache directory."),  # noqa: UP007
) -> None:
    """Show WAV cache counts, titles, and author breakdown."""
    from cli.cache_status import main as status_main
    status_main()


@app.command()
def verify(
    cache_root: Optional[Path] = typer.Argument(None, help="WAV cache root directory."),  # noqa: UP007
    delete: bool = typer.Option(False, "--delete", help="Delete corrupt files."),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Verify WAV cache integrity and optionally fix corrupt files."""
    # Auto-populate cache root from shared config if not provided.
    effective_root = cache_root or _auto_wav_cache()
    argv: list[str] = []
    if effective_root:
        argv.append(str(effective_root))
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
    from cli.nn_report import main as nn_main
    if output:
        nn_main(["--output", str(output)])
    else:
        _render_markdown_report(nn_main)


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
    _render_markdown_report(report_main)


@app.command(name="config")
def config_cmd() -> None:
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
    markers: Optional[str] = typer.Option(None, "--markers", help="Pytest marker filter expression."),  # noqa: UP007
    integration: bool = typer.Option(False, "--integration", help="Run integration tests only."),
    experiments: bool = typer.Option(False, "--experiments", help="Run experiment tests only."),
) -> None:
    """Run spike tests (default: unit only, excludes integration + experiment).

    Torch tests run in a separate pytest invocation to avoid a segfault
    caused by torch + PyQt6 in the same process on macOS (MPS conflict).
    """
    test_selector = file or "src/test/python/spike/"
    env = {**os.environ, "AUTO_BEQ_ADVISOR": advisor}
    base_cmd = ["poetry", "run", "pytest", "-v"]
    if verbose:
        base_cmd.append("-s")

    marker_args = []
    if markers:
        marker_args = ["-m", markers]
    elif integration:
        marker_args = ["-m", "integration"]
    elif experiments:
        marker_args = ["-m", "experiment"]
    else:
        marker_args = ["-m", "not integration and not experiment"]

    # Pass 1: all tests except torch (avoids Qt + torch segfault).
    console.print("[bold]Pass 1:[/bold] Running tests (excluding torch)...")
    cmd1 = base_cmd + [test_selector, "--ignore=src/test/python/spike/test_auto_beq_torch.py"] + marker_args
    r1 = subprocess.run(cmd1, env=env, cwd=str(REPO_ROOT))

    # Pass 2: torch tests only (separate process, no Qt loaded).
    torch_test = "src/test/python/spike/test_auto_beq_torch.py"
    if not file or "torch" in (file or ""):
        console.print("\n[bold]Pass 2:[/bold] Running torch tests (separate process)...")
        cmd2 = base_cmd + [torch_test]
        r2 = subprocess.run(cmd2, env=env, cwd=str(REPO_ROOT))
        if r1.returncode != 0 or r2.returncode != 0:
            raise SystemExit(1)
    elif r1.returncode != 0:
        raise SystemExit(1)


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


@dev_app.command(name="train")
def dev_train() -> None:
    """Train the production XGBoost model (E82 50:1 weighted hybrid)."""
    from cli.train_production_model import main as train_main
    train_main()


@dev_app.command(name="train-torch")
def dev_train_torch() -> None:
    """Train the differentiable-DSP model (E85+)."""
    from cli.train_torch_model import main as train_main
    train_main()


# ---------------------------------------------------------------------------
# Report subcommands
# ---------------------------------------------------------------------------

report_app = typer.Typer(help="Analysis reports — cache bias, author patterns, acquisition recommendations.")
app.add_typer(report_app, name="report")


@report_app.command(name="cache-bias")
def report_cache_bias(
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Save report to file."),  # noqa: UP007
) -> None:
    """Compare WAV cache distribution to the full BEQ catalogue."""
    from cli.nn_cache_bias_report import main as bias_main
    if output:
        bias_main(["-o", str(output)])
    else:
        _render_markdown_report(bias_main)


@report_app.command(name="author-patterns")
def report_author_patterns(
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Save report to file."),  # noqa: UP007
) -> None:
    """Per-author distribution analysis from the BEQ catalogue."""
    from cli.nn_author_pattern_report import main as author_main
    if output:
        author_main(["-o", str(output)])
    else:
        _render_markdown_report(author_main)


@report_app.command(name="acquisitions")
def report_acquisitions(
    count: int = typer.Option(50, "-n", "--count", help="Number of titles to recommend."),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Save report to file."),  # noqa: UP007
) -> None:
    """Recommend missing catalogue titles to acquire (greedy bias correction)."""
    from cli.nn_acquisition_recommender import main as acq_main
    argv = ["-n", str(count)]
    if output:
        argv.extend(["-o", str(output)])
        acq_main(argv)
    else:
        _render_markdown_report(acq_main, argv)


@report_app.command(name="f-experiments")
def report_f_experiments(
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Save report to file."),  # noqa: UP007
) -> None:
    """Generate comparison report from F-experiment CSV results."""
    from cli.nn_f_experiment_report import main as f_main
    if output:
        f_main(["-o", str(output)])
    else:
        _render_markdown_report(f_main)


# ---------------------------------------------------------------------------
# Main callback — interactive menu when no subcommand given
# ---------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show DEBUG messages."),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Only show warnings and errors."),
) -> None:
    """BEQ Designer — run with no subcommand for interactive menus."""
    import sys as _sys

    # Log level: default INFO, -v for DEBUG, -q for WARNING only.
    if quiet:
        log_level = logging.WARNING
    elif verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(level=log_level, handlers=[logging.NullHandler()])

    # All log output goes to stderr — standard CLI behaviour.
    _stderr = logging.StreamHandler(_sys.stderr)
    _stderr.setLevel(log_level)
    _stderr.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(_stderr)

    cfg = load_config()

    if ctx.invoked_subcommand is None:
        # Only show banner and validate for interactive mode.
        log_file = setup_log_file(cfg.output_dir)
        plain_banner = show_banner("BEQ Designer CLI", cfg, log_file=log_file)
        _file_logger = logging.getLogger("beq_cli.banner")
        _file_logger.propagate = False
        for h in logging.getLogger().handlers:
            if isinstance(h, logging.FileHandler):
                _file_logger.addHandler(h)
        _file_logger.info(plain_banner)

        _validate_config_paths()
        _interactive_menu_loop(cfg, verbose)


# ---------------------------------------------------------------------------
# Startup config validation
# ---------------------------------------------------------------------------


def _validate_config_paths() -> None:
    """Check all configured paths at startup. Warn and offer to fix if invalid."""
    import json as _json

    problems: list[str] = []
    _beq = None

    # Shared dir — required, everything derives from it.
    from spike._auto_beq_helpers import beq_dir as _bd
    try:
        _beq = _bd()
    except Exception:
        console.print()
        console.print("[red bold]ERROR: BEQ shared directory is not configured.[/red bold]")
        console.print()
        console.print("Set one of the following:")
        console.print("  • [bold]BEQ_SHARED_DIR[/bold] environment variable")
        console.print("  • [bold]shared_beq_dir[/bold] in ~/.config/beqdesigner/settings.json")
        console.print()
        console.print("Example:")
        console.print("  export BEQ_SHARED_DIR=/path/to/your/beqdesigner")
        console.print()
        raise SystemExit(1)

    # Check if shared dir is empty — might be a misconfigured mount.
    if _beq.exists() and not any(_beq.iterdir()):
        console.print()
        console.print(f"[yellow bold]WARNING: Shared directory is empty:[/yellow bold] {_beq}")
        console.print()
        try:
            from cli.common import menu_select
            answer = menu_select(
                "Is this a new installation?",
                [("Yes — initialise this directory", "yes"),
                 ("No — something is wrong (exit)", "no")],
            )
        except (KeyboardInterrupt, EOFError):
            answer = None
        if answer != "yes":
            console.print()
            console.print("Check that the directory path is correct and any")
            console.print("network mounts are active, then try again.")
            console.print()
            raise SystemExit(1)
        log.info("initialising new shared directory: %s", _beq)

    # WAV cache — derived from shared dir.
    from spike._auto_beq_helpers import wav_cache_dir
    log.info("checking WAV cache configuration...")
    try:
        cache = wav_cache_dir()
        if not cache.exists():
            problems.append(f"WAV cache does not exist: {cache}")
    except RuntimeError as exc:
        problems.append(str(exc))

    # Validate media roots — uses the same resolution as the extract command.
    from cli.extract import get_configured_media_roots
    roots = get_configured_media_roots()
    if roots:
        log.info("validating %d configured media root(s)...", len(roots))
        for r in roots:
            if not r.exists():
                problems.append(f"Media root does not exist: {r}")

    # Pre-cache model status so Tools menu doesn't have to stat the NAS.
    log.info("checking for production model...")
    _model_status_line()  # populates _cached_model_status

    if not problems:
        return

    console.print()
    for p in problems:
        console.print(f"  [yellow]WARNING:[/yellow] {p}")
    console.print()

    # Offer to fix invalid media roots interactively.
    media_problems = [p for p in problems if "Media root" in p]
    if media_problems and _beq and sys.stdin.isatty():
        from InquirerPy import inquirer
        fix = inquirer.confirm(
            message="Update media roots now?",
            default=True,
        ).execute()
        if fix:
            _repair_media_roots(_beq / ".extract_config.json")


def _repair_media_roots(config_path: Path) -> None:
    """Prompt user for new media roots, save to .extract_config.json.

    Supports:
    - One path per prompt (empty to finish)
    - Comma-separated paths: /path/a, /path/b
    - Escaped spaces (backslash) or unescaped spaces
    """
    import json as _json
    from InquirerPy import inquirer

    new_roots: list[str] = []
    console.print("[bold]Enter media library paths[/bold]")
    console.print("[dim]One per line, or comma-separated. Empty to finish.[/dim]")
    while True:
        raw = inquirer.filepath(message="Media root:", default="").execute()
        if not raw:
            break
        # Support comma-separated paths.
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        for part in parts:
            part = part.replace("\\ ", " ")
            p = Path(part).resolve()
            if p.is_dir():
                new_roots.append(str(p))
                console.print(f"  [green]✓[/green] {p}")
            else:
                console.print(f"  [red]✗ Not a directory: {p}[/red]")
    if new_roots:
        config_path.write_text(_json.dumps({"media_roots": new_roots}, indent=2) + "\n")
        console.print(f"\n[green]Saved {len(new_roots)} media roots to {config_path}[/green]")
    else:
        console.print("[yellow]No valid roots entered — config unchanged.[/yellow]")


def main():
    app()


if __name__ == "__main__":
    main()
