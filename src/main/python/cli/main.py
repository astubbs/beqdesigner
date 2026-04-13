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

def _cache_status_line() -> str:
    """Quick status: how many WAVs in cache."""
    try:
        from spike._auto_beq_helpers import wav_cache_dir
        cache = wav_cache_dir()
        wavs = list(cache.rglob("*.wav"))
        return f"✓ {len(wavs)} WAVs cached"
    except Exception:
        return "✗ not configured — set wav_cache_dir in settings.json"


def _model_status_line() -> str:
    """Quick status: is production model present."""
    from cli.profile import _check_production_model
    if _check_production_model():
        return "✓ production model found"
    return "✗ no model — train one first"


def _build_tools_menu() -> list[tuple[str, object]]:
    """Build the Tools menu dynamically with live status."""
    cache_status = _cache_status_line()
    model_status = _model_status_line()

    return [
        # --- Step 1: Cache ---
        (f"STEP 1: EXTRACT AUDIO ({cache_status})", None),

        ("Pre-extract audio\n"
         "      Extract bass/LFE from media files ahead of time so profile\n"
         "      generation is instant. Best run on the machine closest to\n"
         "      your media drives (NAS, or via Docker).", "extract"),
        ("Check cache health\n"
         "      How many files are cached, how many match the BEQ catalogue,\n"
         "      and whether you have enough for training.", "cache-status"),
        ("Repair cache\n"
         "      Scan for corrupt audio files and optionally delete them\n"
         "      so they get re-extracted on next use.", "verify"),

        # --- Step 2: Train ---
        (f"STEP 2: TRAIN MODEL ({model_status})", None),

        ("Train model (recommended)\n"
         "      Builds the production XGBoost model (E82) from your WAV\n"
         "      cache. Takes ~1 minute. Required once before generating\n"
         "      profiles.", "dev-train"),
        ("Train torch model\n"
         "      Builds the differentiable-DSP model (E85). More accurate\n"
         "      but requires PyTorch and takes longer. Experimental.", "dev-train-torch"),

        # --- Step 3: Analyse ---
        ("STEP 3: ANALYSE & TEST (requires trained model)", None),

        ("NN accuracy report\n"
         "      Compare the model's predictions against hand-tuned BEQ\n"
         "      catalogue entries across your WAV cache.", "nn-report"),
        ("Discover media\n"
         "      Scan media library folders and match titles against the\n"
         "      BEQ catalogue for training and testing.", "sweep-discover"),
        ("Test predictions\n"
         "      Run the model against all discovered media and measure\n"
         "      prediction accuracy.", "sweep-run"),
        ("Experiment results\n"
         "      Compare results from different experiment runs.", "sweep-report"),

        # --- Reports ---
        ("REPORTS — data about your cache and the BEQ catalogue", None),

        ("Acquisition recommendations\n"
         "      Which BEQ catalogue titles are missing from your library?\n"
         "      Suggests titles to improve training coverage.", "report-acquisitions"),
        ("Cache bias report\n"
         "      Compare your cache distribution to the full catalogue —\n"
         "      find gaps in genre, era, or author coverage.", "report-cache-bias"),
        ("Author patterns\n"
         "      Per-author analysis of how different BEQ authors apply\n"
         "      correction filters.", "report-author-patterns"),

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
        except SystemExit:
            pass  # typer.Exit() from subcommands

    console.print("[dim]Goodbye.[/dim]")


def _tools_menu_loop(config: CliConfig, verbose: bool) -> None:
    """Show the tools submenu until the user goes back."""
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
        "dev-train": _do_train,
        "dev-train-torch": _do_train_torch,
        "report-acquisitions": _do_report_acquisitions,
        "report-cache-bias": _do_report_cache_bias,
        "report-author-patterns": _do_report_author_patterns,
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
    from InquirerPy import inquirer
    from spike._auto_beq_helpers import audio_cache_dir

    try:
        default_cache = str(audio_cache_dir())
    except RuntimeError:
        default_cache = ""

    media_roots = []
    console.print("[bold]Add media library roots[/bold] (press Enter with empty path to finish):")
    while True:
        root = inquirer.filepath(message="Media root (empty to finish):", default="").execute()
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

    limit = inquirer.text(message="Max titles to extract (0 = unlimited):", default="0").execute()

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
    from InquirerPy import inquirer
    from spike._auto_beq_helpers import wav_cache_dir

    try:
        default_cache = str(wav_cache_dir())
    except Exception:
        default_cache = ""

    cache_root = inquirer.filepath(message="WAV cache root:", default=default_cache).execute()
    if not cache_root:
        return
    delete = inquirer.confirm(message="Delete corrupt files?", default=False).execute()

    argv = [cache_root]
    if delete:
        argv.append("--delete")
    if verbose:
        argv.append("-v")

    from cli.verify_cache import main as verify_main
    verify_main(argv)


def _do_nn_report(config: CliConfig, verbose: bool) -> None:
    """NN comparison report — delegates to nn_comparison_report."""
    from InquirerPy import inquirer

    output = inquirer.filepath(message="Save report to (empty for stdout):", default="").execute()
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
    from InquirerPy import inquirer

    limit = inquirer.text(message="Max files to process:", default="10").execute()
    parallel = inquirer.confirm(message="Run in parallel?", default=False).execute()

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


def _do_train(config: CliConfig, verbose: bool) -> None:
    """Train the production XGBoost model."""
    from cli.train_production_model import main as train_main
    train_main()


def _do_train_torch(config: CliConfig, verbose: bool) -> None:
    """Train the differentiable-DSP model."""
    from cli.train_torch_model import main as train_main
    train_main()


def _do_report_acquisitions(config: CliConfig, verbose: bool) -> None:
    """Acquisition recommendations."""
    from cli.nn_acquisition_recommender import main as acq_main
    acq_main()


def _do_report_cache_bias(config: CliConfig, verbose: bool) -> None:
    """Cache bias report."""
    from cli.nn_cache_bias_report import main as bias_main
    bias_main()


def _do_report_author_patterns(config: CliConfig, verbose: bool) -> None:
    """Author patterns report."""
    from cli.nn_author_pattern_report import main as author_main
    author_main()


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
    markers: Optional[str] = typer.Option(None, "--markers", help="Pytest marker filter expression."),  # noqa: UP007
    integration: bool = typer.Option(False, "--integration", help="Run integration tests only."),
    experiments: bool = typer.Option(False, "--experiments", help="Run experiment tests only."),
) -> None:
    """Run spike tests (default: unit only, excludes integration + experiment)."""
    test_selector = file or "src/test/python/spike/"
    cmd = ["poetry", "run", "pytest", test_selector, "-v"]
    if verbose:
        cmd.append("-s")
    # Marker filtering.
    if markers:
        cmd.extend(["-m", markers])
    elif integration:
        cmd.extend(["-m", "integration"])
    elif experiments:
        cmd.extend(["-m", "experiment"])
    else:
        cmd.extend(["-m", "not integration and not experiment"])
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
    argv = ["-o", str(output)] if output else []
    from cli.nn_cache_bias_report import main as bias_main
    bias_main(argv)


@report_app.command(name="author-patterns")
def report_author_patterns(
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Save report to file."),  # noqa: UP007
) -> None:
    """Per-author distribution analysis from the BEQ catalogue."""
    argv = ["-o", str(output)] if output else []
    from cli.nn_author_pattern_report import main as author_main
    author_main(argv)


@report_app.command(name="acquisitions")
def report_acquisitions(
    count: int = typer.Option(50, "-n", "--count", help="Number of titles to recommend."),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Save report to file."),  # noqa: UP007
) -> None:
    """Recommend missing catalogue titles to acquire (greedy bias correction)."""
    argv = ["-n", str(count)]
    if output:
        argv.extend(["-o", str(output)])
    from cli.nn_acquisition_recommender import main as acq_main
    acq_main(argv)


@report_app.command(name="f-experiments")
def report_f_experiments(
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Save report to file."),  # noqa: UP007
) -> None:
    """Generate comparison report from F-experiment CSV results."""
    argv = ["-o", str(output)] if output else []
    from cli.nn_f_experiment_report import main as f_main
    f_main(argv)


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
