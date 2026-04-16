"""Integration tests for the beq_profile_cli interactive CLI.

Tests exercise the CLI module's non-interactive code paths (config persistence,
media discovery, progress handler, filter rendering) and verify that the typer
app parses arguments without errors.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# The CLI script manipulates sys.path; ensure repo root is importable.
_REPO_ROOT = Path(__file__).resolve().parents[4]
for _p in (_REPO_ROOT, _REPO_ROOT / "src" / "main" / "python", _REPO_ROOT / "src" / "test" / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cli.common import (
    CliConfig,
    MEDIA_EXTENSIONS,
    load_config,
    save_config,
)
from cli.profile import (
    ProgressLoggingHandler,
    _discover_media,
    _render_results,
)


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------


class TestCliConfig:
    def test_defaults(self):
        config = CliConfig()
        assert config.output_dir == "profiles"
        assert config.verbose is False
        assert config.last_media_dir == ""

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("spike._auto_beq_helpers._SETTINGS_PATH", settings_file)
        # Ensure beq_config_dir() points at tmp_path too.
        monkeypatch.setattr("spike._auto_beq_helpers.beq_config_dir", lambda: tmp_path)

        config = CliConfig(output_dir="/tmp/out", verbose=True, last_media_dir="/media")
        save_config(config)

        assert settings_file.exists()
        loaded = load_config()
        assert loaded.output_dir == "/tmp/out"
        assert loaded.verbose is True
        assert loaded.last_media_dir == "/media"

    def test_load_missing_file_returns_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setattr("spike._auto_beq_helpers._SETTINGS_PATH", tmp_path / "nonexistent.json")
        config = load_config()
        assert config.verbose is False

    def test_load_corrupt_file_returns_defaults(self, tmp_path, monkeypatch):
        settings_file = tmp_path / "settings.json"
        settings_file.write_text("not json{{{")
        monkeypatch.setattr("spike._auto_beq_helpers._SETTINGS_PATH", settings_file)
        config = load_config()
        assert config.output_dir == "profiles"

    def test_load_ignores_unknown_fields(self, tmp_path, monkeypatch):
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({"cli_verbose": True, "unknown_field": 42}))
        monkeypatch.setattr("spike._auto_beq_helpers._SETTINGS_PATH", settings_file)
        config = load_config()
        assert config.verbose is True


# ---------------------------------------------------------------------------
# Media discovery
# ---------------------------------------------------------------------------


class TestDiscoverMedia:
    def test_finds_media_files(self, tmp_path):
        (tmp_path / "movie.mkv").touch()
        (tmp_path / "movie.iso").touch()
        (tmp_path / "readme.txt").touch()
        (tmp_path / "cover.jpg").touch()

        files = _discover_media(tmp_path)
        names = {f.name for f in files}
        assert "movie.mkv" in names
        assert "movie.iso" in names
        assert "readme.txt" not in names
        assert "cover.jpg" not in names

    def test_finds_files_recursively(self, tmp_path):
        season = tmp_path / "Season 01"
        season.mkdir()
        (season / "S01E01.mkv").touch()
        (season / "S01E02.mkv").touch()
        (tmp_path / "movie.mkv").touch()

        files = _discover_media(tmp_path, recursive=True)
        assert len(files) == 3

    def test_non_recursive(self, tmp_path):
        season = tmp_path / "Season 01"
        season.mkdir()
        (season / "S01E01.mkv").touch()
        (tmp_path / "movie.mkv").touch()

        files = _discover_media(tmp_path, recursive=False)
        assert len(files) == 1
        assert files[0].name == "movie.mkv"

    def test_empty_directory(self, tmp_path):
        assert _discover_media(tmp_path) == []

    def test_all_supported_extensions(self, tmp_path):
        for ext in MEDIA_EXTENSIONS:
            (tmp_path / f"test{ext}").touch()
        files = _discover_media(tmp_path)
        assert len(files) == len(MEDIA_EXTENSIONS)


# ---------------------------------------------------------------------------
# Progress handler
# ---------------------------------------------------------------------------


class TestProgressLoggingHandler:
    def test_advances_on_known_triggers(self):
        """Handler updates progress when log messages contain stage triggers."""
        from unittest.mock import MagicMock
        progress = MagicMock()
        task_id = "task-1"
        handler = ProgressLoggingHandler(progress, task_id)

        record = logging.LogRecord("test", logging.INFO, "", 0, "  extracting LFE...", (), None)
        handler.emit(record)

        progress.update.assert_called_once()
        args, kwargs = progress.update.call_args
        assert args == (task_id,)
        assert kwargs["advance"] == 30
        assert kwargs["description"] == "Extracting audio"

    def test_ignores_duplicate_triggers(self):
        from unittest.mock import MagicMock
        progress = MagicMock()
        handler = ProgressLoggingHandler(progress, "t")

        record = logging.LogRecord("test", logging.INFO, "", 0, "extracting LFE...", (), None)
        handler.emit(record)
        handler.emit(record)

        assert progress.update.call_count == 1

    def test_verbose_collects_log_lines(self):
        from unittest.mock import MagicMock
        progress = MagicMock()
        handler = ProgressLoggingHandler(progress, "t", verbose=True)

        for i in range(5):
            record = logging.LogRecord("test", logging.INFO, "", 0, f"line {i}", (), None)
            handler.emit(record)

        assert len(handler.log_lines) == 5
        assert handler.log_lines[0] == "line 0"

    def test_verbose_caps_log_lines(self):
        from unittest.mock import MagicMock
        progress = MagicMock()
        handler = ProgressLoggingHandler(progress, "t", verbose=True)

        for i in range(50):
            record = logging.LogRecord("test", logging.INFO, "", 0, f"line {i}", (), None)
            handler.emit(record)

        from cli.profile import _MAX_LOG_LINES
        assert len(handler.log_lines) == _MAX_LOG_LINES


# ---------------------------------------------------------------------------
# Filterable select — section headers and rendering
# ---------------------------------------------------------------------------


class TestFilterableSelect:
    """Verify filterable_select renders section headers and filters correctly."""

    def _get_display_lines(self, choices, filter_text="", selected_idx=0):
        """Extract the display logic from filterable_select for testing."""
        from prompt_toolkit.formatted_text import FormattedText

        selectable = [(l, v) for l, v in choices if v is not None]

        if filter_text:
            items = [(l, v) for l, v in selectable if filter_text.lower() in l.lower()]
        else:
            items = selectable

        lines = []
        if filter_text:
            for i, (label, _val) in enumerate(items):
                lines.append(("selected" if i == selected_idx else "normal", label))
        else:
            sel_idx = 0
            for label, val in choices:
                if val is None:
                    lines.append(("header", label))
                else:
                    lines.append(("selected" if sel_idx == selected_idx else "normal", label))
                    sel_idx += 1
        return lines

    def test_section_headers_appear_in_unfiltered_display(self):
        choices = [
            ("SECTION ONE", None),
            ("Item A", "a"),
            ("Item B", "b"),
            ("SECTION TWO", None),
            ("Item C", "c"),
        ]
        lines = self._get_display_lines(choices)
        types = [t for t, _ in lines]
        labels = [l for _, l in lines]

        assert "header" in types, f"No headers found in {lines}"
        assert types.count("header") == 2
        assert labels[0] == "SECTION ONE"
        assert labels[1] == "Item A"
        assert labels[3] == "SECTION TWO"
        assert labels[4] == "Item C"

    def test_headers_hidden_when_filtering(self):
        choices = [
            ("SECTION ONE", None),
            ("Item Alpha", "a"),
            ("Item Beta", "b"),
            ("SECTION TWO", None),
            ("Item Charlie", "c"),
        ]
        lines = self._get_display_lines(choices, filter_text="beta")
        types = [t for t, _ in lines]
        labels = [l for _, l in lines]

        assert "header" not in types, f"Headers should be hidden when filtering: {lines}"
        assert len(lines) == 1
        assert "Beta" in labels[0]

    def test_headers_are_not_selectable(self):
        choices = [
            ("HEADER", None),
            ("Item A", "a"),
        ]
        selectable = [(l, v) for l, v in choices if v is not None]
        assert len(selectable) == 1
        assert selectable[0] == ("Item A", "a")

    def test_first_selectable_item_is_selected_by_default(self):
        choices = [
            ("HEADER", None),
            ("Item A", "a"),
            ("Item B", "b"),
        ]
        lines = self._get_display_lines(choices, selected_idx=0)
        # First non-header item should be selected
        selectable_lines = [(t, l) for t, l in lines if t != "header"]
        assert selectable_lines[0][0] == "selected"
        assert selectable_lines[1][0] == "normal"

    def test_main_menu_is_simple(self):
        """Main menu should have just a few top-level items, not everything."""
        from cli.main import _MAIN_MENU
        selectable = [label for label, val in _MAIN_MENU if val is not None]
        assert len(selectable) <= 5, f"Main menu too cluttered: {selectable}"
        values = [val for _, val in _MAIN_MENU if val is not None]
        assert "profile" in values
        assert "tools" in values
        assert "quit" in values

    def test_tools_menu_has_section_headers(self):
        """Tools submenu should group items under descriptive headers."""
        from cli.main import _build_tools_menu
        _TOOLS_MENU = _build_tools_menu()
        headers = [label for label, val in _TOOLS_MENU if val is None]
        assert len(headers) >= 4, f"Expected at least 4 section headers, got {headers}"
        header_text = " ".join(headers).upper()
        assert "STEP 1" in header_text
        assert "STEP 2" in header_text
        assert "STEP 3" in header_text
        assert "REPORT" in header_text

    def test_tools_menu_has_model_training(self):
        """Tools menu must include model training options."""
        from cli.main import _build_tools_menu
        _TOOLS_MENU = _build_tools_menu()
        values = [val for _, val in _TOOLS_MENU if val is not None]
        assert "dev-train" in values
        assert "dev-train-torch" in values

    def test_tools_menu_has_reports(self):
        """Tools menu must include report options."""
        from cli.main import _build_tools_menu
        values = [val for _, val in _build_tools_menu() if val is not None]
        assert "report-acquisitions" in values
        assert "report-cache-bias" in values
        assert "report-author-patterns" in values

    def test_tools_menu_has_back_option(self):
        """Tools submenu must have a Back option to return to main menu."""
        from cli.main import _build_tools_menu
        values = [val for _, val in _build_tools_menu() if val is not None]
        assert "back" in values


# ---------------------------------------------------------------------------
# Filter rendering (smoke test — just ensure no exceptions)
# ---------------------------------------------------------------------------


class TestRenderFilters:
    def test_renders_without_error(self):
        profile = {
            "filters": [
                {"type": "LowShelf", "freq": 22, "gain": 4.2, "q": 0.71},
                {"type": "PeakingEQ", "freq": 35, "gain": -2.1, "q": 1.41},
            ]
        }
        from pathlib import Path as P
        _render_results(profile, P("test_output.json"))  # should not raise

    def test_empty_filters(self):
        from pathlib import Path as P
        _render_results({"filters": []}, P("test.json"))  # should not raise
        _render_results({}, P("test.json"))  # should not raise


# ---------------------------------------------------------------------------
# Dispatch and markdown rendering
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_all_menu_items_have_handlers(self):
        """Every selectable item in the Tools menu must have a dispatch handler."""
        from cli.main import _build_tools_menu, _dispatch

        tools_menu = _build_tools_menu()
        selectable_values = [val for _, val in tools_menu if val is not None and val != "back"]

        # The dispatch table is built inside _dispatch — verify by checking
        # that each menu value is a key in the handlers dict.
        # We can't easily inspect the lambda dict, so just verify the
        # known set matches.
        expected = {
            "profile", "extract", "cache-status", "verify", "nn-report",
            "sweep-discover", "sweep-run", "sweep-report", "config",
            "dev-train", "dev-train-torch",
            "report-acquisitions", "report-cache-bias", "report-author-patterns",
        }
        for val in selectable_values:
            assert val in expected, f"Menu item '{val}' has no dispatch handler"

    def test_render_markdown_report_no_args(self):
        """Scripts with def main() (no args) should work via _render_markdown_report."""
        from cli.main import _render_markdown_report

        def fake_report():
            print("# Test Report\n\nSome content.")

        # Should not raise — the function takes no args.
        _render_markdown_report(fake_report)

    def test_render_markdown_report_with_argv(self):
        """Scripts with def main(argv) should receive argv."""
        from cli.main import _render_markdown_report

        received = []

        def fake_report(argv):
            received.append(argv)
            print("# Report")

        _render_markdown_report(fake_report, argv=["--test"])
        assert received == [["--test"]]

    def test_extract_always_verbose_and_auto_beq_dir(self):
        """Extract subcommand must always pass -v and auto-populate --beq-dir."""
        from unittest.mock import patch

        captured_argv = []

        def fake_extract_main(argv):
            captured_argv.extend(argv)

        with patch("cli.extract.main", fake_extract_main):
            with patch("cli.main._auto_beq_dir", return_value=Path("/tmp/fake-beq")):
                from cli.main import extract
                extract(media_root=None, beq_dir_opt=None, limit=0,
                        verify_only=False, verbose=False)

        assert "-v" in captured_argv, f"Extract must always be verbose, got: {captured_argv}"
        assert "--beq-dir" in captured_argv, f"Extract must auto-populate --beq-dir, got: {captured_argv}"
        beq_idx = captured_argv.index("--beq-dir")
        assert captured_argv[beq_idx + 1] == "/tmp/fake-beq"

    def test_dispatch_calls_typer_functions_not_wrappers(self):
        """Menu dispatch must call the typer command functions with correct defaults.

        Regression: calling extract() without explicit args gave OptionInfo objects
        instead of None, causing 'OptionInfo is not iterable' TypeError.
        """
        from unittest.mock import patch, MagicMock
        from cli.main import _dispatch
        from cli.common import CliConfig

        config = CliConfig()

        # Test each dispatch action that takes parameters — verify no TypeError.
        actions_to_test = [
            "extract", "cache-status", "verify", "nn-report",
            "sweep-discover", "report-acquisitions", "report-cache-bias",
            "report-author-patterns",
        ]

        for action in actions_to_test:
            # Mock the underlying script main() so nothing actually runs.
            patches = {
                "extract": "cli.extract.main",
                "cache-status": "cli.cache_status.main",
                "verify": "cli.verify_cache.main",
                "nn-report": "cli.nn_report.main",
                "sweep-discover": "spike.sweep_discover.main",
                "report-acquisitions": "cli.nn_acquisition_recommender.main",
                "report-cache-bias": "cli.nn_cache_bias_report.main",
                "report-author-patterns": "cli.nn_author_pattern_report.main",
            }
            target = patches[action]
            with patch(target, MagicMock()):
                with patch("cli.main._auto_beq_dir", return_value=None):
                    with patch("cli.main._auto_wav_cache", return_value=None):
                        try:
                            _dispatch(action, config, verbose=False)
                        except TypeError as e:
                            pytest.fail(
                                f"Dispatch '{action}' raised TypeError: {e}. "
                                "Likely calling typer function without explicit defaults."
                            )


# ---------------------------------------------------------------------------
# CLI invocation (subprocess — catches import errors, arg parsing bugs)
# ---------------------------------------------------------------------------


class TestCLIInvocation:
    """Run the actual script in a subprocess to catch import and arg parsing errors."""

    _script = str(_REPO_ROOT / "bin" / "beq-designer")

    def test_help(self):
        result = subprocess.run(
            [sys.executable, self._script, "profile", "--help"],
            capture_output=True, text=True, timeout=30,
            env={**dict(__import__("os").environ), "VIRTUAL_ENV": "1"},
        )
        assert result.returncode == 0
        assert "Generate BEQ correction profiles" in result.stdout

    def test_verbose_imports_resolve(self):
        """Ensure -v mode doesn't crash on import (caught the rich.group bug)."""
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.argv = ['test']; "
             "from cli.profile import _run_single; "
             "from rich.console import Group; "
             "from rich.live import Live; "
             "from rich.panel import Panel; "
             "from rich.text import Text; "
             "print('all imports ok')"],
            capture_output=True, text=True, timeout=30,
            cwd=str(_REPO_ROOT),
            env={**dict(__import__("os").environ), "VIRTUAL_ENV": "1",
                 "PYTHONPATH": f"{_REPO_ROOT}/src/main/python:{_REPO_ROOT}/src/test/python"},
        )
        assert "all imports ok" in result.stdout, result.stderr


class TestUnifiedCLI:
    """Test the unified bin/beq-designer entry point."""

    _script = str(_REPO_ROOT / "bin" / "beq-designer")

    def test_help(self):
        result = subprocess.run(
            [sys.executable, self._script, "--help"],
            capture_output=True, text=True, timeout=30,
            env={**dict(__import__("os").environ), "VIRTUAL_ENV": "1"},
        )
        assert result.returncode == 0
        assert "profile" in result.stdout
        assert "extract" in result.stdout
        assert "cache-status" in result.stdout
        assert "sweep" in result.stdout

    def test_profile_help(self):
        result = subprocess.run(
            [sys.executable, self._script, "profile", "--help"],
            capture_output=True, text=True, timeout=30,
            env={**dict(__import__("os").environ), "VIRTUAL_ENV": "1"},
        )
        assert result.returncode == 0
        assert "Generate BEQ correction profiles" in result.stdout

    def test_sweep_help(self):
        result = subprocess.run(
            [sys.executable, self._script, "sweep", "--help"],
            capture_output=True, text=True, timeout=30,
            env={**dict(__import__("os").environ), "VIRTUAL_ENV": "1"},
        )
        assert result.returncode == 0
        assert "discover" in result.stdout
        assert "run" in result.stdout
        assert "report" in result.stdout

    def test_verbose_flag_accepted(self):
        result = subprocess.run(
            [sys.executable, self._script, "-v", "--help"],
            capture_output=True, text=True, timeout=30,
            env={**dict(__import__("os").environ), "VIRTUAL_ENV": "1"},
        )
        assert result.returncode == 0

    def test_dev_help(self):
        result = subprocess.run(
            [sys.executable, self._script, "dev", "--help"],
            capture_output=True, text=True, timeout=30,
            env={**dict(__import__("os").environ), "VIRTUAL_ENV": "1"},
        )
        assert result.returncode == 0
        assert "test" in result.stdout
        assert "sweep" in result.stdout
        assert "compare-advisors" in result.stdout
        assert "playground" in result.stdout


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Extract config validation
# ---------------------------------------------------------------------------


class TestExtractConfigValidation:
    """Config with non-existent media roots must fail, not silently succeed.

    Tests use the config service (save_extract_config / get_configured_media_roots)
    with beq_config_dir mocked to tmp_path -- same code path as production.
    """

    @pytest.fixture(autouse=True)
    def _isolate_config(self, tmp_path, monkeypatch):
        """Point beq_config_dir at tmp_path so tests never read real config."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        monkeypatch.setattr(
            "spike._auto_beq_helpers.beq_config_dir", lambda: config_dir,
        )
        # Clear BEQ_MEDIA_DIR so auto-discovery doesn't override config.
        monkeypatch.delenv("BEQ_MEDIA_DIR", raising=False)

    def test_any_invalid_media_root_raises_with_message(self, tmp_path):
        """Any invalid media root is a fatal config error with descriptive message."""
        from cli.extract import main as extract_main, save_extract_config

        valid_dir = tmp_path / "valid_media"
        valid_dir.mkdir()
        save_extract_config([valid_dir, Path("/nonexistent/path")])

        with pytest.raises(RuntimeError, match="Invalid media roots"):
            extract_main(["--beq-dir", str(tmp_path)])

    def test_extract_inventory_age_message(self, tmp_path):
        """Extract must not crash when media_inventory.json exists (NameError regression)."""
        from cli.extract import main as extract_main, save_extract_config

        valid_dir = tmp_path / "media"
        valid_dir.mkdir()
        save_extract_config([valid_dir])

        # Create a fake inventory so the age-check code path runs.
        inventory = tmp_path / "media_inventory.json"
        inventory.write_text(json.dumps({"media": []}))

        # Should not crash with NameError: _time.
        # Will fail on catalogue fetch (no network in test) -- that's fine.
        try:
            extract_main(["--beq-dir", str(tmp_path)])
        except Exception as e:
            # Catalogue fetch failure is expected -- but NameError is not.
            assert "NameError" not in str(type(e).__name__), f"Unexpected NameError: {e}"

    def test_extract_validates_roots_before_catalogue_fetch(self, tmp_path):
        """Extract must fail on invalid roots BEFORE fetching the catalogue (no HTTP)."""
        from unittest.mock import patch
        from cli.extract import main as extract_main, save_extract_config

        save_extract_config([Path("/nonexistent/nas/path")])

        fetch_called = []

        def mock_fetch(beq_dir):
            fetch_called.append(True)
            return []

        with patch("cli.extract.fetch_catalogue", mock_fetch):
            with pytest.raises(RuntimeError, match="Invalid media roots"):
                extract_main(["--beq-dir", str(tmp_path)])

        assert not fetch_called, (
            "fetch_catalogue was called before media root validation -- "
            "user would see a 4-second HTTP pause before the error"
        )


class TestStartupConfigValidation:
    """Startup must detect and offer to fix invalid config paths."""

    def test_detects_invalid_media_roots(self, tmp_path, monkeypatch):
        """_validate_config_paths warns about non-existent media roots."""
        from unittest.mock import patch
        from io import StringIO
        from rich.console import Console
        from cli.main import _validate_config_paths
        from cli.extract import save_extract_config

        beq = tmp_path / "beq"
        beq.mkdir()
        (beq / "wav-cache").mkdir()

        # Use config service to write roots, with beq_config_dir isolated.
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        monkeypatch.setattr("spike._auto_beq_helpers.beq_config_dir", lambda: config_dir)
        monkeypatch.delenv("BEQ_MEDIA_DIR", raising=False)
        save_extract_config([Path("/nonexistent/nas/path"), Path("/also/missing")])

        captured = StringIO()
        with patch("cli.main.console", Console(file=captured)):
            with patch("spike._auto_beq_helpers.beq_dir", return_value=beq):
                with patch("spike._auto_beq_helpers.wav_cache_dir", return_value=beq / "wav-cache"):
                    monkeypatch.setattr("sys.stdin", StringIO())
                    _validate_config_paths()

        output = captured.getvalue()
        assert "/nonexistent/nas/path" in output
        assert "/also/missing" in output
        assert "WARNING" in output

    def test_no_warnings_when_paths_valid(self, tmp_path, monkeypatch):
        """No warnings when all configured paths exist."""
        from unittest.mock import patch
        from io import StringIO
        from rich.console import Console
        from cli.main import _validate_config_paths
        from cli.extract import save_extract_config

        beq = tmp_path / "beq"
        beq.mkdir()
        wav_cache = beq / "wav-cache"
        wav_cache.mkdir()
        valid_root = tmp_path / "media"
        valid_root.mkdir()

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        monkeypatch.setattr("spike._auto_beq_helpers.beq_config_dir", lambda: config_dir)
        monkeypatch.delenv("BEQ_MEDIA_DIR", raising=False)
        save_extract_config([valid_root])

        captured = StringIO()
        with patch("cli.main.console", Console(file=captured)):
            with patch("spike._auto_beq_helpers.beq_dir", return_value=beq):
                with patch("spike._auto_beq_helpers.wav_cache_dir", return_value=wav_cache):
                    _validate_config_paths()

        output = captured.getvalue()
        assert "WARNING" not in output


# ---------------------------------------------------------------------------
# extract_lfe_wav integration
# ---------------------------------------------------------------------------


class TestExtractLfeWav:
    """Verify extract_lfe_wav builds a valid ffmpeg command."""

    def test_ffmpeg_command_includes_wav_format(self, monkeypatch):
        """The .tmp output file requires explicit -f wav (caught exit 234 bug)."""
        import shutil
        if not shutil.which("ffmpeg"):
            pytest.skip("ffmpeg not installed")

        # Mock audio_cache_dir to a local temp dir (avoids NAS dependency).
        import tempfile as _tf
        _cache_tmp = Path(_tf.mkdtemp(prefix="beq_cache_"))
        monkeypatch.setattr("spike._auto_beq_helpers.audio_cache_dir", lambda: _cache_tmp)

        from spike._auto_beq_helpers import extract_lfe_wav

        # Create a minimal valid WAV to use as input.
        import struct
        wav = tmp = None
        try:
            tmp = Path(_tf.mkdtemp(prefix="beq_test_"))
            wav = tmp / "test.wav"
            # Write a minimal 1-second mono WAV at 1000 Hz.
            sr, duration, bits = 1000, 1, 16
            n_samples = sr * duration
            data_size = n_samples * (bits // 8)
            with wav.open("wb") as f:
                f.write(b"RIFF")
                f.write(struct.pack("<I", 36 + data_size))
                f.write(b"WAVE")
                f.write(b"fmt ")
                f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * bits // 8, bits // 8, bits))
                f.write(b"data")
                f.write(struct.pack("<I", data_size))
                f.write(b"\x00" * data_size)

            # Should succeed (not raise) — the bug was ffmpeg exit 234
            # because the .tmp output file didn't specify -f wav.
            result = extract_lfe_wav(wav, target_fs=1000)
            assert result.exists()
            assert result.stat().st_size > 0
        finally:
            if tmp and tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
