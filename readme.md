[![Build Status](https://github.com/3ll3d00d/beqdesigner/workflows/run%20tests/badge.svg)](https://github.com/3ll3d00d/beqdesigner/actions)

# BEQDesigner

A Qt desktop app for designing, analysing and applying Bass EQ (BEQ) filters for
movie soundtracks.

**User documentation lives at [beqdesigner.readthedocs.io](https://beqdesigner.readthedocs.io/).**
Start there if you want to know what BEQ is, how to install a release build, or
how to use the app. This README is for developers working on the source.

## Developer quickstart

Requirements: Python 3.13, [Poetry](https://python-poetry.org/), and (optional,
runtime only) `ffmpeg` + `graphviz`.

```sh
poetry env use python3.13
poetry install
PYTHONPATH=./src/main/python poetry run python src/main/python/app.py
```

The `PYTHONPATH` prefix matches how CI invokes the app and tests — the source
root is `src/main/python`, not the repo root.

### BEQ CLI

Single entry point for all operations — profile generation, LFE extraction,
cache management, experiment analysis. Run with no arguments for an interactive
menu, or use subcommands directly:

```sh
bin/beq-designer                                    # interactive menu
bin/beq-designer profile "Avatar (2009).mkv"        # generate profile
bin/beq-designer extract --media-root /mnt/media    # extract LFE cache
bin/beq-designer cache-status                       # WAV cache info
bin/beq-designer discover                            # discover + match media
bin/beq-designer --help                             # list all subcommands
```

Preferences are saved on first run and remembered for future invocations.

### Tests

```sh
PYTHONPATH=./src/main/python poetry run pytest --cov=./src/main/python
```

Tests live in `src/test/python/` and run on every push via
`.github/workflows/test.yaml` across Linux, macOS and Windows.

### Building the app bundle

Release binaries are produced by PyInstaller from `beqdesigner.spec`:

```sh
poetry run pip install pyinstaller
poetry run pyinstaller --clean --log-level=INFO beqdesigner.spec
```

On macOS this yields `dist/beqdesigner.app`. On Linux a virtual X server is
required (`Xvfb`) — see the `Create distribution` step in `test.yaml` for the
exact invocation used in CI.

## Project layout

| Path | Contents |
|---|---|
| `src/main/python/app.py` | Entry point — creates `QApplication` and `BeqDesigner` main window |
| `src/main/python/ui/` | Qt Designer `.ui` files and their generated `.py` equivalents, plus view code |
| `src/main/python/model/` | Non-UI logic: filters, signals, IIR, ffmpeg, minidsp, jriver, htp1, checker, preferences, … |
| `src/main/python/acoustics/` | DSP helpers (smoothing, weighting, standards) |
| `src/main/python/mpl.py`, `svg.py`, `style/` | Matplotlib integration, SVG export, mpl styles |
| `src/test/python/` | pytest suite + fixtures |
| `beqdesigner.spec` | PyInstaller build recipe (Windows / Linux / macOS branches) |
| `.github/workflows/` | `test.yaml` (CI) and `create-app.yaml` (release builds) |
| `docs/` | Source for the readthedocs site (MkDocs) |

## Architecture

```mermaid
graph TD
    App["app.py<br/>BeqDesigner main window"]
    UI["ui/<br/>Qt Designer forms"]

    subgraph CoreModels["model/ — core logic"]
        Filter["filter<br/>CompleteFilter"]
        Signal["signal<br/>audio pipeline"]
        IIR["iir<br/>biquad DSP"]
        Codec["codec<br/>JSON I/O"]
        Analysis["analysis<br/>spectrum"]
        Prefs["preferences<br/>QSettings"]
    end

    subgraph Integrations["Device / app integrations"]
        JRiver["jriver/<br/>JRiver MC"]
        MiniDSP["minidsp<br/>device XML"]
        HTP1["sync<br/>HTP-1 WebSocket"]
        FFmpeg["ffmpeg<br/>extract / remux"]
    end

    App --> UI
    App --> CoreModels
    App --> Prefs
    UI --> CoreModels
    Filter --> IIR
    Signal --> IIR
    Analysis --> Signal
    Filter --> Codec
    JRiver --> Filter
    MiniDSP --> Filter
    HTP1 --> Filter
    FFmpeg --> Signal

    App --> Qt["PyQt6 / qtpy"]
    IIR --> NumPy["numpy / scipy"]
    Analysis --> MPL["matplotlib / pyqtgraph"]
    FFmpeg --> FFbin["ffmpeg binary"]
```

See [`docs/architecture.md`](docs/architecture.md) for a longer walkthrough of
each subsystem.

## Working with Qt Designer files

Forms are edited as `.ui` files in Qt Designer and compiled to Python with
`pyuic6`:

```sh
cd src/main/python/ui
poetry run pyuic6 foo.ui -o foo.py
```

To regenerate every form at once:

```sh
poetry run ui-gen
# or, equivalently:
poetry run python build/regen_ui.py
```

(`src/main/python/ui/convert.sh` and `convert.bat` are kept for reference but
have hard-coded paths to a contributor's venv — prefer `poetry run ui-gen`.)

Generated `.py` files **are** checked in — regenerate them whenever the `.ui`
changes. Do not hand-edit the generated files. To catch this automatically:

```sh
poetry run pip install pre-commit
pre-commit install
```

Once installed, staging a modified `.ui` triggers the hook (defined in
`.pre-commit-config.yaml`), which runs `ui-gen` and re-stages the matching
`.py`. A CI job (`ui-sync-check` in `.github/workflows/test.yaml`) performs the
same check on every push and fails if the committed `.py` files drifted from
their `.ui` sources.

## Conventions worth knowing

- `qtpy` is used as the Qt abstraction layer but the environment is pinned to
  PyQt6 at the top of `app.py`. Import from `qtpy.*`, not `PyQt6.*`, in new code.
- Logging goes through an in-memory `RollingLogger` (`model/log.py`) surfaced
  via *Help → Logs* in the UI — there is no on-disk log file, so run from a
  terminal to see tracebacks.
- User settings are stored via `QSettings` under
  `~/Library/Preferences/com.3ll3d00d.beqdesigner.plist` (macOS) /
  registry (Windows) / `~/.config/3ll3d00d/beqdesigner.conf` (Linux).
- Version string is read from `src/main/python/VERSION` — CI writes the short
  git SHA into this file at build time; running from source without it falls
  back to `0.0.0-alpha.1`.

## Auto-BEQ (research)

Research module exploring automated BEQ filter generation from measured
LFE audio - "take the human out of BEQ-making".

### Quick start (Docker, recommended for NAS / servers)

The LFE extractor — the slow, library-scanning step — ships as a small
Docker image (Python 3.13 slim + ffmpeg + a handful of stdlib-only
modules). Use this on your NAS to extract WAVs without installing
Python or scipy/PyQt deps.

```sh
# 1. Build the image locally (from repo root)
docker build -f docker/Dockerfile -t beq-extract .

# 2. Push it to your NAS over SSH
docker save beq-extract | ssh nas docker load

# 3. Copy the example compose config and edit volume paths to match
#    your NAS layout (media roots, beq working directory).
scp docker/docker-compose.example.yml nas:/path/to/beqdesigner/docker-compose.yml
ssh nas
nano /path/to/beqdesigner/docker-compose.yml  # edit volume mounts

# 4. Run extraction (resumes from cache, breadth-first ordering)
ssh nas
cd /path/to/beqdesigner
docker compose run beq-lfe-extract

# 5. Verify cache integrity later
docker compose run beq-wav-verify
```

Service names are prefixed with `beq-` so they don't collide with any
other Docker services on the same host.

After code changes, rebuild and redeploy:
```sh
docker build -f docker/Dockerfile -t beq-extract .
docker save beq-extract | ssh nas docker load
# No need to re-copy the compose config — the image change is enough.
```

**Outputs (written to `{beq-dir}/` on the NAS):**

| File | Purpose |
|---|---|
| `wav-cache/...` | Extracted LFE WAVs (the training set) |
| `beq_catalogue.json` | Local cache of the BEQ GitHub catalogue |
| `media_inventory.json` | Every media file with a `[tmdb-NNN]`/`[tvdb-NNN]` tag (matched + unmatched). Used to dedupe acquisition recommendations. |
| `missing_ids.txt` | Plain-text list of media files **without** DB ID tags. Renaming them to add the correct tag would let the model learn from them. |

**Pull results back to dev machine** for analysis:
```sh
scp nas:/path/to/beqdesigner/{media_inventory.json,beq_catalogue.json} \
    ~/Downloads/beqdesigner/

# Then locally generate bias and acquisition reports:
bin/beq-designer report cache-bias -o cache_bias.md
bin/beq-designer report acquisitions -n 50 \
    -o recommendations.md
```

### Quick start (local Python, for dev / single machine)

If you don't have a NAS or want to run everything on one machine:

**Prerequisites:** Python 3.13, Poetry, ffmpeg + ffprobe on PATH.

```sh
# 1. Install deps
poetry install

# 2. Configure your audio cache dir + library roots
#    Edit ~/.config/beqdesigner/settings.json:
#    {
#      "shared_beq_dir": "/path/to/beqdesigner"
#    }
#    Edit ~/.config/beqdesigner/extract_config.json:
#    {
#      "media_roots": ["/path/to/your/Movies", "/path/to/your/TV"]
#    }

# 3. Discover your media and match against the BEQ catalogue
bin/beq-designer discover

# 4. Run the auto-BEQ pipeline across discovered media
#    (extracts LFE audio, proposes filters, grades against catalogue)
bin/beq-designer dev benchmark
```

Step 3 walks your library roots, finds media files, matches titles
against the BEQ catalogue (~14,700 entries), and saves the results.
Step 4 runs the auto-generation algorithm on each matched file and
compares the output to the catalogue's expert-authored filters.

### CLI commands

| Command | Purpose |
|---|---|
| `bin/beq-designer` | Interactive menu (no args) |
| `bin/beq-designer profile` | Generate BEQ profiles |
| `bin/beq-designer extract` | Extract LFE WAVs to cache |
| `bin/beq-designer cache-status` | Cache summary |
| `bin/beq-designer verify` | Cache integrity check |
| `bin/beq-designer nn-report` | NN vs catalogue comparison |
| `bin/beq-designer report cache-bias` | Cache vs catalogue bias |
| `bin/beq-designer report acquisitions` | Title acquisition recommendations |
| `bin/beq-designer dev train` | Train production model |
| `bin/beq-designer dev test` | Run auto_beq tests |
| `bin/beq-designer discover` | Discover media in library |
| `bin/beq-designer config` | Edit CLI preferences |
| `bin/beq-designer report experiments` | Compare experiment results |
| `bin/beq-designer --help` | Full subcommand list |


### Environment variables

| Var | Purpose | Default |
|---|---|---|
| `AUTO_BEQ_ADVISOR` | Which advisor: `measurement` (signal-only), `ollama` (LLM), `mock`, `heuristic` | `measurement` |
| `AUTO_BEQ_SWEEP_LIMIT` | Max media files to process in benchmark | `10` |
| `OLLAMA_MODEL` | Ollama model name | `qwen:14b` |
| `BEQ_SHARED_DIR` | BEQ shared working directory (WAV cache, models) | (from settings.json) |
| `BEQ_WAV_CACHE` | Override WAV cache directory | (derived from shared dir) |
| `AUTO_BEQ_MODEL_PATH` | Override path for the trained XGBoost model loaded by `AUTO_BEQ_ADVISOR=trained_model`. Falls back to `{beq-dir}/production_model.joblib`. | (auto-discover) |

### Design docs

- [`docs/design/auto_beq.md`](docs/design/auto_beq.md) - auto-BEQ architecture and reference
- [`docs/design/experiments/`](docs/design/experiments/) - experiment archive (E1-E87)

## Further reading

- User guide, workflows and UI reference: <https://beqdesigner.readthedocs.io/>
- Concepts (what BEQ is, pre- vs post-bass-management): `docs/index.md`,
  `docs/concepts.md`, `docs/workflow/`
- Install instructions for release binaries: <https://beqdesigner.readthedocs.io/en/latest/install/>
- Release/download page: <https://github.com/3ll3d00d/beqdesigner/releases>
