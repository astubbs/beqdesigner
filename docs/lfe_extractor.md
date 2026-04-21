# LFE Extractor (Docker)

The LFE extractor is a standalone command-line tool that walks a media
library, extracts the LFE (low-frequency effects) channel from each
movie/TV file, and writes it to a portable WAV cache. The cache is
used by the auto-BEQ research pipeline to study how mixing engineers
shape low-frequency content across thousands of films.

The extractor ships as a small Docker image (Python 3.13 slim + ffmpeg
+ a handful of stdlib-only modules) so it can run on a NAS or remote
server **without** installing Python, scipy, PyQt or any of
BEQDesigner's GUI dependencies.

## When to use this

- You want to extract LFE WAVs from your full media library for
  analysis or contribution to the BEQ catalogue.
- You have a NAS or server where your media lives and want extraction
  to run there without copying files over the network.
- You want a self-contained, reproducible extraction pipeline that
  doesn't depend on the BEQDesigner GUI app.

If you only want to design BEQ filters in the GUI, you don't need
this — install the regular BEQDesigner binary instead.

## Prerequisites

- **On the dev machine**: Docker (for building the image)
- **On the NAS / server**: Docker, SSH access, the BEQDesigner source
  tree (or just the `docker/` and `scripts/` directories)
- A media library where files are tagged with `[tmdb-NNN]`,
  `[tvdb-NNN]`, or `[imdb-NNN]` in their filenames (the extractor uses
  this to look up each title in the BEQ catalogue)

## One-time setup

```sh
# 1. Build the image locally (from the BEQDesigner repo root)
docker build -f docker/Dockerfile -t beq-extract .

# 2. Push it to your NAS over SSH
docker save beq-extract | ssh nas docker load

# 3. Copy the example compose config to the NAS and edit volume paths
#    to match your layout (media roots and BEQ working directory).
scp docker/docker-compose.example.yml nas:/path/to/beqdesigner/docker-compose.yml
ssh nas
nano /path/to/beqdesigner/docker-compose.yml
```

The volume mounts in the compose file are the only thing you need to
edit. Map your media roots to `/media/...` paths inside the container,
and the BEQ working directory (where extracted WAVs and config will
live) to `/beq`.

## Running

```sh
ssh nas
cd /path/to/beqdesigner

# Extract LFE WAVs (resumes from cache, breadth-first ordering)
docker compose run beq-lfe-extract

# Verify the WAV cache integrity later
docker compose run beq-wav-verify
```

The service names are prefixed with `beq-` so they don't collide with
any other Docker services on the same host.

The extractor:

- Scans your media roots for `.mkv` files with a DB ID tag
- Looks up each title in the BEQ catalogue (auto-fetched from GitHub
  on first run, cached locally with HTTP `If-Modified-Since` checks)
- Extracts the LFE channel via `ffmpeg` and resamples to 1000 Hz mono
- Writes the result to a portable WAV cache under `{beq-dir}/wav-cache/`
- Skips already-extracted titles instantly
- Runs **breadth-first**: alternates movies and TV, one episode per
  show per round, so you get diverse coverage early
- Uses **atomic writes** so an interrupted run never produces a corrupt
  WAV — you can `Ctrl-C` and re-run safely

## Outputs

After each run, you'll find these files in your `{beq-dir}` on the NAS:

| File | Purpose |
|---|---|
| `wav-cache/...` | Extracted LFE WAVs, organised by Movies/TV. The training set. |
| `beq_catalogue.json` | Local cache of the BEQ GitHub catalogue. Re-fetched only when the upstream catalogue changes. |
| `media_inventory.json` | Every media file with a DB ID tag — both catalogue-matched and unmatched — with the source path. Used by the analysis tools to know what's in your library. |
| `missing_ids.txt` | Plain-text list of media files in your library that **don't** have a `[tmdb-NNN]`/`[tvdb-NNN]`/`[imdb-NNN]` tag in the filename. Renaming them to add the correct tag would let the model learn from them. |

## After code changes

Rebuild the image and push it again — the compose config doesn't
change between rebuilds:

```sh
docker build -f docker/Dockerfile -t beq-extract .
docker save beq-extract | ssh nas docker load
```

Re-run `docker compose run beq-lfe-extract` and it'll resume from the
existing cache.

## Pulling results back for analysis

The catalogue distribution analysis and acquisition recommender scripts
run on the dev machine (they need numpy, which the NAS image doesn't
have). Pull the inventory back over SSH:

```sh
scp nas:/path/to/beqdesigner/{media_inventory.json,beq_catalogue.json} \
    ~/Downloads/beqdesigner/
```

Then locally:

```sh
bin/beq-designer report cache-bias -o cache_bias.md
bin/beq-designer report acquisitions -n 50 -o recommendations.md
```

The bias report shows how your local WAV cache distribution compares
to the full BEQ catalogue across author, audio format, era and content
type. The acquisition recommender suggests which catalogue titles
you don't yet have would best improve coverage if added.

## Troubleshooting

**"Image not found" on the NAS** — make sure `docker save | ssh nas
docker load` completed without errors. Re-run the deploy step.

**"Missing volume" or "no such file"** — verify the `volumes:` paths
in `docker-compose.yml` actually exist on the NAS. Use absolute paths.

**Extraction skips everything** — your media files probably don't have
`[tmdb-NNN]` tags in their filenames. Check `missing_ids.txt` after a
run to see which files were skipped. Tagging tools like Filebot can
add the tags in bulk.

**Want to extract a specific subset only** — pass `--limit N` to the
extract command to cap the number of files processed. Useful for
testing the pipeline on a small subset before letting it loose on the
full library.
