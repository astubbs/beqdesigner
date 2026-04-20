# FAQ and Glossary

A guide to the key concepts in BEQ Designer's automated profile
generation system. Written for developers who may not have a background
in machine learning or audio engineering.

## Glossary

### BEQ (Bass EQ)

Bass EQ correction. Movies and TV shows are often mastered with
rolled-off low-frequency content - the deep bass is intentionally
reduced compared to what was in the original mix. A BEQ profile adds
filters to restore that lost bass. The BEQ catalogue
(<https://beqcatalogue.readthedocs.io/>) is a community-maintained
database of hand-crafted correction profiles for thousands of titles.

### Advisor

The strategy pattern for generating BEQ filter proposals. Each advisor
takes audio analysis data (frequency curves, metadata) and produces a
set of correction filters. Different advisors use different approaches:

| Advisor | How it works | Needs Ollama? |
|---|---|---|
| `MeasurementAdvisor` | Heuristic rules based on audio frequency analysis | No |
| `TrainedModelAdvisor` | XGBoost model trained on real audio (E82, production default) | No |
| `OllamaAdvisor` | Asks a local LLM about the film's likely bass profile | Yes |
| `TopologyAdvisor` | Classifies the rolloff shape and applies a template | No |
| `SlopeExtensionAdvisor` | Extends the measured rolloff slope | No |

Select an advisor with the `AUTO_BEQ_ADVISOR` env var or `--advisor`
CLI flag. The production default is `trained_model` (E82 XGBoost).

### Experiment (E82, E83, etc.)

Each numbered experiment represents a different approach to generating
BEQ profiles. The number is sequential - E82 was the 82nd iteration.
Key experiments:

- **E82** - XGBoost model trained on 50:1 weighted real+synthetic
  data. Current production model. ~2.8 dB mean error.
- **E83** - Adds Whisper audio embeddings (foundation model features)
  to E82's feature set. Requires `openai-whisper` package.
- **E84** - Self-training: uses E82 to pseudo-label unmatched WAVs,
  then retrains on the expanded dataset.
- **E85** - Differentiable DSP: trains a neural network that directly
  outputs filter parameters, optimised with an acoustic loss function.
  Current champion at ~2.3 dB mean error.

### WAV cache

Extracted LFE (Low Frequency Effects) audio from media files, stored
as WAV files in `{shared_dir}/wav-cache/`. The extraction process
(`bin/beq-designer extract`) reads MKV/MP4 files, finds the LFE
channel, and saves the low-frequency content as a WAV. These WAVs are
the input to the training and evaluation pipelines.

The cache uses an ID-based directory layout
(`wav-cache/tmdb/{shard}/{id}/...`) for cross-filesystem portability.

### Shared directory (`beq_shared_dir()`)

The portable data directory containing wav-cache, catalogue, inventory,
models, and champion history. Configured via `BEQ_SHARED_DIR` env var
or `shared_beq_dir` in `~/.config/beqdesigner/settings.json`. Designed
to be a NAS mount or shared volume that moves between machines.

### Config directory (`beq_config_dir()`)

Machine-specific configuration at `~/.config/beqdesigner/`. Contains
settings.json (preferences), extract_config.json (media root paths),
and the TMDb metadata cache.

## Developer commands

### `dev reassess` vs `dev evaluate`

These answer different questions about model quality:

**`dev reassess`** (tier1 comparison) - Trains each experiment approach
(E82/E83/E84/E85) from scratch on an 80/20 held-out split and compares
them. The test set contains titles the model has never seen during
training. This measures how well each approach *generalises* to unseen
titles. Use this during research when comparing training approaches.

**`dev evaluate`** - Runs an already-trained production model on all
catalogue-matched titles and measures performance against the BEQ
catalogue's hand-written filters. Some of those titles were in the
training set. This measures real-world performance on your actual
library. Use this for quality assurance before deploying a model.

Both are needed:

- If evaluate looks great but reassess shows poor generalisation, the
  model has overfit (memorised training data instead of learning rules)
- If reassess looks great but evaluate shows bad results on specific
  titles, there are coverage gaps in the training data

### `dev benchmark`

Runs multiple advisors on the same media library side-by-side for
comparison. Useful for deciding which advisor to use for production.

### `dev test-advisor`

Tests a single advisor's filter proposal on one catalogue title
interactively. Useful for debugging why an advisor produces specific
results for a specific film.

### `dev train` / `dev train-torch`

Train the production XGBoost model (E82) or differentiable-DSP model
(E85). Outputs the model file plus a `.meta.json` sidecar with
provenance (training date, sample counts, WAV cache state).

## Advisor system - how profile generation works

When BEQ Designer generates a correction profile for a media file, it
follows this pipeline:

1. **Extract** the LFE channel from the media file as a WAV
2. **Analyse** the WAV to extract audio features (frequency response
   curves, spectral statistics, rolloff characteristics)
3. **Enrich** with metadata (TMDb genre, year, runtime, content rating)
4. **Advise** - an advisor takes the features + metadata and produces
   filter parameters (gain, frequency, Q factor for each correction
   filter)
5. **Fit** - the filter parameters are refined using scipy.optimize to
   match the target response curve

Step 4 is where advisors differ. The `TrainedModelAdvisor` (default)
feeds the 102-dimensional feature vector into an XGBoost model that
predicts all filter parameters at once. The `OllamaAdvisor` sends
film metadata to a local LLM and asks it to estimate the rolloff
characteristics.

### Which advisor should I use?

For production profile generation, use `trained_model` (the default).
It's the fastest and most accurate approach - trained on real audio
data with ground truth from the BEQ catalogue.

The other advisors exist for research and comparison. Use `dev
benchmark` to see how they compare on your library.

## Ollama setup and configuration

Ollama is only needed for the `OllamaAdvisor`. It is NOT required for
profile generation (which uses `TrainedModelAdvisor` by default) or for
reassess/training.

Configuration (checked in order):

1. `OLLAMA_HOSTS` env var - comma-separated URLs for load balancing
   (e.g. `http://server1:11434,http://server2:11434`)
2. `OLLAMA_HOST` env var - single URL (legacy)
3. `ollama_hosts` list in `~/.config/beqdesigner/settings.json`
4. Default: `http://localhost:11434`

Commands that use Ollama (`dev evaluate --advisor ollama`, `dev
benchmark`) check host availability at startup. If at least one host
responds, they proceed and warn about unreachable hosts. If no hosts
respond, they fail fast with a clear error.

## Key numbers explained

When running `dev reassess`, you'll see numbers like:

- **563 WAV-catalogue pairs** - the number of WAV files in your cache
  that match a BEQ catalogue entry (have ground truth to compare
  against)
- **448 train / 113 test** - the 80/20 split of those pairs. 80% are
  used for training, 20% held out for testing
- **Unique titles** - one episode per title is used for training to
  prevent the model memorising title-specific patterns instead of
  learning general audio correction rules
- **Mean dB error** - average difference between the model's predicted
  filters and the catalogue's hand-written filters, measured in
  decibels. Lower is better. The production model (E82) achieves ~2.8
  dB; the current champion (E85) achieves ~2.3 dB.

## Champion tracking

Each `dev reassess` run records the best-performing experiment in
`{shared_dir}/champion_history.json`. The output shows:

- Which experiment is the new champion (lowest mean dB error)
- What the previous champion was
- The improvement in dB

The history file persists across runs so you can track model
improvement over time.
