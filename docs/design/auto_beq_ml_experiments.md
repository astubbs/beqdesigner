# Automated BEQ — ML Model Experiments

## Note on Terminology

These experiments use a **trained regression model** (neural network), not an LLM. The distinction matters: an LLM generates text; this model takes audio features as input and outputs filter parameters as numbers. No API costs, runs locally. The BEQ catalogue is the labelled training dataset.

Modules from the existing experiment log (auto_beq_experiments.md, E1-E17) are reused directly — specifically the headless STFT extraction pipeline, chunk processing, and benchmark harness. Nothing is duplicated here.

---

## Dataset

The BEQ catalogue contains approximately **15,000 entries** across 8 contributors. This is a serious dataset — not small ML territory. Large enough to:

- Train a CNN or small transformer without overfitting risk
- Use a richer output representation (variable filter stages rather than padded zeros)
- Hold out a statistically robust test set of 1,500+ titles
- Run full hyperparameter search in practical time on the RTX 3090

### Critical preprocessing — deduplication

Many catalogue entries are multiple format variants of the same film (BD vs UHD vs streaming, different audio codecs). These will have near-identical rolloff shapes. If duplicates straddle the train/test split, you leak near-identical curves across the boundary and get overoptimistic benchmark results.

**Before any split:** group entries by title, deduplicate to one entry per unique title (pick highest-quality format entry: Atmos > TrueHD > DTS-HD MA > DD+ > other). Effective unique training signal estimated at 6,000–10,000 unique titles after deduplication. Still excellent.

---

## Inputs

The model takes two classes of input: audio-derived features and metadata. Both are necessary. On titles with sparse bass content, metadata carries the prediction. On titles with clear rolloff signal, audio dominates. The model learns that weighting automatically.

### Audio features (X — signal-derived)

Two representations, compared as sub-experiments:

**Option A — Percentile curve (9 values)**
- 90th percentile peak curve sampled at fixed bins: 20, 25, 30, 35, 40, 50, 60, 70, 80 Hz
- Normalised relative to flat reference above 80 Hz
- Fast, simple, interpretable

**Option B — Chunk feature matrix (27 values)**
- Per frequency bin across all chunks:
  - 90th percentile (rolloff ceiling estimate)
  - Standard deviation (content variability)
  - Fraction of chunks within 3 dB of ceiling (consistency of signal near ceiling)
- 3 statistics × 9 bins = 27 values
- Captures how confidently the rolloff ceiling is visible in the data

Start with Option A. Escalate to Option B if A plateaus.

### Metadata features (M — catalogue and IMDB-derived)

Prioritised by predictive value:

**Tier 1 — High value, directly predictive of rolloff**

| Feature | Type | Dims | Source | Rationale |
|---|---|---|---|---|
| Studio / distributor | Categorical (embedding, dim 16) | 16 | TMDb/IMDB (external) | Mixing stages are per-studio. Disney, Warner, Universal each have house tendencies. **Single most predictive metadata field.** |
| Release year | Numerical (normalised) | 1 | catalogue | Rolloff practices changed over time — heavier filtering in certain eras. |
| Audio format | Categorical (one-hot, 6 buckets) | 6 | catalogue `audioTypes` | TrueHD, DTS-HD MA, Atmos, etc. Different encode chains have different headroom assumptions. |
| Release type | Categorical (one-hot) | 3 | catalogue `source` | Disc vs Streaming. UHD remasters are often re-filtered vs BD. |

**Tier 2 — Medium value, correlated but indirect**

| Feature | Type | Dims | Source | Rationale |
|---|---|---|---|---|
| Genre | Categorical (multi-hot, 10 buckets) | 10 | catalogue `genres` | Action/sci-fi tends heavier rolloff to protect theatrical chains. Drama/documentary tends lighter. |
| Country of origin | Categorical (one-hot, 5 buckets) | 5 | catalogue `language` | Korean, Japanese, European mixes have different tendencies from Hollywood. |
| Runtime | Numerical (normalised /180) | 1 | catalogue | Proxy for content type — very short = TV episode = different mixing context. |
| Supervising sound mixer | Categorical (embedding, dim 8) | 8 | IMDB (external) | Individual mixers have consistent styles. Zero-padded until data is available. |

**Tier 3 — Low value, include only if Tier 1+2 saturate**

| Feature | Type | Dims | Source | Rationale |
|---|---|---|---|---|
| Certification / rating | Label-encoded | 1 | catalogue | R-rated action weakly heavier than PG animation, but very noisy. |
| Director | Categorical (embedding) | — | IMDB (external) | Some directors insist on specific sound practices, but mediated through the mixer. Noisy. |

**Explicitly excluded:**
- Synopsis / plot description — narrative content does not predict technical mixing decisions made downstream
- Cast — no correlation
- Budget / box office — too noisy
- Awards / critical reception — no correlation

### Metadata sourcing

- Studio, audio format, release type, year, language: already in BEQ catalogue entries
- Studio: TMDb API lookup by title + year — batch job, cache locally
- Genre, runtime, certification, director, mixing engineer: IMDB lookup — automatable via IMDB datasets (public bulk download) or `cinemagoer` Python library
- Build metadata fetch as a one-time batch job keyed on title + year; cache results locally as JSON

**Combined feature vector: 60 dims** (9 audio + 51 metadata with stubs for unavailable fields)

---

## Model Architecture

Two input branches merged before output layers:

```
audio features (Option A or B)                          metadata features (Tier 1+2)
    → CNN / transformer branch                              → embedding lookups (studio, format, mixer)
    → audio embedding vector (64–128 dims)                 → concatenate with numerical features
                                                           → dense layer (32–64 units, ReLU)
                                                           → metadata embedding vector (32–64 dims)
                    ↓                                                   ↓
              audio embedding + metadata embedding
                    → concatenate
                    → 2–3 dense layers
                    → filter parameter output vector (16 dims)
```

For the XGBoost stage, audio and metadata are flattened into a single 60-dim vector. Separate branches are implemented in the CNN stage.

### Architecture progression (in order of escalation)

**Stage 1 — XGBoost with concatenated features (E18, this implementation)**
- Flatten all audio + metadata features into a single 60-dim vector
- Trains in seconds, strong baseline, highly interpretable
- Feature importance output tells you immediately which metadata fields are actually predictive
- Run locally, no GPU
- If studio/year/format don't show high importance → metadata strategy needs revisiting before escalating

**Stage 2 — 1D CNN with dual-branch metadata (E18 escalation target)**
- Audio branch: convolutional layers over 9-bin frequency curve
- Metadata branch: nn.Embedding lookups + dense
- Merged at penultimate layer
- Train on RTX 3090: 10–20 min per run
- Run hyperparameter search (kernel sizes, embedding dims, learning rate): feasible in 2–4 hours on 3090
- Add `torch` to `pyproject.toml` dev deps when starting this stage

**Stage 3 — Small transformer with dual-branch metadata (if CNN plateaus)**
- Replace CNN audio branch with transformer encoder
- Models global relationships between frequency bins — captures that rolloff at 40 Hz implies specific relationships across entire 20–80 Hz range
- At 7k+ deduplicated titles, viable
- Train on RTX 3090: 30–60 min per run

### Output representation

**Option 1 — Fixed padded vector (start here)**
- `MAX_FILTER_SLOTS = 4`, per slot: [type_int, freq_hz, gain_db, q] → 16-dim Y vector
- Filter type: LowShelf=0, HighShelf=1, PeakingEQ=2
- Simple, works for all architectures

**Option 2 — Filter count + parameters (if Option 1 plateaus)**
- Predict number of filters first, then parameters for each
- More natural, avoids penalising equivalent filters in different slot orders

---

## GPU Usage — RTX 3090

| Task | CPU | RTX 3090 |
|---|---|---|
| XGBoost / LightGBM | Seconds | Same — CPU-bound |
| 1D CNN | Several hours | 10–20 min |
| Transformer | Not practical | 30–60 min |
| Hyperparameter search (50 runs) | Days | 2–4 hours |
| Feature extraction (STFT all titles) | I/O bound | Same |

**Practical setup:** run XGBoost and feature extraction locally. Offload CNN and transformer to 3090 via SSH. PyTorch: `device = 'cuda' if torch.cuda.is_available() else 'cpu'` — one line change.

---

## Training Data Preparation

1. Deduplicate catalogue by title (avoid data leakage across BD/UHD/streaming variants)
2. Parse catalogue filter entries → Y label vectors (16-dim padded)
3. For initial synthetic exercise: derive rolloff curves from filter chains (no audio needed)
   - `rolloff = -evaluate_filter_chain(entry["filters"], DEFAULT_GRID)`
   - Extract `CurveFeatures` → build Option A feature vector
4. For full training (after STFT pipeline built):
   - Extract mono bass track via ffmpeg
   - Compute STFT → peak curve
   - Compute chunked peak curves (sliding window)
   - Compute 90th percentile curve across chunks (Option A)
   - Normalise to flat reference above 80 Hz
5. Fetch and cache metadata for all titles (TMDb/IMDB batch lookup)

**Train / validation / test split (post-deduplication):**
- 70% training / 15% validation / 15% held-out test
- Stratify by rolloff severity (heavy ≥20 dB, moderate 10–20 dB, gentle <10 dB)
- Stratify by contributor — no single author's style dominates any split
- **Test set is held-out**: never touched until final evaluation

---

## Experiment 18 — XGBoost Trained Model (initial code path)

**Hypothesis:** A regression model trained on audio features + metadata learns the mapping from rolloff shape + production context to corrective filter, replicating expert judgement without hand-coded rolloff detection logic.

**Implementation:** `src/main/python/model/auto_beq_nn.py` + `TrainedModelAdvisor`

**Training:**
- Loss: MSE on filter parameters as primary training signal
- **Downstream loss as stop condition:** apply predicted filter to audio, measure RMS error vs flat across 20–80 Hz — this is the production metric, not parameter MSE. These can diverge; always stop on downstream loss.
- Early stopping on validation downstream loss

### Key functions

| Function | Description |
|---|---|
| `build_audio_features(features)` | Option A: 9-bin percentile curve from `CurveFeatures` |
| `build_metadata_features(metadata)` | 51-dim encoded metadata vector |
| `build_feature_vector(features, metadata)` | Combined 60-dim input |
| `catalogue_entry_to_labels(entry)` | 16-dim Y label vector |
| `labels_to_filters(y)` | Decode Y vector → filter list |
| `downstream_loss(pred, target, freqs)` | Production metric (mean dB error 20–80 Hz) |
| `deduplicate_by_title(entries)` | Remove BD/UHD/streaming duplicates |
| `split_dataset(X, Y, ...)` | 70/15/15 stratified split |
| `train_xgboost(X_train, Y_train, ...)` | XGBoost multi-output regressor |
| `save_model(model, path)` | joblib serialisation |
| `load_model(path)` | joblib load |
| `TrainedModelAdvisor` | Advisor implementation wrapping trained model |

### Evaluation

Held-out test set. Metrics from E1+:
- Mean frequency response error <2 dB across 20–80 Hz
- Max error <5 dB at any bin
- Zero false boost cases

Additional breakdowns:
- Error by rolloff severity (heavy/moderate/gentle)
- Error by contributor — identifies if any author's style is systematically harder to learn
- Error by studio — validates that studio embedding is learning meaningful representations
- Ablation: audio-only vs audio+metadata — quantifies how much metadata contributes
- XGBoost feature importance — if studio/year/format not high importance, metadata strategy needs revisiting

---

## Experiment 19 — Hybrid: Trained Model + Chunked Segmentation

**Hypothesis:** Combining the trained model's pattern recognition (including metadata priors) with scipy optimiser precision produces better results than either alone. The model warm-starts the optimiser; scene descriptions clean the audio input before either runs.

### Architecture

```
Stage 1 — Data quality (scene descriptions)
  ↓ filtered, weighted chunk set
Stage 2 — Model prediction (warm start)
  audio features from filtered chunks + metadata → model → initial filter estimate + confidence
  ↓
Stage 3 — Scipy refinement
  warm-started optimiser with ±30% bounds → final filter parameters
```

### Stage details

**Stage 1 — Chunk quality filtering** (reuse Exp 3 + 6 modules when built):
- Exclude music-dominant chunks
- Weight high bass-energy chunks (impacts, explosions)
- Flag titles with insufficient high-quality chunks → confidence penalty
- If scene descriptions unavailable: skip, pass all chunks unweighted

**Stage 2 — Model warm start:**
- Run E18 model on quality-filtered audio features + full metadata
- Metadata prior is especially valuable when Stage 1 filtered out many chunks

**Stage 3 — Scipy refinement** (reuse `auto_beq.propose_filters`):
- Initialise at model prediction
- Bounds: ±30% from model prediction on each parameter
- Objective: RMS error between filtered signal and flat target across 20–80 Hz
- With 7k+ training titles and metadata priors, expect Stage 2 to be close enough that Stage 3 is a small refinement

### Confidence scoring

Combined per-title confidence score:
- Quality chunk count from Stage 1
- Model ensemble variance from Stage 2 (requires ensemble training)
- Metadata completeness (missing studio or format = lower confidence)
- Delta between Stage 2 prediction and Stage 3 output (large delta = model was uncertain)
- Optimiser convergence quality

Titles below threshold: flagged for human review in BEQDesigner. Above threshold: auto-loaded via ezBEQ.

---

## Execution Order

1. ✅ E18 initial code path — synthetic training data, XGBoost, full Advisor integration
2. ✅ TMDb metadata fetcher — batch job to populate studio + mixer fields
3. Real-audio validation — train on full catalogue, validate on available WAV files
5. XGBoost ablation: audio-only vs audio+metadata — quantifies metadata contribution
6. Escalate to 1D CNN on RTX 3090 with hyperparameter search
7. If CNN meets threshold: E19 is an enhancement, not a requirement
8. If CNN plateaus: try transformer, then diagnose failure modes
9. Build E19 hybrid once E18 architecture is finalised
10. Always evaluate on fixed held-out test set — never touch it during development

---

## TMDb metadata strategy

The BEQ catalogue carries TMDb IDs on every entry. We fetch studio, mixer,
director, and country from the TMDb API and cache locally at
`~/.config/beqdesigner/tmdb_metadata_cache.json`. Once fetched, entries
are never re-fetched (movie metadata doesn't change). First run fetches
~8k entries; subsequent runs are instant from cache.

**TODO:** Persist the TMDb-enriched catalogue as part of our git DB catalogue
output. When we produce and persist BEQ profiles, include the TMDb metadata
alongside the filter parameters. This way the enriched data propagates
automatically with the profiles — no separate metadata sync step needed,
and new users get the metadata for free without hitting TMDb independently.

---

## Infrastructure

### Dependencies
- `xgboost`, `scikit-learn` (joblib): dev deps
- `torch`: installed for CNN experiments (E28)
- TMDb API: studio/mixer metadata (key in `model/postbuilder.py`)

### WAV audio cache

**Portable cache** (`scripts/extract_lfe.py`):
- Movies: `beq-dir/wav-cache/Movies/B/Blade Runner (1982) [tmdb-78]/Blade Runner (1982) [tmdb-78].lfe-1000hz.wav`
- TV: `beq-dir/wav-cache/TV/E/86 - Eighty Six (2021) [tvdb-378609]/Season 01/86 - Eighty Six S01E02 [tvdb-378609].lfe-1000hz.wav`
- Supports `[tmdb-NNN]`, `[tvdb-NNN]`, `[imdb-NNN]` — searches filename → parent → grandparent
- **Only extracts catalogue-matched media**: auto-fetches BEQ catalogue from
  GitHub, caches at `{beq-dir}/beq_catalogue.json`, freshness-checked via
  HTTP Last-Modified (no arbitrary TTL). Catalogue loaded once into memory
  at startup — indexed by tmdb ID + title+year fallback for tvdb/imdb.
- **Dockerised**: `docker/Dockerfile` + `docker-compose.example.yml`. Build once,
  deploy to NAS via `docker save | ssh nas docker load`. No scp, no version drift.
  Imports from `model.media_constants` + `model.wav_integrity` (stdlib-only, no numpy).
- Atomic writes: `.tmp` → rename on success + `-f wav` for format, prevents corruption
- Sample rate 1000 Hz hardcoded (coupled to BEQ analysis algorithm)
- Config persisted at `{beq-dir}/.extract_config.json` — no args needed after first run
- Self-hash logged at startup for version tracking on remote servers
- Breadth-first extraction: alternates movies + TV, one episode per show per round

**Legacy cache** (`_auto_beq_helpers.py`):
- Structure: mirrors source media path under `~/Downloads/beqdesigner/audio-cache/`
- Now has atomic writes + integrity validation on extract

**Integrity** (`model/wav_integrity.py`, `scripts/verify_wav_cache.py`):
- Header check: declared frames vs actual file size (detects truncation)
- Duration check: WAV duration vs catalogue runtime (detects samples/trailers)
- Validated on every load (load_and_smooth, load_and_smooth_chunked)
- `--verify` flag on extract_lfe.py, standalone verify_wav_cache.py script
- Found and deleted 5 corrupt WAVs from existing cache (0x7FFFFFFF header bug)

### Compute
- RTX 3090 machine: SSH access, CUDA 11+, pytorch with CUDA support
- Full catalogue audio corpus (when available)

---

## Reused Modules

| Module | Source | Used in |
|---|---|---|
| `evaluate_filter_chain()` | `auto_beq.py` | Feature extraction (synthetic), E19 Stage 3 |
| `extract_curve_features()` | `auto_beq_advisor.py` | Feature extraction |
| `_clamp_advice()` | `auto_beq_advisor.py` | `TrainedModelAdvisor.advise()` |
| `propose_filters_from_measured()` | `auto_beq.py` | E19 Stage 3 |
| `compute_match_metrics()` | `auto_beq.py` | Evaluation |
| `_fetch_or_cache()` | `auto_beq_catalogue.py` | Full catalogue training data |
| Benchmark harness + metrics | E1+ | Both |
