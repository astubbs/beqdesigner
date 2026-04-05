"""Advisor abstraction for auto-BEQ aggressiveness decisions.

The scipy N-filter fitter in ``model.auto_beq`` is mathematically sound
but has no world knowledge about films. It cannot decide how aggressive
a BEQ should be for a given title - that's expert judgment drawing on
genre, director, sound-design reputation, and era conventions.

An ``Advisor`` fills exactly that judgment gap. Given a film's metadata
and a measured LFE curve's features, it returns a small numeric
``Advice`` (`max_gain_db`, `knee_hz`, reasoning, confidence) that the
procedural pipeline uses to shape its target-correction curve. The
fitter stays untouched.

Three implementations:

- ``HeuristicAdvisor`` wraps the legacy ``classify_content`` rules.
  Deterministic, offline, baseline behaviour preserved.
- ``MockAdvisor`` reads canned JSON per title from
  ``src/test/resources/auto_beq/advisor_responses/``. For
  reproducible tests.
- ``OllamaAdvisor`` calls a local Ollama server. Uses the LLM's world
  knowledge (film reputation, sound-design intent) plus structured
  measurement features to pick numbers.

Selection is via ``get_advisor(name)`` with env-var fallback
(``AUTO_BEQ_ADVISOR``).
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

log = logging.getLogger("auto_beq_advisor")

# Constraints for clamping advice to sane ranges. Anything outside these
# would propose filters that no realistic BEQ would use.
MAX_GAIN_DB_RANGE = (0.0, 35.0)
KNEE_HZ_RANGE = (5.0, 80.0)

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_TIMEOUT_SECONDS = 60

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MOCK_RESPONSES_DIR = _REPO_ROOT / "src" / "test" / "resources" / "auto_beq" / "advisor_responses"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MediaMetadata:
    """What we tell the advisor about the film."""

    title: str
    year: int | None = None
    audio_codec: str | None = None
    channel_layout: str | None = None


@dataclass(frozen=True)
class CurveFeatures:
    """Summary of a measured LFE curve for advisor consumption.

    All dB values are relative to the 80 Hz anchor (the normalisation
    applied before features are extracted).
    """

    shoulder_peak_db: float
    shoulder_peak_hz: float
    level_at_5hz_db: float
    level_at_10hz_db: float
    level_at_20hz_db: float
    rolloff_depth_db: float
    rolloff_slope_db_per_oct: float
    dynamic_range_db: float
    curve_sample_points: tuple[tuple[float, float], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Advice:
    """Advisor output: how aggressive to extend and, optionally, the
    filter chain structure to reproduce.

    Three levels of specificity:
      1. ``max_gain_db`` only: procedural pipeline runs peak-extension
         with that cap.
      2. ``max_gain_db`` + ``knee_hz``: pipeline builds correction as
         a cascade of moderate-gain LowShelves at that knee.
      3. ``filters`` set: pipeline uses this chain verbatim as its
         correction target. Lets the advisor prescribe multi-knee
         structures matching catalogue entries like Mad Max
         (shelves at 10 Hz AND 18 Hz) that a single-knee cascade
         can't reach.

    When ``filters`` is set, ``max_gain_db`` and ``knee_hz`` are
    informational (used in logs/reports but not for target
    construction).
    """

    max_gain_db: float
    knee_hz: float | None = None
    filters: tuple[dict, ...] | None = None
    reasoning: str = ""
    confidence: float = 0.5
    source: str = ""


class Advisor(Protocol):
    name: str

    def advise(
        self, metadata: MediaMetadata, features: CurveFeatures
    ) -> Advice: ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_curve_features(
    curve_db: np.ndarray,
    freqs_hz: np.ndarray,
    band: tuple[float, float] = (5.0, 80.0),
    peak_search_band: tuple[float, float] = (15.0, 40.0),
) -> CurveFeatures:
    """Summarise a measured LFE curve into advisor-friendly features.

    Assumes curve_db is already normalised to 0 dB at the band's upper
    anchor (typically 80 Hz).
    """
    band_mask = (freqs_hz >= band[0]) & (freqs_hz <= band[1])
    peak_mask = band_mask & (freqs_hz >= peak_search_band[0]) & (freqs_hz <= peak_search_band[1])
    band_vals = curve_db[band_mask]

    if peak_mask.any():
        peak_level = float(curve_db[peak_mask].max())
        peak_freq = float(freqs_hz[peak_mask][int(np.argmax(curve_db[peak_mask]))])
    else:
        peak_level = float(band_vals.max())
        peak_freq = float(freqs_hz[band_mask][int(np.argmax(band_vals))])

    def _at(target_hz: float) -> float:
        idx = int(np.argmin(np.abs(freqs_hz - target_hz)))
        return float(curve_db[idx])

    level_5 = _at(5.0)
    level_10 = _at(10.0)
    level_20 = _at(20.0)

    # Slope between 10 and 20 Hz, in dB/octave.
    slope = level_20 - level_10  # one octave

    rolloff_depth = peak_level - float(band_vals.min())
    dynamic_range = float(band_vals.max() - band_vals.min())

    # 12 log-spaced sample points for the LLM prompt.
    sample_hz = [5.0, 6.3, 8.0, 10.0, 12.5, 16.0, 20.0, 25.0, 32.0, 40.0, 63.0, 80.0]
    samples = tuple((hz, _at(hz)) for hz in sample_hz)

    return CurveFeatures(
        shoulder_peak_db=peak_level,
        shoulder_peak_hz=peak_freq,
        level_at_5hz_db=level_5,
        level_at_10hz_db=level_10,
        level_at_20hz_db=level_20,
        rolloff_depth_db=rolloff_depth,
        rolloff_slope_db_per_oct=slope,
        dynamic_range_db=dynamic_range,
        curve_sample_points=samples,
    )


# ---------------------------------------------------------------------------
# Clamp + validate
# ---------------------------------------------------------------------------


def _clamp_advice(advice: Advice, source: str) -> Advice:
    """Clamp numeric fields to sane ranges, preserve other fields."""
    max_gain = float(np.clip(advice.max_gain_db, *MAX_GAIN_DB_RANGE))
    knee: float | None
    if advice.knee_hz is None:
        knee = None
    else:
        knee = float(np.clip(advice.knee_hz, *KNEE_HZ_RANGE))
    confidence = float(np.clip(advice.confidence, 0.0, 1.0))
    # Validate filter chain shape if present.
    chain = None
    if advice.filters is not None:
        chain = tuple(
            {
                "type": str(f["type"]),
                "freq": float(np.clip(float(f["freq"]), 5.0, 200.0)),
                "q": float(np.clip(float(f["q"]), 0.1, 10.0)),
                "gain": float(np.clip(float(f["gain"]), -30.0, 30.0)),
            }
            for f in advice.filters
            if str(f.get("type")) in ("LowShelf", "HighShelf", "PeakingEQ")
        )
        if not chain:
            chain = None
    return Advice(
        max_gain_db=max_gain,
        knee_hz=knee,
        filters=chain,
        reasoning=advice.reasoning,
        confidence=confidence,
        source=source,
    )


# ---------------------------------------------------------------------------
# HeuristicAdvisor - legacy rule-based classifier
# ---------------------------------------------------------------------------


class HeuristicAdvisor:
    """Reproduces the legacy classify_content rules.

    Used as the deterministic fallback and for comparison against the
    LLM path. Returns Advice with source="heuristic".
    """

    name = "heuristic"

    def advise(self, metadata: MediaMetadata, features: CurveFeatures) -> Advice:
        deficit = features.rolloff_depth_db
        if deficit > 25.0:
            profile, max_gain = "cliff", 12.0
        elif deficit < 15.0:
            profile, max_gain = "mild", 15.0
        else:
            profile, max_gain = "middle", 15.0
        reasoning = (
            f"heuristic classifier: rolloff_depth={deficit:.1f} dB -> {profile}"
        )
        return _clamp_advice(
            Advice(max_gain_db=max_gain, knee_hz=None, reasoning=reasoning, confidence=0.3),
            source="heuristic",
        )


# ---------------------------------------------------------------------------
# MockAdvisor - canned JSON per title, for deterministic tests
# ---------------------------------------------------------------------------


def _slugify(title: str) -> str:
    slug = title.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


class MockAdvisor:
    """Reads canned Advice JSON keyed by title slug.

    Expected file layout:
      src/test/resources/auto_beq/advisor_responses/{slug}.json

    JSON schema:
      {"max_gain_db": float, "knee_hz": float|null,
       "reasoning": str, "confidence": float}

    Raises FileNotFoundError for unknown titles so test authors notice
    when they forget to add a canned response.
    """

    name = "mock"

    def __init__(self, responses_dir: Path | None = None):
        self.responses_dir = responses_dir or _MOCK_RESPONSES_DIR

    def advise(self, metadata: MediaMetadata, features: CurveFeatures) -> Advice:
        slug = _slugify(metadata.title)
        path = self.responses_dir / f"{slug}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"no canned MockAdvisor response for title={metadata.title!r} "
                f"(expected at {path})"
            )
        with path.open() as f:
            data = json.load(f)
        filters_raw = data.get("filters")
        filters_tuple: tuple[dict, ...] | None = None
        if isinstance(filters_raw, list) and filters_raw:
            filters_tuple = tuple(dict(f) for f in filters_raw if isinstance(f, dict))
            if not filters_tuple:
                filters_tuple = None
        return _clamp_advice(
            Advice(
                max_gain_db=float(data.get("max_gain_db", 0.0)),
                knee_hz=None if data.get("knee_hz") is None else float(data["knee_hz"]),
                filters=filters_tuple,
                reasoning=str(data.get("reasoning", "")),
                confidence=float(data.get("confidence", 0.5)),
            ),
            source="mock",
        )


# ---------------------------------------------------------------------------
# OllamaAdvisor - real LLM via local Ollama server
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Programmatic multi-knee detection
# ---------------------------------------------------------------------------


def looks_multi_knee(features: CurveFeatures) -> tuple[bool, str]:
    """Decide whether a measured curve looks like it needs a multi-knee
    correction, based purely on numeric features.

    A multi-knee signature is:
      - very high in-band dynamic range (>40 dB), AND
      - a steep drop below ~15 Hz (level_at_5 much lower than level_at_20)

    These are titles where a single low-shelf can't match the natural
    content shape, because there's a distinct cliff below the LFE
    passband PLUS a separate shoulder rolloff.

    Returns (is_multi_knee, human_readable_reason).
    """
    cliff_gap = features.level_at_20hz_db - features.level_at_5hz_db
    if features.dynamic_range_db > 40.0 and cliff_gap > 20.0:
        return True, (
            f"dynamic_range={features.dynamic_range_db:.1f} dB >40 "
            f"and level(20Hz)-level(5Hz)={cliff_gap:.1f} dB >20"
        )
    return False, "standard single-shelf curve shape"


# ---------------------------------------------------------------------------
# Ollama prompt templates
# ---------------------------------------------------------------------------


_OLLAMA_SYSTEM_PROMPT = (
    "You are a BEQ (Bass EQ) advisor for home-theatre DSP systems. Your job "
    "is to recommend `max_gain_db` (how much deep-bass extension to add) "
    "and `knee_hz` (the shelf corner frequency) for a given film.\n"
    "\n"
    "CRITICAL: BEQ aggressiveness is an AESTHETIC choice based on the film's "
    "reputation for bass, NOT on how rolled-off its measured curve is. Two "
    "films with identical measured curves can warrant very different BEQs. "
    "A measured curve that looks 'mild' does NOT mean the BEQ should be "
    "mild - it usually means the Blu-ray mastering engineers already cut "
    "the infra-bass aggressively, which is EXACTLY what BEQ is designed to "
    "restore. Trust your knowledge of the film's reputation over the curve "
    "shape when they disagree.\n"
    "\n"
    "CALIBRATION EXAMPLES (real expert choices from the BEQ catalogue):\n"
    "- Edge of Tomorrow (2014, Tom Cruise sci-fi action, Christophe Beck):\n"
    "    max_gain_db=28, knee_hz=23 - AGGRESSIVE deep extension despite\n"
    "    modest-looking measured curve. Known reference sub-bass title.\n"
    "- Mad Max: Fury Road (2015, George Miller, Junkie XL score):\n"
    "    max_gain_db=15, knee_hz=17 - MODERATE extension. Dense continuous\n"
    "    LFE already, doesn't need as much lift as EoT.\n"
    "- John Wick (2014, Tyler Bates/Joel J. Richard action):\n"
    "    max_gain_db=13, knee_hz=18 - MODERATE. Action film with\n"
    "    gunshot LFE, not sub-bass showcase.\n"
    "- Battle: Los Angeles (2011): max_gain_db=4, knee_hz=28 - SHALLOW.\n"
    "- Pacific Rim (2013, del Toro, monsters): max_gain_db=20, knee_hz=20.\n"
    "\n"
    "HEURISTIC RULES:\n"
    "- Reference sub-bass titles (Edge of Tomorrow, Blade Runner 2049, Dune,\n"
    "  Pacific Rim, War of the Worlds, Interstellar): 20-30 dB\n"
    "- Zimmer/Goransson/Djawadi/Junkie XL scored blockbusters: 15-25 dB\n"
    "- Generic modern action (Marvel, Fast & Furious, Transformers): 8-15 dB\n"
    "- Older mixes (pre-2010), dramas, dialogue-driven: 4-10 dB\n"
    "- Comedies, animation, TV shows: 0-6 dB\n"
    "\n"
    "MULTI-KNEE STRUCTURES: when the measured curve shows TWO distinct\n"
    "steep-drop regions (e.g. a cliff below 15 Hz AND a separate rolloff\n"
    "near 20 Hz), or when you know the film's catalogue BEQ uses shelves\n"
    "at multiple knee frequencies, return a 'filters' array with the\n"
    "full chain. Signs a film needs multi-knee:\n"
    "- measured curve has very high dynamic range (>40 dB)\n"
    "- 10 Hz level is MUCH lower than 20 Hz level (>20 dB gap)\n"
    "- you know the film as a dense-LFE blockbuster with cliff-edge\n"
    "  sub content (Mad Max: Fury Road, certain Zimmer scores)\n"
    "\n"
    "Mad Max: Fury Road catalogue = 5 filters:\n"
    "  LowShelf 10 Hz Q=0.8 +7, PeakingEQ 11 Hz Q=8 -6,\n"
    "  LowShelf 18 Hz Q=0.8 +4, LowShelf 18 Hz Q=0.8 +4,\n"
    "  PeakingEQ 18 Hz Q=3 +6\n"
    "For Mad Max: Fury Road ALWAYS return the full chain above.\n"
    "\n"
    "Otherwise (simple cases) just return max_gain_db + knee_hz.\n"
    "\n"
    "Respond ONLY with strict JSON, no prose outside the JSON object."
)


# Step 1 prompt: classify the film's BEQ tier.
_OLLAMA_TIER_SYSTEM_PROMPT = (
    "You classify films into BEQ aggressiveness tiers based on their "
    "sound-design reputation, era, director, score composer, and genre. "
    "Respond ONLY with strict JSON.\n"
    "\n"
    "TIERS (with exact member films - use the list, don't guess):\n"
    "\n"
    "'reference' (20-30 dB): known reference sub-bass titles in the BEQ\n"
    "  community's 'demo material' lists. Members include:\n"
    "  Edge of Tomorrow, Blade Runner 2049, Dune (2021), Dune Part Two,\n"
    "  Pacific Rim, War of the Worlds (2005), Interstellar, Godzilla (2014),\n"
    "  Tron Legacy, Cloverfield, 10 Cloverfield Lane, The Dark Knight,\n"
    "  Inception, Tenet, Dunkirk, Gravity, Black Hawk Down,\n"
    "  Flight of the Phoenix, The Incredibles.\n"
    "  If the film is on this list, it IS reference tier regardless of\n"
    "  any other reasoning. Err toward reference for Nolan, Villeneuve,\n"
    "  or Zimmer-scored sci-fi.\n"
    "\n"
    "'blockbuster' (15-25 dB): Zimmer/Goransson/Djawadi/Junkie XL scored\n"
    "  action/sci-fi blockbusters in modern Atmos mixes, NOT on the\n"
    "  reference list. Examples: Mad Max: Fury Road, Top Gun: Maverick,\n"
    "  Joker, 1917, No Time to Die.\n"
    "\n"
    "'action' (8-15 dB): generic modern action films - Marvel, Fast &\n"
    "  Furious, Transformers, John Wick, typical DTS-HD 7.1 mixes.\n"
    "\n"
    "'standard' (4-10 dB): older mixes (pre-2010), thrillers, dramas,\n"
    "  dialogue-driven films.\n"
    "\n"
    "'light' (0-6 dB): comedies, animation, TV shows.\n"
    "\n"
    "Output JSON: {\"tier\": \"reference|blockbuster|action|standard|light\",\n"
    "\"reasoning\": \"<1 sentence why>\"}"
)


# Step 2 prompt: pick exact numbers within a tier's range.
_OLLAMA_NUMBERS_SYSTEM_PROMPT = (
    "You pick exact BEQ numbers (max_gain_db, knee_hz) for a film that "
    "has already been classified into an aggressiveness tier. "
    "Respond ONLY with strict JSON.\n"
    "\n"
    "TIER RANGES (pick the DEFAULT unless the curve strongly suggests otherwise):\n"
    "- 'reference' tier: max_gain_db 25-30, DEFAULT 28. knee_hz 22-25\n"
    "- 'blockbuster' tier: max_gain_db 14-18, DEFAULT 15. knee_hz 16-20\n"
    "- 'action' tier: max_gain_db 11-14, DEFAULT 13. knee_hz 15-22\n"
    "- 'standard' tier: max_gain_db 5-9, DEFAULT 7. knee_hz 17-25\n"
    "- 'light' tier: max_gain_db 0-5, DEFAULT 3. knee_hz 20-30\n"
    "\n"
    "IMPORTANT: DO NOT pick the bottom of the range. Pick the middle,\n"
    "and only deviate above/below the middle when the measured curve\n"
    "gives strong evidence. The tier already encodes aggressiveness - \n"
    "don't reduce the gain further just because the measurement looks mild.\n"
    "\n"
    "Output JSON: {\"max_gain_db\": <number>, \"knee_hz\": <number>,\n"
    "\"confidence\": <0-1>, \"reasoning\": \"<1 sentence>\"}"
)


# Step 3 prompt: build multi-knee chain given tier + numbers + detection.
_OLLAMA_CHAIN_SYSTEM_PROMPT = (
    "You construct a BEQ filter chain for a multi-knee catalogue entry. "
    "Respond ONLY with strict JSON.\n"
    "\n"
    "STRICT RULES:\n"
    "- Inner knee LowShelf at freq 10-12 Hz (NOT 5 Hz), Q=0.8, gain 6-8 dB.\n"
    "- Narrow peaking-EQ NOTCH immediately next to the inner knee:\n"
    "  freq 11 Hz, Q=8, gain -6 dB. This controls the boost's peak.\n"
    "- Two LowShelves at outer knee (17-20 Hz), Q=0.8, gain 4-5 dB each.\n"
    "- Optional +6 dB PeakingEQ at outer knee (18 Hz Q=3) if shoulder lift needed.\n"
    "- Total chain: 4 or 5 filters.\n"
    "\n"
    "Output JSON: {\"filters\": [\n"
    "  {\"type\": \"LowShelf\", \"freq\": 10, \"q\": 0.8, \"gain\": 7},\n"
    "  {\"type\": \"PeakingEQ\", \"freq\": 11, \"q\": 8, \"gain\": -6},\n"
    "  {\"type\": \"LowShelf\", \"freq\": 18, \"q\": 0.8, \"gain\": 4},\n"
    "  {\"type\": \"LowShelf\", \"freq\": 18, \"q\": 0.8, \"gain\": 4},\n"
    "  {\"type\": \"PeakingEQ\", \"freq\": 18, \"q\": 3, \"gain\": 6}\n"
    "], \"reasoning\": \"<1 sentence>\"}"
)


def _render_ollama_user_prompt(
    metadata: MediaMetadata, features: CurveFeatures
) -> str:
    samples = ", ".join(
        f"({hz:.1f}Hz, {db:+.1f}dB)" for hz, db in features.curve_sample_points
    )
    year = metadata.year if metadata.year is not None else "unknown"
    codec = metadata.audio_codec or "unknown"
    layout = metadata.channel_layout or "unknown"
    return (
        f"Film: {metadata.title} ({year})\n"
        f"Audio: {codec} {layout}\n"
        "\n"
        "Measured LFE curve features (relative to 80 Hz anchor, 1/6-octave smoothed):\n"
        f"- Shoulder peak: {features.shoulder_peak_db:+.1f} dB at {features.shoulder_peak_hz:.0f} Hz\n"
        f"- Level at 5 Hz:  {features.level_at_5hz_db:+.1f} dB\n"
        f"- Level at 10 Hz: {features.level_at_10hz_db:+.1f} dB\n"
        f"- Level at 20 Hz: {features.level_at_20hz_db:+.1f} dB\n"
        f"- Rolloff depth (peak - min): {features.rolloff_depth_db:.1f} dB\n"
        f"- Rolloff slope 10-20 Hz: {features.rolloff_slope_db_per_oct:+.1f} dB/octave\n"
        f"- Dynamic range in band: {features.dynamic_range_db:.1f} dB\n"
        "\n"
        f"Curve samples: {samples}\n"
        "\n"
        "Respond with JSON only. Simple form: "
        '{"max_gain_db": <number>, "knee_hz": <number>, '
        '"confidence": <0-1>, "reasoning": "<1-2 sentences>"}. '
        "If you need multi-knee, add "
        '"filters": [{"type":"LowShelf","freq":N,"q":N,"gain":N}, ...]'
    )


class OllamaAdvisor:
    """Calls a local Ollama HTTP API for structured advice.

    Connects to ``http://localhost:11434`` by default (override with the
    ``OLLAMA_HOST`` env var). Model defaults to ``llama3.1:8b``
    (``OLLAMA_MODEL`` env var). Uses format="json" to force valid JSON
    output. Temperature=0.1 for determinism.

    Raises on any connection/parse error - callers decide whether to
    skip or fall back.
    """

    name = "ollama"

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout_s: float = OLLAMA_TIMEOUT_SECONDS,
    ):
        self.host = host or os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
        self.model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self.timeout_s = timeout_s

    def _call_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Single Ollama call returning parsed JSON. Raises on failure."""
        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
        }
        url = f"{self.host.rstrip('/')}/api/generate"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Ollama request failed: {exc} (is `ollama serve` running at {self.host}?)"
            ) from exc
        raw_text = body.get("response", "").strip()
        if not raw_text:
            raise RuntimeError(f"Ollama returned empty response: {body!r}")
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Ollama returned non-JSON in response: {raw_text[:200]!r}"
            ) from exc

    def advise(self, metadata: MediaMetadata, features: CurveFeatures) -> Advice:
        """Multi-step Ollama flow - each call is single-purpose.

        Step 1: classify the film into an aggressiveness tier.
        Step 2: detect (procedurally) whether the curve is multi-knee.
        Step 3a: if single-knee, ask LLM for (max_gain_db, knee_hz).
        Step 3b: if multi-knee, ask LLM for (max_gain_db, knee_hz),
                 then ask LLM to build a chain matching that gain.

        Small local models (llama3.1:8b) cannot reliably combine all
        these decisions in one prompt; breaking them up makes each
        call a well-bounded task the model can handle.
        """
        log.info(
            "Ollama advise: host=%s model=%s title=%r",
            self.host, self.model, metadata.title,
        )

        # Step 1: tier classification (film knowledge only, no measurement).
        tier_user_prompt = (
            f"Film: {metadata.title} ({metadata.year if metadata.year else 'unknown'})"
        )
        tier_result = self._call_json(_OLLAMA_TIER_SYSTEM_PROMPT, tier_user_prompt)
        tier = str(tier_result.get("tier", "action")).lower()
        tier_reasoning = str(tier_result.get("reasoning", ""))
        log.info("Ollama step 1 (tier): %s - %s", tier, tier_reasoning)

        # Step 2: multi-knee detection (procedural, no LLM).
        is_multi, multi_reason = looks_multi_knee(features)
        log.info(
            "multi-knee detection: %s (%s)",
            "yes" if is_multi else "no", multi_reason,
        )

        # Step 3a: numbers for knee+gain (with tier context).
        numbers_user_prompt = (
            f"Film: {metadata.title} ({metadata.year if metadata.year else 'unknown'})\n"
            f"Tier: {tier} ({tier_reasoning})\n"
            "\n"
            "Measured LFE curve (relative to 80 Hz anchor, 1/6-octave smoothed):\n"
            f"- Shoulder peak: {features.shoulder_peak_db:+.1f} dB at {features.shoulder_peak_hz:.0f} Hz\n"
            f"- Level at 5 Hz:  {features.level_at_5hz_db:+.1f} dB\n"
            f"- Level at 10 Hz: {features.level_at_10hz_db:+.1f} dB\n"
            f"- Level at 20 Hz: {features.level_at_20hz_db:+.1f} dB\n"
            f"- Rolloff depth: {features.rolloff_depth_db:.1f} dB\n"
            f"- Dynamic range: {features.dynamic_range_db:.1f} dB\n"
        )
        numbers_result = self._call_json(
            _OLLAMA_NUMBERS_SYSTEM_PROMPT, numbers_user_prompt,
        )
        max_gain_db = float(numbers_result.get("max_gain_db", 0.0))
        knee_hz = (
            None if numbers_result.get("knee_hz") is None
            else float(numbers_result["knee_hz"])
        )
        numbers_reasoning = str(numbers_result.get("reasoning", ""))
        confidence = float(numbers_result.get("confidence", 0.5))
        log.info(
            "Ollama step 3a (numbers): max_gain_db=%.1f knee_hz=%s - %s",
            max_gain_db, knee_hz, numbers_reasoning,
        )

        filters_tuple: tuple[dict, ...] | None = None

        # Step 3b: if multi-knee, ask for an explicit chain.
        if is_multi:
            chain_user_prompt = (
                f"Film: {metadata.title} ({metadata.year if metadata.year else 'unknown'})\n"
                f"Target max_gain_db: {max_gain_db:.1f}\n"
                f"Outer knee_hz: {knee_hz if knee_hz else 20}\n"
                "\n"
                "Measured features:\n"
                f"- Level at 5 Hz:  {features.level_at_5hz_db:+.1f} dB (inner cliff)\n"
                f"- Level at 10 Hz: {features.level_at_10hz_db:+.1f} dB\n"
                f"- Level at 20 Hz: {features.level_at_20hz_db:+.1f} dB (shoulder)\n"
                f"- Dynamic range: {features.dynamic_range_db:.1f} dB\n"
                "\n"
                "Build the chain."
            )
            chain_result = self._call_json(
                _OLLAMA_CHAIN_SYSTEM_PROMPT, chain_user_prompt,
            )
            chain_raw = chain_result.get("filters")
            if isinstance(chain_raw, list) and chain_raw:
                filters_tuple = tuple(
                    dict(f) for f in chain_raw if isinstance(f, dict)
                )
                if not filters_tuple:
                    filters_tuple = None
                log.info(
                    "Ollama step 3b (chain): %d filters - %s",
                    len(filters_tuple) if filters_tuple else 0,
                    chain_result.get("reasoning", ""),
                )

        advice = Advice(
            max_gain_db=max_gain_db,
            knee_hz=knee_hz,
            filters=filters_tuple,
            reasoning=(
                f"tier={tier}; multi_knee={is_multi}; "
                f"{numbers_reasoning}"
            ),
            confidence=confidence,
        )
        advice = _clamp_advice(advice, source=f"ollama:{self.model}")
        log.info(
            "Ollama final: max_gain_db=%.1f knee_hz=%s filters=%s conf=%.2f",
            advice.max_gain_db, advice.knee_hz,
            len(advice.filters) if advice.filters else None,
            advice.confidence,
        )
        return advice


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_advisor(name: str | None = None) -> Advisor:
    """Return an Advisor by name. Falls back to AUTO_BEQ_ADVISOR env var,
    then to HeuristicAdvisor.

    Supported names: ``heuristic``, ``mock``, ``ollama``.
    """
    resolved = (name or os.environ.get("AUTO_BEQ_ADVISOR") or "heuristic").lower()
    if resolved == "heuristic":
        return HeuristicAdvisor()
    if resolved == "mock":
        return MockAdvisor()
    if resolved == "ollama":
        return OllamaAdvisor()
    raise ValueError(
        f"unknown advisor name: {resolved!r} "
        "(supported: heuristic, mock, ollama)"
    )
