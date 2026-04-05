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
    """Advisor output: what max_gain_db and knee_hz to use."""

    max_gain_db: float
    knee_hz: float | None = None
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
    return Advice(
        max_gain_db=max_gain,
        knee_hz=knee,
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
        return _clamp_advice(
            Advice(
                max_gain_db=float(data["max_gain_db"]),
                knee_hz=None if data.get("knee_hz") is None else float(data["knee_hz"]),
                reasoning=str(data.get("reasoning", "")),
                confidence=float(data.get("confidence", 0.5)),
            ),
            source="mock",
        )


# ---------------------------------------------------------------------------
# OllamaAdvisor - real LLM via local Ollama server
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
    "Respond ONLY with strict JSON, no prose outside the JSON object."
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
        "Respond with JSON only: "
        '{"max_gain_db": <number>, "knee_hz": <number>, '
        '"confidence": <0-1>, "reasoning": "<1-2 sentences>"}'
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

    def advise(self, metadata: MediaMetadata, features: CurveFeatures) -> Advice:
        prompt = _render_ollama_user_prompt(metadata, features)
        payload = {
            "model": self.model,
            "system": _OLLAMA_SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
        }
        url = f"{self.host.rstrip('/')}/api/generate"
        log.info(
            "Ollama advise: host=%s model=%s title=%r",
            self.host, self.model, metadata.title,
        )
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
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Ollama returned non-JSON in response: {raw_text[:200]!r}"
            ) from exc

        advice = Advice(
            max_gain_db=float(parsed["max_gain_db"]),
            knee_hz=(
                None if parsed.get("knee_hz") is None else float(parsed["knee_hz"])
            ),
            reasoning=str(parsed.get("reasoning", "")),
            confidence=float(parsed.get("confidence", 0.5)),
        )
        advice = _clamp_advice(advice, source=f"ollama:{self.model}")
        log.info(
            "Ollama advice: max_gain_db=%.1f knee_hz=%s confidence=%.2f reasoning=%r",
            advice.max_gain_db, advice.knee_hz, advice.confidence, advice.reasoning,
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
