"""Backward-compatibility stub. Import from model.audio_extraction,
model.wav_discovery, or model.training_data instead."""
from model.audio_extraction import *  # noqa: F401,F403
from model.wav_discovery import *  # noqa: F401,F403
from model.training_data import *  # noqa: F401,F403

# Backwards-compatible aliases (the existing test_auto_beq.py uses
# underscore-prefixed names).
from model.audio_extraction import have_tool as _have_tool  # noqa: F401
from model.audio_extraction import probe_audio_stream as _probe_audio_stream  # noqa: F401
from model.audio_extraction import extract_lfe_wav as _extract_lfe_wav  # noqa: F401
from model.audio_extraction import load_and_smooth_chunked as _load_and_smooth_chunked  # noqa: F401
from model.audio_extraction import _strategy_from_env  # noqa: F401

# Re-export _SETTINGS_PATH for tests that monkeypatch it.
from model.wav_discovery import _SETTINGS_PATH  # noqa: F401
