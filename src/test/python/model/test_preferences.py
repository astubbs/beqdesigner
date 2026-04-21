from model.preferences import (
    ANALYSIS_RESOLUTION,
    ANALYSIS_RESOLUTION_DEFAULT,
    ANALYSIS_TARGET_FS,
    SCREEN_GEOMETRY,
    Preferences,
)


def test_get_returns_default_when_unset(tmp_settings):
    prefs = Preferences(tmp_settings)
    assert prefs.get(ANALYSIS_RESOLUTION) == ANALYSIS_RESOLUTION_DEFAULT
    assert prefs.get(ANALYSIS_TARGET_FS) == 1000


def test_get_without_default(tmp_settings):
    # SCREEN_GEOMETRY has no entry in DEFAULT_PREFS or TYPES, so unset+no default -> None
    prefs = Preferences(tmp_settings)
    assert prefs.get(SCREEN_GEOMETRY, default_if_unset=False) is None


def test_set_and_get_roundtrip(tmp_settings):
    prefs = Preferences(tmp_settings)
    prefs.set(ANALYSIS_TARGET_FS, 48000)
    assert prefs.get(ANALYSIS_TARGET_FS) == 48000


def test_set_respects_declared_type(tmp_settings):
    prefs = Preferences(tmp_settings)
    prefs.set(ANALYSIS_RESOLUTION, 2.5)
    # TYPES maps ANALYSIS_RESOLUTION to float -- QSettings.ini round-trips
    # everything as strings so the type declaration is what deserialises it.
    value = prefs.get(ANALYSIS_RESOLUTION)
    assert isinstance(value, float)
    assert value == 2.5


def test_has_tracks_explicit_set(tmp_settings):
    prefs = Preferences(tmp_settings)
    # an unset key with no DEFAULT_PREFS entry returns None -> has() is False
    assert prefs.has(SCREEN_GEOMETRY) is False
    prefs.set(SCREEN_GEOMETRY, b'some-geom')
    assert prefs.has(SCREEN_GEOMETRY) is True


def test_clear_removes_value(tmp_settings):
    prefs = Preferences(tmp_settings)
    prefs.set(ANALYSIS_TARGET_FS, 96000)
    prefs.clear(ANALYSIS_TARGET_FS)
    # falls back to DEFAULT_PREFS after clear
    assert prefs.get(ANALYSIS_TARGET_FS) == 1000


def test_reset_clears_all_values(tmp_settings):
    prefs = Preferences(tmp_settings)
    prefs.set(ANALYSIS_TARGET_FS, 96000)
    prefs.set(SCREEN_GEOMETRY, b'some-geom')
    prefs.reset()
    # DEFAULT_PREFS still applies...
    assert prefs.get(ANALYSIS_TARGET_FS) == 1000
    # ...but explicitly-set values without defaults are gone
    assert prefs.get(SCREEN_GEOMETRY) is None
