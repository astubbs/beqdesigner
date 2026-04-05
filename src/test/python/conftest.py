import json
import logging
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOGUE_SNAPSHOT = REPO_ROOT / "src" / "test" / "resources" / "auto_beq" / "database.json"


@pytest.fixture(scope="session")
def catalogue_snapshot():
    """Pre-downloaded subset of the BEQ catalogue for offline tests.

    The snapshot is committed under src/test/resources/auto_beq/database.json
    and contains only the entries referenced by auto_beq test fixtures, so it
    stays small (tens of KB) and deterministic across runs.
    """
    if not CATALOGUE_SNAPSHOT.exists():
        pytest.skip(f"catalogue snapshot missing at {CATALOGUE_SNAPSHOT}")
    with CATALOGUE_SNAPSHOT.open() as f:
        entries = json.load(f)

    def find(title: str, filter_count: int | None = None) -> dict:
        matches = [
            e for e in entries
            if e.get("title") == title
            and (filter_count is None or len(e.get("filters", [])) == filter_count)
        ]
        if not matches:
            pytest.fail(
                f"no catalogue snapshot entry for title={title!r} "
                f"filter_count={filter_count!r}"
            )
        return matches[0]

    return find


@pytest.fixture(scope="session", autouse=True)
def logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


@pytest.fixture
def tmpdirPath(tmpdir):
    yield str(tmpdir)
    # required due to https://github.com/pytest-dev/pytest/issues/1120
    shutil.rmtree(str(tmpdir))


@pytest.fixture
def tmp_settings(tmp_path):
    """An ``ini``-backed ``QSettings`` isolated to a temp file -- safe for tests
    because it never touches the host's real user preferences."""
    from qtpy.QtCore import QSettings
    path = tmp_path / "beqdesigner_test.ini"
    settings = QSettings(str(path), QSettings.Format.IniFormat)
    yield settings
    settings.sync()


@pytest.fixture
def sample_filter():
    """A small ``CompleteFilter`` covering the filter families that appear in
    real user projects (peaking EQ, shelves, complex high-pass)."""
    from model.iir import CompleteFilter, ComplexHighPass, FilterType, LowShelf, PeakingEQ
    return CompleteFilter(
        filters=[
            PeakingEQ(1000, 50, 3.2, -5),
            LowShelf(1000, 25, 1, 3.2, count=3),
            ComplexHighPass(FilterType.BUTTERWORTH, 6, 1000, 12),
        ],
        description='sample fixture',
    )
