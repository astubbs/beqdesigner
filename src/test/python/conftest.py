import logging
import shutil

import pytest


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
