"""Frequency-domain data classes and Qt-dependent interpolation.

Pure data classes (MagnitudeData, ComplexData) and smoothing functions
live in model/xy_data (Qt-free). This module re-exports them for
backward compatibility and adds the Qt-dependent interp() function.
"""
import logging
import time

import numpy as np
from model.preferences import DISPLAY_SMOOTH_GRAPHS, Preferences
from qtpy.QtCore import QSettings
from scipy.interpolate import PchipInterpolator

# Re-export from xy_data so existing code continues to work.
from model.xy_data import (  # noqa: F401
    SAVGOL_WINDOW_LENGTH,
    SAVGOL_POLYORDER,
    ComplexData,
    MagnitudeData,
    must_interpolate,
    smooth,
    smooth_octave,
    smooth_savgol,
)

logger = logging.getLogger('xy')
preferences = Preferences(QSettings("3ll3d00d", "beqdesigner"))


def interp(x1, y1, x2):
    ''' Interpolates xy based on the preferred smoothing style. '''
    start = time.time()
    smooth = preferences.get(DISPLAY_SMOOTH_GRAPHS)
    if smooth:
        cs = PchipInterpolator(x1, y1)
        y2 = cs(x2)
    else:
        y2 = np.interp(x2, x1, y1)
    end = time.time()
    logger.debug(f"Interpolation from {len(x1)} to {len(x2)} in {round((end - start) * 1000, 3)}ms")
    return x2, y2