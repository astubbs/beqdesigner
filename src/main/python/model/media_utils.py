"""Shared media filesystem utilities.

Stdlib-only — no scipy, numpy, or PyQt dependencies. Safe to import
from standalone scripts (Docker containers, NAS extraction, etc.).
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

from model.media_constants import MEDIA_EXTENSIONS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Duration formatting
# ---------------------------------------------------------------------------


def format_duration(seconds: float) -> str:
    """Format seconds as a human-readable duration, omitting zero units.

    Examples: ``"45s"``, ``"2m 30s"``, ``"1h 23m"``, ``"1h 23m 4s"``.
    """
    s = int(seconds)
    if s < 0:
        s = 0
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    if s or not parts:
        parts.append(f"{s}s")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# ProgressLogger — time-throttled progress with ETA
# ---------------------------------------------------------------------------


class ProgressLogger:
    """Log progress lines only when enough wall-clock time has elapsed.

    Prevents log spam during fast loops while ensuring the user always
    sees progress during slow operations. ETA is computed automatically
    from elapsed time and items completed.

    Usage::

        progress = ProgressLogger(total=2400, logger=log, min_interval_s=5)
        for i, item in enumerate(items):
            do_work(item)
            progress.update(i + 1, label=item.name)
        progress.finish("scan complete")

    Only emits a log line when ``min_interval_s`` seconds have passed
    since the last log, or on the final item. The first log is always
    emitted after ``min_interval_s`` (not immediately on item 1).
    """

    def __init__(
        self,
        total: int,
        logger: logging.Logger | None = None,
        min_interval_s: float = 5.0,
        level: int = logging.INFO,
    ):
        self.total = total
        self.logger = logger or log
        self.min_interval_s = min_interval_s
        self.level = level
        self._t0 = time.time()
        self._last_log_time = self._t0  # first log after min_interval_s elapses

    def update(self, current: int, label: str = "") -> None:
        """Record progress. Logs only if enough time has passed or this is the last item."""
        now = time.time()
        is_last = current >= self.total
        if not is_last and (now - self._last_log_time) < self.min_interval_s:
            return

        self._last_log_time = now
        elapsed = now - self._t0
        pct = current * 100 // self.total if self.total > 0 else 100

        if current > 0 and elapsed > 0 and not is_last:
            eta_s = (elapsed / current) * (self.total - current)
            eta_time = datetime.now() + timedelta(seconds=eta_s)
            remaining = format_duration(eta_s)
            eta_str = f" {remaining} remaining, ETA {eta_time.strftime('%H:%M')}"
        else:
            eta_str = ""

        suffix = f"  {label}" if label else ""
        self.logger.log(
            self.level,
            "  [%d/%d %d%%%s]%s",
            current, self.total, pct, eta_str, suffix,
        )

    def finish(self, message: str = "done") -> float:
        """Log completion and return total elapsed seconds."""
        elapsed = time.time() - self._t0
        self.logger.log(
            self.level, "  %s (%s)", message, format_duration(elapsed),
        )
        return elapsed

_HAS_YEAR = re.compile(r"\(\d{4}\)")


def find_media_dirs(library_root: Path) -> list[Path]:
    """Find directories at the depth where media files live.

    Finds the first ``.mkv`` file, determines its parent's depth
    relative to the root, then lists all directories at that depth.
    This adapts to any layout (flat, one-level, genre-grouped, etc.).
    Falls back to immediate children if no media files found.

    Use for progress bars and directory counting — the returned list
    length is the denominator for "N of M directories scanned".
    """
    # Find one media file to determine the depth. Check each top-level
    # child individually so we don't traverse the entire tree on a slow
    # network volume — we stop as soon as we find the first file.
    sample: Path | None = None
    for child in library_root.iterdir():
        if child.is_dir():
            for ext in MEDIA_EXTENSIONS:
                for p in child.rglob(f"*{ext}"):
                    sample = p
                    break
                if sample:
                    break
        elif any(child.suffix == ext for ext in MEDIA_EXTENSIONS):
            sample = child
        if sample:
            break
    if sample is None:
        return sorted(p for p in library_root.iterdir() if p.is_dir())

    # Walk up from the media file's parent to find the title directory —
    # the highest ancestor (below root) whose name contains "(YEAR)".
    # For "root/Show (2023)/Season 1/file.mkv" that's "Show (2023)" at depth 1.
    # For "root/file.mkv" that's root itself.
    title_dir: Path | None = None
    cur = sample.parent
    while cur != library_root and cur != cur.parent:
        if _HAS_YEAR.search(cur.name):
            title_dir = cur
        cur = cur.parent

    if title_dir is None:
        # No dir with (YEAR) found — use immediate children of root.
        return sorted(p for p in library_root.iterdir() if p.is_dir())

    # The title dir's parent is the level we want to list.
    level_parent = title_dir.parent
    return sorted(p for p in level_parent.iterdir() if p.is_dir())
