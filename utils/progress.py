"""Progress-reporting primitives shared by backend and services."""

from __future__ import annotations

from typing import Callable

ProgressCallback = Callable[[float, str], None]
"""Signature: ``callback(fraction_complete_0_to_1, human_readable_message)``."""


def noop_progress(_: float, __: str) -> None:
    """Default callback that ignores progress updates."""


class ProgressTracker:
    """Maps the 0..1 progress of a sub-task into a slice of the overall bar.

    Example:
        tracker = ProgressTracker(ui_callback)
        extract_cb = tracker.stage(0.0, 0.15)   # extraction occupies 0-15 %
        extract_cb(0.5, "Page 5/10")             # reports 7.5 % overall
    """

    def __init__(self, callback: ProgressCallback | None = None) -> None:
        self._callback = callback or noop_progress
        self._last = 0.0

    def report(self, fraction: float, message: str) -> None:
        fraction = min(1.0, max(self._last, fraction))  # never move backwards
        self._last = fraction
        self._callback(fraction, message)

    def stage(self, start: float, end: float) -> ProgressCallback:
        span = end - start

        def _callback(fraction: float, message: str) -> None:
            self.report(start + span * min(1.0, max(0.0, fraction)), message)

        return _callback
