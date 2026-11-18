"""A time / radius trace with a keep mask, worked on block by block so a multi-GB trace is never copied whole."""
import numpy as np

from . import config
from .fast import is_sorted_fast, value_bounds_fast


class Cancelled(Exception):
    """Raised inside a long read or write when the user cancels it."""


class Trace:
    """t and r are contiguous float arrays (time in s, radius in um); keep marks the retained rows."""

    def __init__(self, t, r, keep=None, is_sorted=None):
        self.t, self.r = t, r
        self.keep = np.ones(len(t), dtype=bool) if keep is None else keep
        self.count = int(np.count_nonzero(self.keep))
        self.sorted = is_sorted_fast(t) if is_sorted is None else is_sorted
        self.version = 0  # changes whenever keep changes
        self._bounds_cache = {}

    def __len__(self):
        return len(self.t)

    def window(self, lo=None, hi=None):
        """(start, stop) rows that hold every time in [lo, hi]. The whole trace if the times are not sorted."""
        n = len(self.t)
        if not self.sorted:
            return 0, n
        start = 0 if lo is None else int(np.searchsorted(self.t, lo, "left"))
        stop = n if hi is None else int(np.searchsorted(self.t, hi, "right"))
        return start, max(start, stop)

    def blocks(self, lo=None, hi=None, rows=None):
        """Yield (t, r) for the retained rows with lo <= t <= hi, in order, at most `rows` at a time."""
        start, stop = self.window(lo, hi)
        rows = rows or config.CHUNK_ROWS
        for a in range(start, stop, rows):
            b = min(a + rows, stop)
            t, r, mask = self.t[a:b], self.r[a:b], self.keep[a:b]
            if lo is not None:
                mask = mask & (t >= lo)
            if hi is not None:
                mask = mask & (t <= hi)
            if mask.all():
                yield t, r
            elif mask.any():
                yield t[mask], r[mask]

    def active(self):
        """All retained rows. No copy is made while nothing is culled."""
        if self.count == len(self.t):
            return self.t, self.r
        return self.t[self.keep], self.r[self.keep]

    def first(self):
        """(time, radius) of the first retained row."""
        i = int(self.keep.argmax())
        return float(self.t[i]), float(self.r[i])

    def time_range(self):
        """(lowest, highest) retained time."""
        if self.sorted:
            a, b = int(self.keep.argmax()), len(self.keep) - 1 - int(self.keep[::-1].argmax())
            return float(self.t[a]), float(self.t[b])
        _, t_lo, t_hi, _, _ = self.bounds()
        return t_lo, t_hi

    def bounds(self, lo=None, hi=None):
        """(count, t_min, t_max, r_min, r_max) of retained rows in [lo, hi], cached per keep-mask version."""
        key = (self.version, None if lo is None else float(lo), None if hi is None else float(hi))
        cached = self._bounds_cache.get(key)
        if cached is not None:
            return cached
        start, stop = self.window(lo, hi)
        res = value_bounds_fast(self.t, self.r, self.keep, start, stop, lo, hi)
        if len(self._bounds_cache) > 16:
            self._bounds_cache.clear()
        self._bounds_cache[key] = res
        return res

    def snapshot(self):
        """A trace that shares the data but not the keep mask, so later culls here do not change it."""
        return Trace(self.t, self.r, self.keep.copy(), self.sorted)
