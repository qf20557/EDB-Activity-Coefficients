"""A time / radius trace with a keep mask, worked on block by block so a multi-GB trace is never copied whole."""
import numpy as np

from . import config


class Cancelled(Exception):
    """Raised inside a long read or write when the user cancels it."""


def _is_sorted(t):
    rows = config.CHUNK_ROWS
    for a in range(0, len(t) - 1, rows):
        block = t[a:a + rows + 1]
        if np.any(block[1:] < block[:-1]):
            return False
    return True


def _extremes(r, keep, bucket, first):
    """Row numbers of the lowest and highest retained radius in each run of `bucket` rows; r starts at row `first`."""
    if keep.all():
        low = high = r
    else:
        low, high = np.where(keep, r, np.inf), np.where(keep, r, -np.inf)
    rows = len(r) // bucket
    full = rows * bucket
    found = []
    if rows:
        base = first + np.arange(rows) * bucket
        found += [base + low[:full].reshape(rows, bucket).argmin(1), base + high[:full].reshape(rows, bucket).argmax(1)]
    if full < len(r):
        found.append(np.array([first + full + low[full:].argmin(), first + full + high[full:].argmax()]))
    return found


class Trace:
    """t and r are contiguous float arrays (time in s, radius in um); keep marks the retained rows."""

    def __init__(self, t, r, keep=None, is_sorted=None):
        self.t, self.r = t, r
        self.keep = np.ones(len(t), dtype=bool) if keep is None else keep
        self.count = int(np.count_nonzero(self.keep))
        self.sorted = _is_sorted(t) if is_sorted is None else is_sorted
        self.version = 0  # changes whenever keep changes

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
        lows, highs = zip(*((t.min(), t.max()) for t, _ in self.blocks()))
        return float(min(lows)), float(max(highs))

    def snapshot(self):
        """A trace that shares the data but not the keep mask, so later culls here do not change it."""
        return Trace(self.t, self.r, self.keep.copy(), self.sorted)

    def display(self, lo=None, hi=None, stride=None, budget=None):
        """(t, r) to draw. With a stride: every stride-th row of the whole trace. Without (auto): the rows with
        lo <= t <= hi, all of them if they fit the budget; otherwise half the budget is an even sample and half is the
        lowest and highest radius of each run of rows, so no spike is hidden however far the trace is thinned."""
        if stride is not None:
            t, r, keep = self.t[::stride], self.r[::stride], self.keep[::stride]
            return (t, r) if keep.all() else (t[keep], r[keep])
        budget = budget or config.MAX_PLOT_POINTS
        start, stop = self.window(lo, hi)
        if stop - start <= budget:
            rows = np.arange(start, stop)
        else:
            bucket = -(-(stop - start) // (budget // 4))
            step = bucket * max(1, config.CHUNK_ROWS // bucket)
            found = [np.arange(start, stop, max(1, (stop - start) // (budget // 2)))]
            for a in range(start, stop, step):
                b = min(a + step, stop)
                found += _extremes(self.r[a:b], self.keep[a:b], bucket, a)
            rows = np.unique(np.concatenate(found))
        rows = rows[self.keep[rows]]
        return self.t[rows], self.r[rows]
