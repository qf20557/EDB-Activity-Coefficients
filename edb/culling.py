"""Stage 1: removing points with boxes, with undo."""
import numpy as np

from .fast import cull_mask_fast
from .trace import Trace


class PointStore(Trace):
    """The loaded trace. Culling only clears rows of the keep mask; the data itself is never changed or copied."""

    def __init__(self, t, r):
        super().__init__(t, r)
        self.history = []  # per cull: the removed row numbers, or the whole previous mask (bit-packed) if smaller

    def cull_box(self, box, squared, keep_inside=False, scale=1.0):
        xmin, xmax, ymin, ymax = box
        start, stop = self.window(xmin, xmax)
        inside, hits = cull_mask_fast(self.t, self.r, self.keep, start, stop, xmin, xmax, ymin, ymax, squared, scale)
        removed = self.count - hits if keep_inside else hits
        if removed == self.count:
            raise ValueError("That would remove every point.")
        if not removed:
            return 0
        n_total = len(self.keep)
        pack = 64 * removed > n_total
        if pack:
            self.history.append(np.packbits(self.keep))
        elif keep_inside:
            parts = []
            if start:
                parts.append(np.flatnonzero(self.keep[:start]))
            parts.append(start + np.flatnonzero(self.keep[start:stop] & ~inside))
            if stop < n_total:
                parts.append(stop + np.flatnonzero(self.keep[stop:]))
            self.history.append(np.concatenate(parts))
        else:
            self.history.append(start + np.flatnonzero(inside))

        if keep_inside:
            keep = np.zeros(n_total, dtype=bool)
            keep[start:stop] = inside
            self.keep = keep
            self.count = hits
        else:
            self.keep[start:stop] &= ~inside
            self.count -= removed
        self.version += 1
        return removed

    def undo(self):
        if not self.history:
            return False
        last = self.history.pop()
        if last.dtype == np.uint8:
            self.keep = np.unpackbits(last, count=len(self.keep)).view(bool).copy()
            self.count = int(np.count_nonzero(self.keep))
        else:
            self.keep[last] = True
            self.count += len(last)
        self.version += 1
        return True

    def restore(self):
        self.history.clear()
        self.keep = np.ones(len(self.keep), dtype=bool)
        self.count = len(self.keep)
        self.version += 1
