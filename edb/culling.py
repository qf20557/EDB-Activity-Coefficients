"""Stage 1: removing points with boxes, with undo."""
import numpy as np

from . import config
from .trace import Trace


class PointStore(Trace):
    """The loaded trace. Culling only clears rows of the keep mask; the data itself is never changed or copied."""

    def __init__(self, t, r):
        super().__init__(t, r)
        self.history = []  # per cull: the removed row numbers, or the whole previous mask (bit-packed) if smaller

    def ids(self):
        return np.flatnonzero(self.keep)

    def cull_box(self, box, squared, keep_inside=False, scale=1.0):
        xmin, xmax, ymin, ymax = box
        start, stop = self.window(xmin, xmax)
        inside = np.zeros(stop - start, dtype=bool)
        for a in range(start, stop, config.CHUNK_ROWS):
            b = min(a + config.CHUNK_ROWS, stop)
            t, r = self.t[a:b], self.r[a:b] * scale
            y = r * r if squared else r
            inside[a - start:b - start] = self.keep[a:b] & (t >= xmin) & (t <= xmax) & (y >= ymin) & (y <= ymax)
        hits = int(np.count_nonzero(inside))
        removed = self.count - hits if keep_inside else hits
        if removed == self.count:
            raise ValueError("That would remove every point.")
        if not removed:
            return 0
        if keep_inside:
            keep = np.zeros(len(self.keep), dtype=bool)
            keep[start:stop] = inside
        else:
            keep = self.keep.copy()
            keep[start:stop] &= ~inside
        if 64 * removed > len(keep):
            self.history.append(np.packbits(self.keep))
        else:
            self.history.append(np.flatnonzero(self.keep & ~keep))
        self._set(keep)
        return removed

    def _set(self, keep):
        self.keep = keep
        self.count = int(np.count_nonzero(keep))
        self.version += 1

    def undo(self):
        if not self.history:
            return False
        last = self.history.pop()
        if last.dtype == np.uint8:
            keep = np.unpackbits(last, count=len(self.keep)).view(bool)
        else:
            keep = self.keep
            keep[last] = True
        self._set(keep)
        return True

    def restore(self):
        self.history.clear()
        self._set(np.ones(len(self.keep), dtype=bool))
