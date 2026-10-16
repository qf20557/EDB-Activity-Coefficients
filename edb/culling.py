"""Stage 1: removing points with boxes, with undo."""
import numpy as np


class PointStore:
    def __init__(self, data):
        self.data = data
        self.keep = np.ones(len(data), dtype=bool)
        self.history = []

    def ids(self):
        return np.flatnonzero(self.keep)

    def active(self):
        rows = self.data[self.ids()]
        return rows[:, 0], rows[:, 1]

    def cull_box(self, box, squared, keep_inside=False, scale=1.0):
        xmin, xmax, ymin, ymax = box
        ids = self.ids()
        t, r = self.data[ids, 0], self.data[ids, 1] * scale
        y = r * r if squared else r
        inside = (t >= xmin) & (t <= xmax) & (y >= ymin) & (y <= ymax)
        removed = ids[~inside if keep_inside else inside]
        if len(removed) == len(ids):
            raise ValueError("That would remove every point.")
        if len(removed):
            self.keep[removed] = False
            self.history.append(removed)
        return len(removed)

    def undo(self):
        if not self.history:
            return False
        self.keep[self.history.pop()] = True
        return True

    def restore(self):
        self.keep[:] = True
        self.history.clear()
