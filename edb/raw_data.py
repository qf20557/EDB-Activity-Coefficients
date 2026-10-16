"""Stage 1: reading raw EDB files (or a cleaned CSV) in acquisition order."""
import math

import numpy as np


def read_pair(line):
    parts = line.replace(",", " ").split()
    if len(parts) < 2:
        return None
    try:
        t, r = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    return (t, r) if math.isfinite(t) and math.isfinite(r) and r > 0 else None


def read_files(paths):
    """Concatenate EDB files in acquisition order. Column 1 = time, column 2 = radius (um); header lines skipped,
    so a cleaned CSV saved by this program (radius_raw_um in column 2) loads the same way."""
    if not paths:
        raise ValueError("Add one or more EDB files first.")
    rows, skipped = [], 0
    last_raw = last_time = None
    last_step = 0.0
    for path in paths:
        offset, first = 0.0, True
        with open(path, encoding="utf-8-sig", errors="replace") as handle:
            for line in handle:
                pair = read_pair(line)
                if pair is None:
                    skipped += 1
                    continue
                raw, radius = pair
                if first:
                    if last_raw is not None and raw <= last_raw:
                        offset = last_time + last_step - raw
                    first = False
                time = raw + offset
                if last_raw is not None and raw > last_raw:
                    last_step = raw - last_raw
                last_raw, last_time = raw, time
                rows.append((time, radius))
    if not rows:
        raise ValueError("No valid rows found (expected time and positive radius columns).")
    data = np.asarray(rows, dtype=float)
    data[:, 0] -= data[0, 0]
    return data, skipped
