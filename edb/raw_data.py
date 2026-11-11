"""Stage 1: reading raw EDB files (or cleaned data saved by this program) in acquisition order, and saving cleaned
data. Files are parsed in blocks on a few threads into two float arrays, so memory use is 16 bytes per row however
large the files are. pandas (only its C parser is used) is imported on the first read, not at start-up."""
import csv
import io
import math
import os
import re
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import config
from .trace import Cancelled

NEWLINE = re.compile(rb"\r\n|\r|\n")
SLOW_BYTES = 4096  # a block this small that the fast parser refuses is read line by line


def read_pair(line):
    parts = line.replace(",", " ").split()
    if len(parts) < 2:
        return None
    try:
        t, r = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    return (t, r) if math.isfinite(t) and math.isfinite(r) and r > 0 else None


def is_binary(path):
    return str(path).lower().endswith(".npz")


class Columns:
    """Float columns that grow as blocks are appended: one array per column, not a list of rows or blocks."""

    def __init__(self, count=2):
        self.arrays = [np.empty(1024) for _ in range(count)]
        self.n = 0

    def reserve(self, rows):
        if rows > len(self.arrays[0]):
            for array in self.arrays:
                array.resize(rows, refcheck=False)

    def append(self, *columns):
        end = self.n + len(columns[0])
        if end > len(self.arrays[0]):
            self.reserve(max(end, len(self.arrays[0]) * 3 // 2))
        for array, column in zip(self.arrays, columns):
            array[self.n:end] = column
        self.n = end

    def finish(self):
        for array in self.arrays:
            array.resize(self.n, refcheck=False)
        return self.arrays


def valid_rows(t, r):
    """The rows read_pair accepts (finite time, finite positive radius) and how many it rejects."""
    good = np.isfinite(t) & np.isfinite(r) & (r > 0)
    bad = len(t) - int(np.count_nonzero(good))
    return (t[good], r[good], bad) if bad else (t, r, 0)


def parse_lines(block):
    """Line-by-line parser: defines what a valid row is, and reads whatever the fast parser refuses."""
    lines = NEWLINE.split(block)
    if lines and not lines[-1]:
        lines.pop()
    pairs = [read_pair(line.decode("utf-8", "replace")) for line in lines]
    good = [pair for pair in pairs if pair is not None]
    values = np.array(good, dtype=float).reshape(-1, 2)
    return values[:, 0].copy(), values[:, 1].copy(), len(pairs) - len(good)


def parse_fast(block):
    """pandas' C parser on a block of clean numeric lines; raises on anything else (header, text, bad number)."""
    import pandas as pd
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = pd.read_csv(io.BytesIO(block.replace(b",", b" ")), sep=r"\s+", header=None, names=[0, 1],
                            usecols=[0, 1], dtype=np.float64, engine="c", skip_blank_lines=False,
                            quoting=csv.QUOTE_NONE, float_precision="round_trip")
    return valid_rows(frame[0].to_numpy(), frame[1].to_numpy())


def parse_block(block):
    """(time, radius, lines skipped) for a block of whole lines. A block the fast parser refuses is halved until the
    offending lines are isolated, so one header or stray line costs little and every line is still accounted for."""
    try:
        return parse_fast(block)
    except Exception:
        cut = block.find(b"\n", len(block) // 2) + 1
        if len(block) <= SLOW_BYTES or cut in (0, len(block)):
            return parse_lines(block)
        (t1, r1, bad1), (t2, r2, bad2) = parse_block(block[:cut]), parse_block(block[cut:])
        return np.concatenate([t1, t2]), np.concatenate([r1, r2]), bad1 + bad2


def byte_blocks(path):
    """Yield the file as byte blocks of about READ_BLOCK_BYTES that each end on a line break."""
    with open(path, "rb") as handle:
        rest, start = b"", True
        while True:
            data = handle.read(config.READ_BLOCK_BYTES)
            if start and data.startswith(b"\xef\xbb\xbf"):
                data = data[3:]
            start = False
            if not data:
                if rest:
                    yield rest
                return
            data = rest + data
            cut = data.rfind(b"\n") + 1
            if not cut:  # old Mac line ends; a final \r may be half of a \r\n pair, so never cut there
                cut = data.rfind(b"\r", 0, len(data) - 1) + 1
            if cut:
                yield data[:cut]
            rest = data[cut:]


def parsed_text(path, pool, cancel):
    """Yield (time, radius, lines skipped, bytes) per block, in file order, with a few blocks parsing at once."""
    pending = deque()
    for block in byte_blocks(path):
        if cancel is not None and cancel():
            raise Cancelled()
        pending.append((pool.submit(parse_block, block), len(block)))
        if len(pending) > config.READ_THREADS:
            future, size = pending.popleft()
            yield (*future.result(), size)
    for future, size in pending:
        yield (*future.result(), size)


def parsed_binary(path, size):
    """A cleaned .npz saved by write_cleaned, as one block: time and raw radius."""
    with np.load(path) as saved:
        if "time_s" not in saved or "radius_raw_um" not in saved:
            raise ValueError(f"{os.path.basename(path)} is not cleaned data saved by this program.")
        yield (*valid_rows(saved["time_s"], saved["radius_raw_um"]), size)


def read_files(paths, progress=None, cancel=None):
    """Concatenate EDB files in acquisition order: (time, radius, lines skipped). Column 1 = time, column 2 = radius
    (um); header lines skipped, so a cleaned CSV saved by this program (radius_raw_um in column 2) loads the same
    way, as does a cleaned .npz. A file whose clock restarts carries on from the previous file. progress(fraction)
    and cancel() -> bool are optional and are called between blocks."""
    if not paths:
        raise ValueError("Add one or more EDB files first.")
    sizes = [os.path.getsize(path) for path in paths]
    total, done = sum(sizes) or 1, 0
    out, skipped = Columns(2), 0
    last_raw = last_time = None
    last_step = 0.0
    with ThreadPoolExecutor(config.READ_THREADS) as pool:
        for path, size in zip(paths, sizes):
            offset, first = 0.0, True
            blocks = parsed_binary(path, size) if is_binary(path) else parsed_text(path, pool, cancel)
            for raw, radius, bad, block_bytes in blocks:
                skipped += bad
                done += block_bytes
                if len(raw):
                    if first:
                        if last_raw is not None and raw[0] <= last_raw:
                            offset = last_time + last_step - float(raw[0])
                        first = False
                    time = raw + offset
                    steps = np.diff(raw, prepend=raw[0] if last_raw is None else last_raw)
                    rising = np.flatnonzero(steps > 0)
                    if len(rising):
                        last_step = float(steps[rising[-1]])
                    last_raw, last_time = float(raw[-1]), float(time[-1])
                    if not out.n:  # size the arrays once from the first block's rows per byte
                        out.reserve(int(len(raw) / max(block_bytes, 1) * total * 1.02) + 1024)
                    out.append(time, radius)
                if progress is not None:
                    progress(min(done / total, 1.0))
    if not out.n:
        raise ValueError("No valid rows found (expected time and positive radius columns).")
    t, r = out.finish()
    t -= t[0]
    return t, r, skipped


def write_cleaned(path, trace, factor, sigma=None, progress=None, cancel=None):
    """Save the retained rows; returns how many. CSV: time_s, radius_raw_um, radius_corrected_um, plus the error
    columns when sigma (corrected-radius uncertainty, um) is given. A path ending .npz saves the same information as
    NumPy binary instead: about a third of the size, and written and re-read in seconds whatever the row count.
    Written to a .part file first, so a cancelled or failed save leaves no half-written file behind."""
    part = f"{path}.part"
    try:
        if is_binary(path):
            t, r = trace.active()
            with open(part, "wb") as handle:
                np.savez(handle, time_s=t, radius_raw_um=r, radius_correction_factor=factor,
                         radius_uncertainty_um=math.nan if sigma is None else sigma)
        else:
            with open(part, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                header = ["time_s", "radius_raw_um", "radius_corrected_um"]
                if sigma is not None:
                    header += ["radius_corrected_err_um", "radius_corrected_squared_um2",
                               "radius_corrected_squared_err_um2"]
                writer.writerow(header)
                written = 0
                for t, r in trace.blocks(rows=config.WRITE_ROWS):
                    if cancel is not None and cancel():
                        raise Cancelled()
                    rc = r * factor
                    if sigma is None:
                        writer.writerows(zip(t, r, rc))
                    else:
                        writer.writerows(zip(t, r, rc, np.full_like(rc, sigma), rc * rc, 2 * rc * sigma))
                    written += len(t)
                    if progress is not None:
                        progress(written / trace.count)
        os.replace(part, path)
    except BaseException:
        try:
            os.remove(part)
        except OSError:
            pass
        raise
    return trace.count
