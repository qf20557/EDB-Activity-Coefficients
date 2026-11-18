"""High-throughput kernels for multi-million-row traces.

Uses Numba JIT-compiled native machine code (cached on disk) when arrays are large and numba is installed,
with identical chunked NumPy fallbacks when arrays are small or numba is unavailable.
"""
import math

import numpy as np

from . import config

_NB_KERNELS = None
_JIT_THRESHOLD = 65_536  # below this row count, NumPy C loops are already sub-millisecond


def _load_numba():
    global _NB_KERNELS
    if _NB_KERNELS is not None:
        return _NB_KERNELS
    try:
        import numba as nb
    except ImportError:
        _NB_KERNELS = False
        return False

    @nb.njit(cache=True)
    def nb_is_sorted(t):
        for i in range(len(t) - 1):
            if t[i + 1] < t[i]:
                return False
        return True

    @nb.njit(cache=True)
    def nb_cull_mask(t, r, keep, start, stop, xmin, xmax, ymin, ymax, squared, scale):
        n = stop - start
        inside = np.zeros(n, dtype=np.bool_)
        hits = 0
        for j in range(n):
            i = start + j
            if not keep[i]:
                continue
            ti = t[i]
            if ti < xmin or ti > xmax:
                continue
            ri = r[i] * scale
            yi = ri * ri if squared else ri
            if ymin <= yi <= ymax:
                inside[j] = True
                hits += 1
        return inside, hits

    @nb.njit(cache=True)
    def nb_value_bounds(t, r, keep, start, stop, lo, hi, has_lo, has_hi):
        cnt = 0
        t_min, t_max = math.inf, -math.inf
        r_min, r_max = math.inf, -math.inf
        for i in range(start, stop):
            if not keep[i]:
                continue
            ti = t[i]
            if (has_lo and ti < lo) or (has_hi and ti > hi):
                continue
            ri = r[i]
            cnt += 1
            if ti < t_min:
                t_min = ti
            if ti > t_max:
                t_max = ti
            if ri < r_min:
                r_min = ri
            if ri > r_max:
                r_max = ri
        return cnt, t_min, t_max, r_min, r_max

    @nb.njit(cache=True)
    def nb_scan_fit_range(t, keep, start, stop, lo, hi):
        n = 0
        t_lo, t_hi = math.inf, -math.inf
        distinct = np.empty(10, dtype=np.float64)
        n_distinct = 0
        for i in range(start, stop):
            if not keep[i]:
                continue
            ti = t[i]
            if ti < lo or ti > hi:
                continue
            n += 1
            if ti < t_lo:
                t_lo = ti
            if ti > t_hi:
                t_hi = ti
            if n_distinct < 10:
                seen = False
                for k in range(n_distinct):
                    if distinct[k] == ti:
                        seen = True
                        break
                if not seen:
                    distinct[n_distinct] = ti
                    n_distinct += 1
        return n, t_lo, t_hi, n_distinct

    @nb.njit(cache=True)
    def nb_bin_trace(t, r, keep, start, stop, lo, hi, t_lo, t_hi, scale, max_bins):
        count = np.zeros(max_bins, dtype=np.float64)
        sum_t = np.zeros(max_bins, dtype=np.float64)
        sum_y = np.zeros(max_bins, dtype=np.float64)
        sum_yy = np.zeros(max_bins, dtype=np.float64)
        inv_span = max_bins / (t_hi - t_lo)
        last_bin = max_bins - 1
        for i in range(start, stop):
            if not keep[i]:
                continue
            ti = t[i]
            if ti < lo or ti > hi:
                continue
            yi = r[i] * scale
            idx = int((ti - t_lo) * inv_span)
            if idx > last_bin:
                idx = last_bin
            elif idx < 0:
                idx = 0
            count[idx] += 1.0
            sum_t[idx] += ti
            sum_y[idx] += yi
            sum_yy[idx] += yi * yi
        return count, sum_t, sum_y, sum_yy

    @nb.njit(cache=True, parallel=True)
    def nb_project_parallel(t, r, keep, start, stop, x_lo, x_hi, y_lo, y_hi, scale, squared, width, height):
        n_total = stop - start
        n_chunks = 4 if n_total >= 1_000_000 else 1
        grids = np.zeros((n_chunks, height, width), dtype=np.int32)
        vis_counts = np.zeros(n_chunks, dtype=np.int64)
        inv_x = (width - 1) / (x_hi - x_lo)
        inv_y = (height - 1) / (y_hi - y_lo)
        h1 = height - 1
        for c in nb.prange(n_chunks):
            a = start + (n_total * c) // n_chunks
            b = start + (n_total * (c + 1)) // n_chunks
            g = grids[c]
            vis = 0
            for i in range(a, b):
                if not keep[i]:
                    continue
                ti = t[i]
                if ti < x_lo or ti > x_hi:
                    continue
                ri = r[i] * scale
                yi = ri * ri if squared else ri
                if yi < y_lo or yi > y_hi:
                    continue
                px = int((ti - x_lo) * inv_x + 0.5)
                py = h1 - int((yi - y_lo) * inv_y + 0.5)
                if 0 <= px < width and 0 <= py < height:
                    g[py, px] += 1
                    vis += 1
            vis_counts[c] = vis
        out = grids[0]
        total_vis = int(vis_counts[0])
        for c in range(1, n_chunks):
            out += grids[c]
            total_vis += int(vis_counts[c])
        return out, total_vis

    @nb.njit(cache=True)
    def nb_error_envelope(t, r, keep, start, stop, x_lo, x_hi, y_lo, y_hi, scale, sigma, squared, width, height):
        """Per-column [py_top, py_bot] pixel span of [r*scale - sigma, r*scale + sigma] across all visible points."""
        col_top = np.full(width, height, dtype=np.int32)
        col_bot = np.full(width, -1, dtype=np.int32)
        inv_x = (width - 1) / (x_hi - x_lo)
        inv_y = (height - 1) / (y_hi - y_lo)
        h1 = height - 1
        for i in range(start, stop):
            if not keep[i]:
                continue
            ti = t[i]
            if ti < x_lo or ti > x_hi:
                continue
            px = int((ti - x_lo) * inv_x + 0.5)
            if px < 0 or px >= width:
                continue
            ri = r[i] * scale
            lo_r = ri - sigma if ri > sigma else 0.0
            hi_r = ri + sigma
            ylo = lo_r * lo_r if squared else lo_r
            yhi = hi_r * hi_r if squared else hi_r
            if yhi < y_lo or ylo > y_hi:
                continue
            py_top = h1 - int((min(yhi, y_hi) - y_lo) * inv_y + 0.5)
            py_bot = h1 - int((max(ylo, y_lo) - y_lo) * inv_y + 0.5)
            if py_top < 0:
                py_top = 0
            if py_bot > h1:
                py_bot = h1
            if py_top < col_top[px]:
                col_top[px] = py_top
            if py_bot > col_bot[px]:
                col_bot[px] = py_bot
        return col_top, col_bot

    _NB_KERNELS = dict(
        is_sorted=nb_is_sorted,
        cull_mask=nb_cull_mask,
        value_bounds=nb_value_bounds,
        scan_fit_range=nb_scan_fit_range,
        bin_trace=nb_bin_trace,
        project=nb_project_parallel,
        error_envelope=nb_error_envelope,
    )
    return _NB_KERNELS


def is_sorted_fast(t):
    if len(t) <= 1:
        return True
    if len(t) >= _JIT_THRESHOLD:
        nb_k = _load_numba()
        if nb_k:
            return bool(nb_k["is_sorted"](t))
    rows = config.CHUNK_ROWS
    for a in range(0, len(t) - 1, rows):
        block = t[a:a + rows + 1]
        if np.any(block[1:] < block[:-1]):
            return False
    return True


def cull_mask_fast(t, r, keep, start, stop, xmin, xmax, ymin, ymax, squared, scale):
    n = stop - start
    if n <= 0:
        return np.zeros(0, dtype=bool), 0
    if n >= _JIT_THRESHOLD:
        nb_k = _load_numba()
        if nb_k:
            inside, hits = nb_k["cull_mask"](
                t, r, keep, start, stop, float(xmin), float(xmax), float(ymin), float(ymax),
                bool(squared), float(scale)
            )
            return inside, int(hits)
    inside = np.zeros(n, dtype=bool)
    for a in range(start, stop, config.CHUNK_ROWS):
        b = min(a + config.CHUNK_ROWS, stop)
        tt, rr = t[a:b], r[a:b] * scale
        yy = rr * rr if squared else rr
        inside[a - start:b - start] = keep[a:b] & (tt >= xmin) & (tt <= xmax) & (yy >= ymin) & (yy <= ymax)
    return inside, int(np.count_nonzero(inside))


def value_bounds_fast(t, r, keep, start, stop, lo=None, hi=None):
    """Return (count, t_min, t_max, r_min, r_max) of retained rows in [start, stop) matching [lo, hi]."""
    n = stop - start
    if n <= 0:
        return 0, math.inf, -math.inf, math.inf, -math.inf
    if n >= _JIT_THRESHOLD:
        nb_k = _load_numba()
        if nb_k:
            cnt, t0, t1, r0, r1 = nb_k["value_bounds"](
                t, r, keep, start, stop,
                0.0 if lo is None else float(lo),
                0.0 if hi is None else float(hi),
                lo is not None, hi is not None,
            )
            return int(cnt), float(t0), float(t1), float(r0), float(r1)
    cnt = 0
    t_min, t_max = math.inf, -math.inf
    r_min, r_max = math.inf, -math.inf
    for a in range(start, stop, config.CHUNK_ROWS):
        b = min(a + config.CHUNK_ROWS, stop)
        tt, rr, m = t[a:b], r[a:b], keep[a:b]
        if lo is not None:
            m = m & (tt >= lo)
        if hi is not None:
            m = m & (tt <= hi)
        if m.all():
            ts, rs = tt, rr
        elif m.any():
            ts, rs = tt[m], rr[m]
        else:
            continue
        cnt += len(ts)
        t_min, t_max = min(t_min, float(ts.min())), max(t_max, float(ts.max()))
        r_min, r_max = min(r_min, float(rs.min())), max(r_max, float(rs.max()))
    return cnt, t_min, t_max, r_min, r_max


def scan_and_bin_trace(trace, lo, hi, scale, max_bins):
    """Return (n, t_lo, t_hi, distinct_count, tb, rb, wb, within) for fit_trace."""
    start, stop = trace.window(lo, hi)
    if stop - start >= _JIT_THRESHOLD:
        nb_k = _load_numba()
        if nb_k:
            n, t_lo, t_hi, n_distinct = nb_k["scan_fit_range"](
                trace.t, trace.keep, start, stop, float(lo), float(hi)
            )
            n = int(n)
            if n < 10 or int(n_distinct) < 10:
                return n, float(t_lo), float(t_hi), int(n_distinct), None, None, None, 0.0
            if n <= max_bins:
                cols = [(tt, rr * scale) for tt, rr in trace.blocks(lo, hi)]
                tb = np.concatenate([c[0] for c in cols])
                rb = np.concatenate([c[1] for c in cols])
                return n, float(t_lo), float(t_hi), int(n_distinct), tb, rb, np.ones(n), 0.0
            count, sum_t, sum_y, sum_yy = nb_k["bin_trace"](
                trace.t, trace.r, trace.keep, start, stop,
                float(lo), float(hi), float(t_lo), float(t_hi), float(scale), int(max_bins)
            )
            valid = count > 0
            mean_y = np.where(valid, sum_y / np.where(valid, count, 1.0), 0.0)
            within = max(float(np.sum(sum_yy - count * mean_y ** 2)), 0.0)
            tb = (sum_t / np.where(valid, count, 1.0))[valid]
            return n, float(t_lo), float(t_hi), int(n_distinct), tb, mean_y[valid], count[valid], within
    return None


def project_points_fast(trace, x_lo, x_hi, y_lo, y_hi, scale=1.0, squared=False,
                        width=config.RASTER_WIDTH, height=config.RASTER_HEIGHT):
    """Project ALL retained points of `trace` inside [x_lo, x_hi] x [y_lo, y_hi] onto a (height, width) count grid."""
    if x_hi <= x_lo or y_hi <= y_lo or trace.count == 0:
        return np.zeros((height, width), dtype=np.int32), 0
    start, stop = trace.window(x_lo, x_hi)
    if stop <= start:
        return np.zeros((height, width), dtype=np.int32), 0
    nb_k = _load_numba()
    if nb_k:
        grid, vis = nb_k["project"](
            trace.t, trace.r, trace.keep, start, stop,
            float(x_lo), float(x_hi), float(y_lo), float(y_hi),
            float(scale), bool(squared), int(width), int(height),
        )
        return grid, int(vis)

    # Pure-NumPy chunked fallback (also plots 100% of points, zero decimation)
    grid = np.zeros((height, width), dtype=np.int32)
    inv_x = (width - 1) / (x_hi - x_lo)
    inv_y = (height - 1) / (y_hi - y_lo)
    h1 = height - 1
    vis = 0
    flat_grid = grid.ravel()
    for tt, rr in trace.blocks(x_lo, x_hi):
        rs = rr * scale
        yy = rs * rs if squared else rs
        m = (yy >= y_lo) & (yy <= y_hi)
        if not m.any():
            continue
        if not m.all():
            tt, yy = tt[m], yy[m]
        px = np.clip(((tt - x_lo) * inv_x + 0.5).astype(np.int64), 0, width - 1)
        py = np.clip(h1 - ((yy - y_lo) * inv_y + 0.5).astype(np.int64), 0, h1)
        np.add.at(flat_grid, py * width + px, 1)
        vis += len(tt)
    return grid, vis


def error_envelope_fast(trace, x_lo, x_hi, y_lo, y_hi, scale, sigma, squared,
                        width=config.RASTER_WIDTH, height=config.RASTER_HEIGHT):
    """Per-column [top, bot] pixel indices for vertical uncertainty bars across all visible points."""
    start, stop = trace.window(x_lo, x_hi)
    if stop <= start or sigma <= 0:
        return None
    nb_k = _load_numba()
    if nb_k:
        return nb_k["error_envelope"](
            trace.t, trace.r, trace.keep, start, stop,
            float(x_lo), float(x_hi), float(y_lo), float(y_hi),
            float(scale), float(sigma), bool(squared), int(width), int(height),
        )
    col_top = np.full(width, height, dtype=np.int32)
    col_bot = np.full(width, -1, dtype=np.int32)
    inv_x = (width - 1) / (x_hi - x_lo)
    inv_y = (height - 1) / (y_hi - y_lo)
    h1 = height - 1
    for tt, rr in trace.blocks(x_lo, x_hi):
        px = ((tt - x_lo) * inv_x + 0.5).astype(np.int64)
        valid_x = (px >= 0) & (px < width)
        if not valid_x.any():
            continue
        tt, rr, px = tt[valid_x], rr[valid_x] * scale, px[valid_x]
        ylo = np.clip(rr - sigma, 0.0, None)
        yhi = rr + sigma
        if squared:
            ylo, yhi = ylo * ylo, yhi * yhi
        vis = (yhi >= y_lo) & (ylo <= y_hi)
        if not vis.any():
            continue
        px = px[vis]
        py_top = np.clip(h1 - ((np.minimum(yhi[vis], y_hi) - y_lo) * inv_y + 0.5).astype(np.int32), 0, h1)
        py_bot = np.clip(h1 - ((np.maximum(ylo[vis], y_lo) - y_lo) * inv_y + 0.5).astype(np.int32), 0, h1)
        np.minimum.at(col_top, px, py_top)
        np.maximum.at(col_bot, px, py_bot)
    return col_top, col_bot
