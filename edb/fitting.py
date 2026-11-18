"""Stage 1: monotone convex fit (non-negative sum of decaying exponentials)."""
import math

import numpy as np

from .config import DICT_TAUS, FIT_EXPORT_POINTS, FIT_EXPORT_STEP, MAX_FIT_BINS, MAX_FIT_EXPORT_POINTS
from .fast import scan_and_bin_trace


def _nnls(A, b):
    m, n = A.shape
    P = np.zeros(n, dtype=bool)
    x = np.zeros(n)
    w = A.T @ (b - A @ x)
    tol = 1e-12 * max(1.0, float(np.abs(w).max()))
    for _ in range(10 * n):
        if P.all() or w[~P].max() <= tol:
            break
        j = int(np.argmax(np.where(~P, w, -np.inf)))
        P[j] = True
        while True:
            s = np.zeros(n)
            s[P] = np.linalg.lstsq(A[:, P], b, rcond=None)[0]
            if s[P].min() > 0:
                x = s
                break
            neg = P & (s <= 0)
            alpha = np.min(x[neg] / (x[neg] - s[neg]))
            x = x + alpha * (s - x)
            P &= x > 1e-14
            x[~P] = 0.0
        w = A.T @ (b - A @ x)
    return x


def _bin_blocks(blocks, lo, hi, max_bins=MAX_FIT_BINS):
    count, sum_t, sum_y, sum_yy = (np.zeros(max_bins) for _ in range(4))
    for t, y in blocks:
        idx = np.minimum(((t - lo) / (hi - lo) * max_bins).astype(np.int64), max_bins - 1)
        count += np.bincount(idx, minlength=max_bins)
        sum_t += np.bincount(idx, weights=t, minlength=max_bins)
        sum_y += np.bincount(idx, weights=y, minlength=max_bins)
        sum_yy += np.bincount(idx, weights=y * y, minlength=max_bins)
    keep = count > 0
    mean_y = np.where(keep, sum_y / np.where(keep, count, 1), 0.0)
    within = float(np.sum(sum_yy - count * mean_y ** 2))
    return (sum_t / np.where(keep, count, 1))[keep], mean_y[keep], count[keep], max(within, 0.0)


def _solve_decays(u, y, sw, taus):
    cols = [np.ones_like(u)] + [np.exp(-u / tau) for tau in taus]
    A = np.column_stack(cols) * sw[:, None]
    x = _nnls(A, y * sw)
    resid = A @ x - y * sw
    return x[0], x[1:], float(resid @ resid)


def fit_trace(trace, lo=None, hi=None, scale=1.0):
    """Fit scale * radius of the retained rows of a Trace with lo <= t <= hi. The trace is read in blocks (or a single
    compiled Numba pass on large traces), so no copy of it is made however long it is."""
    first, last = trace.time_range()
    lo = first if lo is None else lo
    hi = last if hi is None else hi
    if lo >= hi:
        raise ValueError("Fit start must be below fit end.")

    fast_res = scan_and_bin_trace(trace, lo, hi, scale, MAX_FIT_BINS)
    if fast_res is not None:
        n, t_lo, t_hi, n_distinct, tb, rb, wb, within = fast_res
        if n < 10 or n_distinct < 10:
            raise ValueError("Need at least 10 retained points at distinct times for this model.")
    else:
        n, t_lo, t_hi, distinct, small = 0, math.inf, -math.inf, np.empty(0), []
        for tt, rr in trace.blocks(lo, hi):
            n += len(tt)
            t_lo, t_hi = min(t_lo, float(tt.min())), max(t_hi, float(tt.max()))
            if len(distinct) < 10:
                distinct = np.unique(np.concatenate([distinct, tt]))
            if n <= MAX_FIT_BINS:
                small.append((tt, rr * scale))
        if n < 10 or len(distinct) < 10:
            raise ValueError("Need at least 10 retained points at distinct times for this model.")
        if n <= MAX_FIT_BINS:
            tb, rb = (np.concatenate(column) for column in zip(*small))
            wb, within = np.ones(n), 0.0
        else:
            tb, rb, wb, within = _bin_blocks(((tt, rr * scale) for tt, rr in trace.blocks(lo, hi)), t_lo, t_hi)

    span = t_hi - t_lo
    u, sw = (tb - t_lo) / span, np.sqrt(wb)
    offset, amps, _ = _solve_decays(u, rb, sw, DICT_TAUS)
    used = amps > 0
    taus = DICT_TAUS[used]
    amps = amps[used]

    def func(time):
        uu = (np.asarray(time, dtype=float) - t_lo) / span
        return offset + sum(a * np.exp(-uu / g) for a, g in zip(amps, taus))

    terms = " + ".join(f"{a:.6g}·exp(-t/{g * span:.6g} s)" for a, g in zip(amps, taus))
    detail = f"r(t) = {offset:.8g} + {terms}  (plateau and amplitudes constrained ≥ 0)"
    fitted = func(tb)
    ss_res = float(np.sum(wb * (rb - fitted) ** 2)) + within
    mean = float(np.sum(wb * rb) / wb.sum())
    ss_tot = float(np.sum(wb * (rb - mean) ** 2)) + within
    return dict(func=func, label="Monotone convex fit", lo=t_lo, hi=t_hi, n=n, rms=math.sqrt(ss_res / n),
                r2=(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"), detail=detail)


def export_points(fit):
    """How many points the exported fit has: FIT_EXPORT_POINTS, or more on a long trace (see config)."""
    by_step = math.ceil((fit["hi"] - fit["lo"]) / FIT_EXPORT_STEP) + 1
    return int(min(MAX_FIT_EXPORT_POINTS, max(FIT_EXPORT_POINTS, by_step)))


def fit_curve(fit, points=800):
    x = np.linspace(fit["lo"], fit["hi"], points)
    radius = np.asarray(fit["func"](x), dtype=float)
    return x, radius, radius ** 2
