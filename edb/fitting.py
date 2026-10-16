"""Stage 1: monotone convex fit (non-negative sum of decaying exponentials)."""
import math

import numpy as np

from .config import DICT_TAUS, MAX_FIT_BINS


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


def bin_series(t, y, max_bins=MAX_FIT_BINS):
    if len(t) <= max_bins:
        return t, y, np.ones(len(t)), 0.0
    lo, hi = float(t.min()), float(t.max())
    idx = np.minimum(((t - lo) / (hi - lo) * max_bins).astype(np.int64), max_bins - 1)
    count = np.bincount(idx, minlength=max_bins).astype(float)
    keep = count > 0
    sum_t = np.bincount(idx, weights=t, minlength=max_bins)
    sum_y = np.bincount(idx, weights=y, minlength=max_bins)
    sum_yy = np.bincount(idx, weights=y * y, minlength=max_bins)
    mean_y = np.where(keep, sum_y / np.where(keep, count, 1), 0.0)
    within = float(np.sum(sum_yy - count * mean_y ** 2))
    return (sum_t / np.where(keep, count, 1))[keep], mean_y[keep], count[keep], max(within, 0.0)


def _solve_decays(u, y, sw, taus):
    cols = [np.ones_like(u)] + [np.exp(-u / tau) for tau in taus]
    A = np.column_stack(cols) * sw[:, None]
    x = _nnls(A, y * sw)
    resid = A @ x - y * sw
    return x[0], x[1:], float(resid @ resid)


def fit_model(t, r, lo=None, hi=None):
    lo = t.min() if lo is None else lo
    hi = t.max() if hi is None else hi
    if lo >= hi:
        raise ValueError("Fit start must be below fit end.")
    mask = (t >= lo) & (t <= hi)
    n = int(mask.sum())
    tt, rr = t[mask], r[mask]
    if n < 10 or len(np.unique(tt)) < 10:
        raise ValueError("Need at least 10 retained points at distinct times for this model.")
    t_lo, t_hi = float(tt.min()), float(tt.max())
    span = t_hi - t_lo
    tb, rb, wb, within = bin_series(tt, rr)
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


def fit_curve(fit, points=800):
    x = np.linspace(fit["lo"], fit["hi"], points)
    radius = np.asarray(fit["func"](x), dtype=float)
    return x, radius, radius ** 2
