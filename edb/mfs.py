"""Stage 2: mass fraction of solute (MFS) model and binned gradients."""
import math

import numpy as np

from .config import AUTO_BIN_SECONDS, BIN_RESAMPLE_POINTS, MAX_BINS, MIN_POINTS


def make_mfs_function(records, name_a, name_b, name_s, mass_a, mass_b, mass_s, r0):
    if len({name_a, name_b, name_s}) < 3:
        raise ValueError("Choose three different compounds for A, B and the solvent.")
    if not all(name in records for name in (name_a, name_b, name_s)):
        raise ValueError("Choose A, B and the solvent from compounds.json.")
    if not all(math.isfinite(v) for v in (mass_a, mass_b, mass_s, r0)):
        raise ValueError("Masses and initial radius must be finite numbers.")
    if mass_a <= 0 or mass_b <= 0 or mass_s < 0 or r0 <= 0:
        raise ValueError("A and B masses and initial radius must be positive; solvent mass cannot be negative.")
    a, b, s = records[name_a], records[name_b], records[name_s]
    vol_a, vol_b = mass_a / a["density_g_ml"], mass_b / b["density_g_ml"]
    vol_s = mass_s / s["density_g_ml"]
    v_dry, v0 = vol_a + vol_b, vol_a + vol_b + vol_s
    n_a0, n_b = mass_a / a["molar_mass_g_mol"], mass_b / b["molar_mass_g_mol"]
    rho_a, m_a = a["density_g_ml"], a["molar_mass_g_mol"]

    def mfs(r2):
        r2 = np.asarray(r2, dtype=float)
        volume = v0 * (r2 / r0 ** 2) ** 1.5
        n_a = np.where(volume >= v_dry, n_a0, (volume - vol_b) * rho_a / m_a)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(n_a > 0, n_a / (n_a + n_b), np.nan)

    info = dict(x_a0=n_a0 / (n_a0 + n_b), r_dry=r0 * (v_dry / v0) ** (1 / 3), v0=v0, v_dry=v_dry)
    return mfs, info


def auto_bin_count(lo, hi, target=AUTO_BIN_SECONDS):
    """How many equal bins of about `target` seconds fill the interval (at least one)."""
    return max(1, round((hi - lo) / target))


def bin_edges(lo, hi, width=None):
    """Bin edges from lo to hi. width None (automatic): equal bins that fill the interval exactly, as many as makes
    each closest to AUTO_BIN_SECONDS. A width: bins of that width from lo, the last one shorter if it does not
    divide the interval; 0 gives a single interval."""
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        raise ValueError("Interval start must be below interval end.")
    if width is None:
        return np.linspace(lo, hi, auto_bin_count(lo, hi) + 1)
    if not math.isfinite(width) or width < 0:
        raise ValueError("Bin width must be zero (single interval) or positive.")
    if width == 0:
        return np.array([lo, hi])
    if (hi - lo) / width > MAX_BINS:
        raise ValueError("Too many bins; increase the bin width.")
    return np.r_[np.arange(lo, hi, width), hi]


def interval_results(t, scenarios, lo, hi, width=None):
    """Returns (rows, skipped, resampled). scenarios: [(r2, mfs), ...], nominal first (then radius - uncertainty,
    radius + uncertainty). Each row holds (average MFS, gradient) for every scenario in turn. A bin is skipped only
    when the nominal scenario is invalid there (A would be negative); another scenario that is invalid in a kept bin
    gives NaN for that bin. A bin holding fewer than MIN_POINTS fit points (a fit sampled more coarsely than the
    bins) is not skipped: the fit is interpolated at BIN_RESAMPLE_POINTS evenly spaced times across it; resampled
    counts those bins."""
    x_all = [mfs(r2) for r2, mfs in scenarios]
    edges = bin_edges(lo, hi, width)
    rows, skipped, resampled = [], 0, 0
    for j, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        last = j == len(edges) - 2
        mask = (t >= a) & ((t <= b) if last else (t < b))
        tt = t[mask]
        sparse = len(tt) < MIN_POINTS
        if sparse:
            tt = np.linspace(max(a, t[0]), min(b, t[-1]), BIN_RESAMPLE_POINTS)
            if tt[-1] <= tt[0]:  # the bin lies outside the fit
                skipped += 1
                continue
            values = [np.interp(tt, t, r2) for r2, _ in scenarios]
            samples = [(r2, mfs(r2)) for r2, (_, mfs) in zip(values, scenarios)]
        else:
            samples = [(r2[mask], x[mask]) for (r2, _), x in zip(scenarios, x_all)]
        if np.any(~np.isfinite(samples[0][1])):
            skipped += 1
            continue
        resampled += sparse
        span = tt[-1] - tt[0]
        row = []
        for yy, xx in samples:
            if np.all(np.isfinite(xx)) and np.all(np.isfinite(yy)):
                row += [float(np.sum(0.5 * (xx[1:] + xx[:-1]) * np.diff(tt)) / span),
                        float(np.polyfit(tt - tt.mean(), yy, 1)[0])]
            else:
                row += [math.nan, math.nan]
        rows.append(tuple(row))
    return rows, skipped, resampled
