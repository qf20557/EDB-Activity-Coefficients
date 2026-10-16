"""Stage 2: mass fraction of solute (MFS) model and binned gradients."""
import math

import numpy as np

from .config import MAX_BINS, MIN_POINTS


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


def bin_edges(lo, hi, width):
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        raise ValueError("Interval start must be below interval end.")
    if not math.isfinite(width) or width < 0:
        raise ValueError("Bin width must be zero (single interval) or positive.")
    if width == 0:
        return np.array([lo, hi])
    if (hi - lo) / width > MAX_BINS:
        raise ValueError("Too many bins; increase the bin width.")
    return np.r_[np.arange(lo, hi, width), hi]


def interval_results(t, scenarios, lo, hi, width):
    """scenarios: [(r2, mfs), ...], nominal first (then radius - uncertainty, radius + uncertainty). Each row holds
    (average MFS, gradient) for every scenario in turn. A bin is skipped when the nominal scenario is invalid there;
    another scenario that is invalid in a kept bin gives NaN for that bin."""
    x_all = [mfs(r2) for r2, mfs in scenarios]
    edges = bin_edges(lo, hi, width)
    rows, skipped = [], 0
    for j, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        last = j == len(edges) - 2
        mask = (t >= a) & ((t <= b) if last else (t < b))
        tt = t[mask]
        if len(tt) < MIN_POINTS or np.any(~np.isfinite(x_all[0][mask])):
            skipped += 1
            continue
        span = tt[-1] - tt[0]
        row = []
        for (r2, _), x in zip(scenarios, x_all):
            yy, xx = r2[mask], x[mask]
            if np.all(np.isfinite(xx)) and np.all(np.isfinite(yy)):
                row += [float(np.sum(0.5 * (xx[1:] + xx[:-1]) * np.diff(tt)) / span),
                        float(np.polyfit(tt - tt.mean(), yy, 1)[0])]
            else:
                row += [math.nan, math.nan]
        rows.append(tuple(row))
    return rows, skipped
