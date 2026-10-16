"""Stage 3: activity coefficients and the vapour-pressure uncertainty."""
import math
import re

import numpy as np
import pandas as pd

from .config import CSV_COLUMNS, PVAP_COLUMNS, THEORY_CHECK, THEORY_G, THEORY_P, THEORY_SIGMA, THEORY_X


def sort_mfs_gradient(x, g, extra=None):
    """Sort by MFS. extra: optional (n, 4) array of (MFS, gradient) at radius - / + uncertainty, kept aligned."""
    x, g = np.asarray(x, dtype=float), np.asarray(g, dtype=float)
    good = np.isfinite(x) & np.isfinite(g)
    if not good.any():
        raise ValueError("No numeric rows found.")
    order = np.argsort(x[good], kind="stable")
    if extra is not None:
        extra = np.asarray(extra, dtype=float).reshape(len(x), 4)[good][order]
    return x[good][order], g[good][order], extra


def read_mfs_gradient(path):
    """(MFS, gradient, extra). extra holds the uncertainty columns of a stage 2 export, or None."""
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    if df.shape[1] < 2:
        raise ValueError("The CSV needs two columns: Average MFS and Gradient.")
    x = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(float)
    g = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(float)
    extra = None
    if [str(c).strip() for c in df.columns[2:6]] == CSV_COLUMNS[2:]:
        extra = df.iloc[:, 2:6].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return sort_mfs_gradient(x, g, extra)


def activity_coefficients(x_exp, g_exp, x_theory, g_theory, extra=None):
    """gamma = experimental / theoretical gradient at the same MFS. With extra (see sort_mfs_gradient) each shifted
    (MFS, gradient) pair gives its own gamma; bounds = (x_lo, x_hi, gamma_lo, gamma_hi) span the nominal value and
    both shifted values (a shifted gamma outside the theoretical range is left out, so that bar is truncated)."""
    ux, idx = np.unique(x_theory, return_index=True)
    gt = g_theory[idx]
    tiny = 1e-12 * max(np.abs(gt).max(), 1e-300)

    def theory_at(x):
        inside = (x >= ux[0]) & (x <= ux[-1])
        value = np.full(len(x), np.nan)
        value[inside] = np.interp(x[inside], ux, gt)
        return inside, value

    inside, theory = theory_at(x_exp)
    nonzero = np.abs(np.nan_to_num(theory)) > tiny
    good = inside & nonzero
    x, gamma = x_exp[good], g_exp[good] / theory[good]
    bounds, truncated = None, 0
    if extra is not None:
        xs, gammas = [x], [gamma]
        for xe, ge in ((extra[good, 0], extra[good, 1]), (extra[good, 2], extra[good, 3])):
            ok, th = theory_at(xe)
            ok &= np.abs(np.nan_to_num(th)) > tiny
            with np.errstate(divide="ignore", invalid="ignore"):
                gammas.append(np.where(ok & np.isfinite(ge), ge / th, np.nan))
            xs.append(xe)
            truncated |= ~np.isfinite(gammas[-1])
        xs, gammas = np.vstack(xs), np.vstack(gammas)
        bounds = (np.nanmin(xs, 0), np.nanmax(xs, 0), np.nanmin(gammas, 0), np.nanmax(gammas, 0))
        truncated = int(np.sum(truncated))
        if not any(np.any(hi > lo) for lo, hi in (bounds[:2], bounds[2:])):
            bounds = None  # zero uncertainty
    return x, gamma, int((~inside).sum()), int((inside & ~nonzero).sum()), bounds, truncated


def column_key(name):
    """Header normalised for matching by name: lower case, µ/μ -> u, ² -> 2, letters and digits only."""
    text = str(name).strip().lower().replace("µ", "u").replace("μ", "u").replace("²", "2")
    return re.sub(r"[^a-z0-9]", "", text)


def read_theory_csv(path):
    """(MFS, gradient, pvap) from a theoretical (DropletLab) CSV, columns matched by name. A file without the named
    MFS / gradient columns (older 2-column files) is read exactly as before, with pvap None. Otherwise pvap is None
    (no uncertainty columns) or a dict: p, p_lower, p_upper (Pa, or None), sigma (gradient uncertainty aligned with
    the returned MFS, or None) and check (worst deviation of bound-run gradient / nominal gradient from p_bound / p,
    or None). The bound-run columns are only used for that check: the lower run does not reach low MFS."""
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    cols = {column_key(c): c for c in df.columns}
    if column_key(THEORY_X) not in cols or column_key(THEORY_G) not in cols:
        return (*read_mfs_gradient(path)[:2], None)

    def column(name):
        key = column_key(name)
        return pd.to_numeric(df[cols[key]], errors="coerce").to_numpy(float) if key in cols else None

    def constant(values):
        values = values[np.isfinite(values) & (values > 0)] if values is not None else []
        return float(np.median(values)) if len(values) else None

    x, g = column(THEORY_X), column(THEORY_G)
    good = np.isfinite(x) & np.isfinite(g)
    x_sorted, g_sorted, _ = sort_mfs_gradient(x, g)
    order = np.argsort(x[good], kind="stable")  # the order sort_mfs_gradient used
    sigma = column(THEORY_SIGMA)
    if sigma is not None:
        sigma = sigma[good][order]
        if not np.any(np.isfinite(sigma)):
            sigma = None
    p = [constant(column(name)) for name in THEORY_P]
    if None in p:
        p = [None] * 3
    check = None
    if p[0] is not None:
        deviations = []
        for name, p_bound in zip(THEORY_CHECK, p[1:]):
            g_bound = column(name)
            if g_bound is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    d = np.abs(g_bound / g / (p_bound / p[0]) - 1)
                d = d[np.isfinite(d)]
                if d.size:
                    deviations.append(float(d.max()))
        check = max(deviations) if deviations else None
    if p[0] is None and sigma is None:
        return x_sorted, g_sorted, None
    return x_sorted, g_sorted, dict(p=p[0], p_lower=p[1], p_upper=p[2], sigma=sigma, check=check)


def pvap_factors(x, x_theory, g_theory, pvap):
    """Per-point multiplicative factors (lower, upper, source) on gamma = g_exp / g_ideal for the acid vapour-pressure
    uncertainty. g_ideal is proportional to p_vap and sits in the denominator, so the factors are p / p_upper and
    p / p_lower (source 'a', from the p_vap columns), else 1 / (1 + rel) and 1 / (1 - rel) with rel = sigma / |g_ideal|
    interpolated in MFS (source 'b'; upper is inf where rel >= 1). None if the file has neither."""
    if pvap is None:
        return None
    x = np.asarray(x, dtype=float)
    if pvap["p"] is not None:
        p, p_lo, p_hi = pvap["p"], min(pvap["p_lower"], pvap["p_upper"]), max(pvap["p_lower"], pvap["p_upper"])
        return np.full(len(x), p / p_hi), np.full(len(x), p / p_lo), "a"
    if pvap["sigma"] is None:
        return None
    ux, idx = np.unique(x_theory, return_index=True)
    gt, st = g_theory[idx], np.abs(pvap["sigma"][idx])
    ok = np.isfinite(st)
    rel = np.full(len(x), np.nan)
    inside = (x >= ux[ok][0]) & (x <= ux[ok][-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        rel[inside] = np.interp(x[inside], ux[ok], st[ok]) / np.abs(np.interp(x[inside], ux, gt))
        upper = np.where(rel >= 1, np.inf, 1 / (1 - rel))
        return 1 / (1 + rel), upper, "b"


def pvap_bounds(gamma, bounds, factors, pvap):
    """p_vap uncertainty on gamma from pvap_factors. lower / upper: gamma times the factors. radius_lower / _upper: the
    radius-uncertainty bounds at the p_vap extremes (None without radius bounds). combined_lower / _upper: per point,
    log-space quadrature of the radius and p_vap relative uncertainties (NaN where gamma <= 0)."""
    f_lo, f_hi, source = factors
    gamma = np.asarray(gamma, dtype=float)
    out = dict(source=source, f_lower=f_lo, f_upper=f_hi, lower=gamma * f_lo, upper=gamma * f_hi,
               radius_lower=None, radius_upper=None,
               p=(pvap["p"], pvap["p_lower"], pvap["p_upper"]), check=pvap["check"])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        d_lo, d_hi = -np.log(f_lo), np.log(f_hi)
        r_lo = r_hi = np.zeros(len(gamma))
        if bounds is not None:
            out["radius_lower"], out["radius_upper"] = bounds[2] * f_lo, bounds[3] * f_hi
            r_lo = np.where(bounds[2] > 0, np.log(gamma / bounds[2]), np.inf)
            r_hi = np.log(bounds[3] / gamma)
        positive = gamma > 0
        out["combined_lower"] = np.where(positive, gamma * np.exp(-np.hypot(r_lo, d_lo)), np.nan)
        out["combined_upper"] = np.where(positive, gamma * np.exp(np.hypot(r_hi, d_hi)), np.nan)
    return out


def _range(values, digits=3):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return "n/a"
    lo, hi = values.min(), values.max()
    return f"{lo:.{digits}f}" if round(lo, digits) == round(hi, digits) else f"{lo:.{digits}f}-{hi:.{digits}f}"


def pvap_note(unc):
    """Status text for pvap_bounds output, e.g. 'p_vap 1.1e-3 Pa (0.8-1.4e-3): gamma x0.786-x1.375, systematic'."""
    if unc["source"] == "a":
        p, p_lo, p_hi = unc["p"]
        e = math.floor(math.log10(p))
        m = lambda v: f"{v / 10 ** e:.3g}"
        text = (f"p_vap {m(p)}e{e} Pa ({m(min(p_lo, p_hi))}-{m(max(p_lo, p_hi))}e{e}): gamma x{p / max(p_lo, p_hi):.3f}"
                f"-x{p / min(p_lo, p_hi):.3f}, systematic (source a: p_vap columns).")
        if unc["check"] is not None:
            text += (f" Bound-run check: bound / nominal gradient within {100 * unc['check']:.2g}% of "
                     f"p_bound / p.")
        return text
    upper = unc["f_upper"]
    text = (f"p_vap from gradient uncertainty: gamma x{_range(unc['f_lower'])}-x{_range(upper)}, systematic "
            f"(source b: no p_vap columns).")
    if np.any(np.isinf(upper)):
        text += f" Upper bound open (infinite) for {int(np.sum(np.isinf(upper)))} point(s): relative uncertainty >= 1."
    missing = int(np.sum(~np.isfinite(unc["f_lower"])))
    if missing:
        text += f" {missing} point(s) outside the gradient-uncertainty range have no p_vap bounds."
    return text


def activity_frame(x, gamma, bounds=None, unc=None):
    """Export table: MFS and gamma, the radius bounds when present, and the p_vap columns when present."""
    columns = {"MFS": x, "Activity coefficient": gamma}
    if bounds is not None:
        columns.update(zip(("MFS min", "MFS max", "Activity coefficient min", "Activity coefficient max"), bounds))
    if unc is not None:
        n, nan = len(x), np.full(len(x), np.nan)
        values = [unc["lower"], unc["upper"],
                  nan if unc["radius_lower"] is None else unc["radius_lower"],
                  nan if unc["radius_upper"] is None else unc["radius_upper"],
                  unc["combined_lower"], unc["combined_upper"]]
        values += [np.full(n, np.nan if p is None else p) for p in unc["p"]]
        columns.update(zip(PVAP_COLUMNS, values))
    return pd.DataFrame(columns)
