"""Stage 2: reading the fit and cleaned EDB CSVs exported by stage 1."""
import numpy as np
import pandas as pd

from .config import MIN_POINTS, RADIUS_UNCERTAINTY_UM


def csv_uncertainty(df, cols, err_col, factor_col=None, raw_col=None, value_col=None):
    """Corrected-radius uncertainty (um) stored in a stage 1 CSV: the error column if present, otherwise
    RADIUS_UNCERTAINTY_UM times the correction factor (stored, or corrected / raw). None if neither is there."""
    def first(col):
        values = pd.to_numeric(df[cols[col]], errors="coerce").to_numpy(float)
        values = values[np.isfinite(values) & (values > 0)]
        return float(np.median(values)) if values.size else None
    if err_col in cols:
        return first(err_col)
    if factor_col in cols:
        factor = first(factor_col)
        return None if factor is None else RADIUS_UNCERTAINTY_UM * factor
    if raw_col in cols and value_col in cols:
        raw = pd.to_numeric(df[cols[raw_col]], errors="coerce").to_numpy(float)
        corrected = pd.to_numeric(df[cols[value_col]], errors="coerce").to_numpy(float)
        ratio = corrected / raw
        ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        return RADIUS_UNCERTAINTY_UM * float(np.median(ratio)) if ratio.size else None
    return None


def read_fit_csv(path):
    """Return time (s), fitted corrected radius squared (um^2) and the radius uncertainty (um, or None) from a
    fit CSV exported by stage 1. Fits without a corrected-radius column are refused."""
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    cols = {str(c).strip().lower(): c for c in df.columns}
    if "time_s" not in cols:
        raise ValueError("This file has no 'time_s' column. Load a fit CSV exported by stage 1.")
    if "fitted_radius_corrected_squared_um2" in cols:
        r2 = pd.to_numeric(df[cols["fitted_radius_corrected_squared_um2"]], errors="coerce")
    elif "fitted_radius_corrected_um" in cols:
        r2 = pd.to_numeric(df[cols["fitted_radius_corrected_um"]], errors="coerce") ** 2
    else:
        raise ValueError("This fit CSV has no corrected radius column ('fitted_radius_corrected_um'). It was "
                         "probably exported by an older version, on the raw radius. Re-export the fit from "
                         "stage 1 with a compound selected.")
    t = pd.to_numeric(df[cols["time_s"]], errors="coerce")
    sigma = csv_uncertainty(df, cols, "fitted_radius_corrected_err_um", "radius_correction_factor")
    return (*clean_fit_arrays(t.to_numpy(float), r2.to_numpy(float)), sigma)


def clean_fit_arrays(t, r2):
    t, r2 = np.asarray(t, dtype=float), np.asarray(r2, dtype=float)
    good = np.isfinite(t) & np.isfinite(r2) & (r2 > 0)
    t, r2 = t[good], r2[good]
    if len(t) < MIN_POINTS:
        raise ValueError("The fit needs at least three valid rows.")
    order = np.argsort(t, kind="stable")
    t, r2 = t[order], r2[order]
    if np.any(np.diff(t) <= 0):
        raise ValueError("Fit times must be strictly increasing.")
    return t, r2


def read_edb_csv(path):
    """Cleaned EDB CSV saved by stage 1: time_s and radius_corrected_um (header required)."""
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    cols = {str(c).strip().lower(): c for c in df.columns}
    if "time_s" not in cols or "radius_corrected_um" not in cols:
        raise ValueError("This file has no 'time_s' and 'radius_corrected_um' columns. Load a cleaned CSV saved by "
                         "stage 1 ('Save cleaned data'), not a raw EDB file.")
    t = pd.to_numeric(df[cols["time_s"]], errors="coerce").to_numpy(float)
    r = pd.to_numeric(df[cols["radius_corrected_um"]], errors="coerce").to_numpy(float)
    good = np.isfinite(t) & np.isfinite(r) & (r > 0)
    if not good.any():
        raise ValueError("No valid time / corrected radius rows found in the cleaned CSV.")
    sigma = csv_uncertainty(df, cols, "radius_corrected_err_um", raw_col="radius_raw_um",
                            value_col="radius_corrected_um")
    return t[good], r[good], sigma
