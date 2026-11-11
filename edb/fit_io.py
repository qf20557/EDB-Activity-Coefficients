"""Stage 2: reading the fit and cleaned EDB data exported by stage 1."""
import math
import os

import numpy as np
import pandas as pd

from . import config
from .config import MIN_POINTS, RADIUS_UNCERTAINTY_UM
from .raw_data import Columns, is_binary
from .trace import Cancelled

EDB_COLUMNS = ("time_s", "radius_corrected_um", "radius_corrected_err_um", "radius_raw_um")


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


def read_edb_csv(path, progress=None, cancel=None):
    """Cleaned EDB data saved by stage 1: (time, corrected radius, radius uncertainty or None). CSV (header
    required: time_s and radius_corrected_um) or .npz. The CSV is read with the C parser in blocks, keeping only the
    columns needed, so a multi-GB file costs 16 bytes of memory per row; the uncertainty is taken from the first
    block (stage 1 writes the same value on every row). progress(fraction) and cancel() -> bool are optional."""
    if is_binary(path):
        return read_edb_binary(path)
    with open(path, "rb") as handle:
        header = handle.readline().decode("utf-8-sig", "replace")
        handle.seek(0)
        sep = max(",", "\t", ";", "|", key=header.count)
        sep = sep if sep in header else r"\s+"
        names = pd.read_csv(handle, sep=sep, encoding="utf-8-sig", nrows=0).columns
        cols = {str(c).strip().lower(): c for c in names}
        if "time_s" not in cols or "radius_corrected_um" not in cols:
            raise ValueError("This file has no 'time_s' and 'radius_corrected_um' columns. Load cleaned data saved "
                             "by stage 1 ('Save cleaned data'), not a raw EDB file.")
        handle.seek(0)
        size = os.path.getsize(path) or 1
        out, sigma, first = Columns(2), None, True
        for chunk in pd.read_csv(handle, sep=sep, encoding="utf-8-sig", chunksize=config.CHUNK_ROWS,
                                 usecols=[cols[c] for c in EDB_COLUMNS if c in cols]):
            if cancel is not None and cancel():
                raise Cancelled()
            t = pd.to_numeric(chunk[cols["time_s"]], errors="coerce").to_numpy(float)
            r = pd.to_numeric(chunk[cols["radius_corrected_um"]], errors="coerce").to_numpy(float)
            good = np.isfinite(t) & np.isfinite(r) & (r > 0)
            if first:
                sigma = csv_uncertainty(chunk, cols, "radius_corrected_err_um", raw_col="radius_raw_um",
                                        value_col="radius_corrected_um")
                out.reserve(int(len(chunk) / max(handle.tell(), 1) * size * 1.02) + 1024)
                first = False
            out.append(t[good], r[good])
            if progress is not None:
                progress(min(handle.tell() / size, 1.0))
    if not out.n:
        raise ValueError("No valid time / corrected radius rows found in the cleaned CSV.")
    return (*out.finish(), sigma)


def read_edb_binary(path):
    """Cleaned .npz saved by stage 1 (raw_data.write_cleaned)."""
    with np.load(path) as saved:
        if "time_s" not in saved or "radius_raw_um" not in saved:
            raise ValueError("This .npz file is not cleaned data saved by stage 1 ('Save cleaned data').")
        factor = float(saved["radius_correction_factor"]) if "radius_correction_factor" in saved else 1.0
        sigma = float(saved["radius_uncertainty_um"]) if "radius_uncertainty_um" in saved else math.nan
        t, r = saved["time_s"], saved["radius_raw_um"]
    good = np.isfinite(t) & np.isfinite(r) & (r > 0)
    if not good.any():
        raise ValueError("No valid time / radius rows found in the cleaned file.")
    if not good.all():
        t, r = t[good], r[good]
    r *= factor
    return t, r, sigma if math.isfinite(sigma) and sigma > 0 else RADIUS_UNCERTAINTY_UM * factor
