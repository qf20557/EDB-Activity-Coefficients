import csv
import json
import math
import re
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import is_color_like
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector, SpanSelector

COMPOUNDS_PATH = Path(__file__).resolve().with_name("compounds.json")

# Stage 1 (EDB Plotter)
RADIUS_UNCERTAINTY_UM = 0.1
MAX_ERROR_BARS = 20000  # error bars are thinned to this many; at that density they overlap into a band anyway
MAX_FIT_BINS = 3000
FIT_EXPORT_POINTS = 1000  # points in the exported fit CSV and in the fit handed to stage 2 (kept identical)
DICT_TAUS = np.geomspace(0.003, 30.0, 70)
EDB_ASSUMED_RI = 1.335
MEDIAN_OBS_ANGLE = math.pi * (58.725 + 31.275) / (2 * 180)

# Stage 2 (EDB Fit Processing)
MIN_POINTS = 3
MAX_BINS = 20000
# columns 3-6: the same quantities with the whole radius trace (fit and initial radius) shifted down / up by the
# radius uncertainty; stage 3 turns these into error bars
CSV_COLUMNS = ["Average MFS", "Gradient (um2/s)", "Average MFS (radius - unc)", "Gradient (radius - unc) (um2/s)",
               "Average MFS (radius + unc)", "Gradient (radius + unc) (um2/s)"]
LIMIT_KEYS = (("X min (s)", "x_min"), ("X max (s)", "x_max"), ("Y min", "y_min"), ("Y max", "y_max"))

# Stage 3 (EDB Activity Coefficients)
AIOMFAC_COLUMN = "a_x"
SAVE_DPI = 1200
matplotlib.rcParams["savefig.dpi"] = SAVE_DPI  # every save (including toolbar save buttons) is 1200 dpi
COLOURS = ("black", "red", "blue", "green", "orange", "purple", "grey", "brown", "teal", "magenta")
MARKERS = {"square": "s", "circle": "o", "triangle": "^", "diamond": "D", "cross (x)": "x", "plus": "+"}
LINE_STYLES = {"solid": "-", "dashed": "--", "dash-dot": "-.", "dotted": ":"}


# ----------------------------------------------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------------------------------------------

def load_compounds(path=COMPOUNDS_PATH):
    """Return {name: record} with numeric density, molar mass and (optional) refractive index."""
    with open(path, encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "compounds" in data:
        data = data["compounds"]
    if isinstance(data, list):
        data = {record["name"]: record for record in data}
    for name, record in data.items():
        for key in ("density_g_ml", "molar_mass_g_mol", "refractive_index"):
            if key not in record:
                if key == "refractive_index":
                    continue
                raise ValueError(f"{name}: missing {key}.")
            value = record[key]
            value = float(value["value"] if isinstance(value, dict) else value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name}: {key} must be positive.")
            record[key] = value
    return data


def scrollable(parent):
    """A vertically scrolling frame. The content fills the window when it fits and scrolls when it does not."""
    outer = ttk.Frame(parent)
    canvas = tk.Canvas(outer, highlightthickness=0)
    bar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=bar.set)
    canvas.pack(side="left", fill="both", expand=True)
    bar.pack(side="right", fill="y")
    content = ttk.Frame(canvas)
    window = canvas.create_window((0, 0), window=content, anchor="nw")

    def fit(_=None):
        height = max(canvas.winfo_height(), content.winfo_reqheight())
        canvas.itemconfigure(window, width=canvas.winfo_width(), height=height)
        canvas.configure(scrollregion=(0, 0, canvas.winfo_width(), height))

    content.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", fit)
    return outer, content


def parse_number(text, label, optional=False):
    text = text.strip()
    if not text:
        if optional:
            return None
        raise ValueError(f"{label} is required.")
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"{label} must be a number.")
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite.")
    return value


def nav_bar(parent, back=None, next_=None, next_text="Next →"):
    bar = ttk.Frame(parent, padding=(6, 4))
    bar.pack(fill="x", side="bottom")
    ttk.Separator(parent, orient="horizontal").pack(fill="x", side="bottom")
    if next_ is not None:
        ttk.Button(bar, text=next_text, command=next_).pack(side="right", padx=4)
    if back is not None:
        ttk.Button(bar, text="← Back", command=back).pack(side="right", padx=4)
    return bar


# ----------------------------------------------------------------------------------------------------------------
# Stage 1 maths: reading, culling, refractive-index correction, monotone convex fit
# ----------------------------------------------------------------------------------------------------------------

def ri_term(angle, ri):
    h = angle / 2.0
    return math.cos(h) + ri * math.sin(h) / math.sqrt(1 + ri ** 2 - 2 * ri * math.cos(h))


def correction_factor(new_ri, previous_ri=EDB_ASSUMED_RI, angle=MEDIAN_OBS_ANGLE):
    return ri_term(angle, previous_ri) / ri_term(angle, new_ri)


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


class PointStore:
    def __init__(self, data):
        self.data = data
        self.keep = np.ones(len(data), dtype=bool)
        self.history = []

    def ids(self):
        return np.flatnonzero(self.keep)

    def active(self):
        rows = self.data[self.ids()]
        return rows[:, 0], rows[:, 1]

    def cull_box(self, box, squared, keep_inside=False, scale=1.0):
        xmin, xmax, ymin, ymax = box
        ids = self.ids()
        t, r = self.data[ids, 0], self.data[ids, 1] * scale
        y = r * r if squared else r
        inside = (t >= xmin) & (t <= xmax) & (y >= ymin) & (y <= ymax)
        removed = ids[~inside if keep_inside else inside]
        if len(removed) == len(ids):
            raise ValueError("That would remove every point.")
        if len(removed):
            self.keep[removed] = False
            self.history.append(removed)
        return len(removed)

    def undo(self):
        if not self.history:
            return False
        self.keep[self.history.pop()] = True
        return True

    def restore(self):
        self.keep[:] = True
        self.history.clear()


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


MODE_HELP = {
    "fit": "Fit region: drag horizontally to fit that time region.",
    "zoom": "Zoom: drag a rectangle to zoom immediately.",
    "cull": "Cull: drag a rectangle to remove all points inside it immediately.",
    "keep": "Keep box: drag a rectangle to remove all points outside it.",
}


# ----------------------------------------------------------------------------------------------------------------
# Stage 2 maths: MFS model and binned gradients
# ----------------------------------------------------------------------------------------------------------------

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


def parse_limits(texts, labels=("X min", "X max", "Y min", "Y max"), positive_y=False):
    values = [parse_number(text, label, optional=True) for label, text in zip(labels, texts)]
    for i, name in ((0, "X"), (2, "Y")):
        if values[i] is not None and values[i + 1] is not None and values[i] >= values[i + 1]:
            raise ValueError(f"{name} min must be below {name} max.")
    if positive_y and any(v is not None and v <= 0 for v in values[2:]):
        raise ValueError("Y limits must be positive (log axis).")
    return values


def apply_limits(ax, lo, hi, which):
    if lo is None and hi is None:
        return
    current = ax.get_xlim() if which == "x" else ax.get_ylim()
    new = (current[0] if lo is None else lo, current[1] if hi is None else hi)
    (ax.set_xlim if which == "x" else ax.set_ylim)(*new)


# ----------------------------------------------------------------------------------------------------------------
# Stage 3 maths: activity coefficients and AIOMFAC
# ----------------------------------------------------------------------------------------------------------------

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


def parse_aiomfac(path):
    with open(path, encoding="utf-8-sig", errors="replace") as handle:
        lines = handle.read().splitlines()
    tables, i = {}, 0
    while i < len(lines):
        match = re.match(r"\s*Mixture's component # :\s*(\d+)", lines[i])
        if not match:
            i += 1
            continue
        number = match.group(1).zfill(2)
        i += 1
        while i < len(lines) and not re.match(r"\s*no\.\s", lines[i]):
            i += 1
        if i >= len(lines):
            break
        names = lines[i].split()
        i += 2
        rows = []
        while i < len(lines) and re.match(r"\s*\d+\s", lines[i]):
            parts = lines[i].split()
            if len(parts) == len(names):
                rows.append(parts)
            i += 1
        if rows:
            tables[number] = pd.DataFrame(rows, columns=names).apply(pd.to_numeric, errors="coerce")
    if not tables:
        raise ValueError("No component tables found. Is this an AIOMFAC-web output file?")
    return tables


def aiomfac_curve(tables, component, quantity=AIOMFAC_COLUMN):
    if component not in tables:
        raise ValueError(f"Component {component} not in file (found: {', '.join(tables)}).")
    df = tables[component]
    x_col, y_col = f"x_i({component})", f"{quantity}({component})"
    if x_col not in df or y_col not in df:
        raise ValueError(f"Columns {x_col} / {y_col} not found.")
    keep = np.isfinite(df[x_col]) & np.isfinite(df[y_col])
    flag = [c for c in df.columns if c.startswith("flag")]
    flagged = 0
    if flag:
        bad = df[flag[0]].fillna(0) != 0
        flagged = int((bad & keep).sum())
        keep &= ~bad
    x, y = df.loc[keep, x_col].to_numpy(float), df.loc[keep, y_col].to_numpy(float)
    order = np.argsort(x)
    return x[order], y[order], flagged


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


def error_bars(ax, x, y, bounds, keep, color, show_x=True, show_y=True, zorder=3):
    """Asymmetric error bars from bounds = (x_lo, x_hi, y_lo, y_hi) for the points selected by keep."""
    if not (show_x or show_y):
        return
    x_lo, x_hi, y_lo, y_hi = (np.asarray(b, dtype=float)[keep] for b in bounds)
    x, y = np.asarray(x, dtype=float)[keep], np.asarray(y, dtype=float)[keep]
    ax.errorbar(x, y, xerr=[x - x_lo, x_hi - x] if show_x else None, yerr=[y - y_lo, y_hi - y] if show_y else None,
                fmt="none", ecolor=color, elinewidth=0.8, capsize=2, zorder=zorder)


def draw_activity(ax, label, points=None, curve=None, limits=None, bounds=None, show_x=True, show_y=True):
    ax.clear()
    if bounds is not None:
        # hidden bars do not widen the automatic limits either
        x_lo, x_hi, y_lo, y_hi = bounds
        x, y = points
        bounds = (x_lo if show_x else x, x_hi if show_x else x, y_lo if show_y else y, y_hi if show_y else y)
    ax.set_yscale("log")
    ax.set_xlabel(rf"$x_{{\mathrm{{{label}}}}}$", fontsize=14)
    ax.set_ylabel(rf"$\gamma_{{\mathrm{{{label}}}}}$", fontsize=14)
    ax.tick_params(direction="in", top=True, right=True, which="both")
    has_points = points is not None and len(points[0]) > 0
    if curve is not None:
        ax.plot(curve[0], curve[1], color="black", linewidth=1.2, label="AIOMFAC")
    if has_points:
        keep = points[1] > 0
        ax.plot(points[0][keep], points[1][keep], "s", color="black", markersize=5, linestyle="none",
                label="Experiment")
        if bounds is not None:
            error_bars(ax, *points, bounds, keep, "black", show_x, show_y)
    if has_points:
        lo, hi = points[0].min(), points[0].max()
        if bounds is not None:
            lo, hi = min(lo, np.nanmin(bounds[0])), max(hi, np.nanmax(bounds[1]))
        pad = 0.05 * max(hi - lo, 0.02)
        xlim = (lo - pad, hi + pad)
    elif curve is not None:
        xlim = (curve[0].min(), curve[0].max())
    else:
        xlim = None
    values = []
    if xlim:
        ax.set_xlim(*xlim)
        if curve is not None:
            inside = (curve[0] >= xlim[0]) & (curve[0] <= xlim[1])
            values.append(curve[1][inside])
        if has_points:
            values.append(points[1][points[1] > 0])
            if bounds is not None:
                values += [bounds[2], bounds[3]]
    values = [v[np.isfinite(v) & (v > 0)] for v in values]
    values = np.concatenate(values) if values else np.array([])
    if values.size:
        ax.set_ylim(10 ** np.floor(np.log10(values.min()) - 0.05), 10 ** np.ceil(np.log10(values.max()) + 0.05))
    xmin, xmax, ymin, ymax = limits or (None, None, None, None)
    apply_limits(ax, xmin, xmax, "x")
    apply_limits(ax, ymin, ymax, "y")
    if curve is not None or has_points:
        ax.legend(frameon=False, loc="upper left")


def first_line(error):
    text = next((line.strip() for line in str(error).splitlines() if line.strip()), "invalid input")
    if "end_group" in str(error) or "Expected" in str(error):
        text += "  (check that every $...$ and { } in your labels is complete)"
    return text[:200]


def _has_data(series):
    return series.get("x") is not None and len(series["x"]) > 0


def auto_axis_limits(series, xscale, yscale, xerr=True, yerr=True):
    """Default (xlim, ylim) from the data. Point series set the x range; lines only if there are no points.
    Shown error bars (xerr / yerr) are included in the range."""
    series = [s for s in series if _has_data(s)]
    points = [s for s in series if s["kind"] == "points"]
    lines = [s for s in series if s["kind"] == "line"]
    source = points or lines
    if not source:
        return None, None
    xs = [np.asarray(s["x"], dtype=float) for s in source]
    if xerr:
        xs += [np.asarray(b, dtype=float) for s in points if s.get("bounds") is not None for b in s["bounds"][:2]]
    xs = np.concatenate(xs)
    xs = xs[np.isfinite(xs)]
    if xscale == "log":
        xs = xs[xs > 0]
    if not xs.size:
        return None, None
    lo, hi = xs.min(), xs.max()
    if xscale == "log":
        xlim = (lo / 10 ** 0.05, hi * 10 ** 0.05) if points else (lo, hi)
    else:
        pad = 0.05 * max(hi - lo, 0.02) if points else 0.0
        xlim = (lo - pad, hi + pad)
    values = []
    for s in lines:
        x, y = np.asarray(s["x"], dtype=float), np.asarray(s["y"], dtype=float)
        values.append(y[(x >= xlim[0]) & (x <= xlim[1])])
    for s in points:
        values.append(np.asarray(s["y"], dtype=float))
        if yerr and s.get("bounds") is not None:
            values += [np.asarray(b, dtype=float) for b in s["bounds"][2:]]
    values = np.concatenate(values) if values else np.array([])
    values = values[np.isfinite(values)]
    if yscale == "log":
        values = values[values > 0]
    if not values.size:
        return xlim, None
    if yscale == "log":
        return xlim, (10 ** np.floor(np.log10(values.min()) - 0.05), 10 ** np.ceil(np.log10(values.max()) + 0.05))
    pad = 0.05 * max(values.max() - values.min(), 1e-12)
    return xlim, (values.min() - pad, values.max() + pad)


def draw_publication(ax, series, o):
    for name, style in o["series"].items():
        if not is_color_like(style["color"]):
            raise ValueError(f"'{style['color']}' is not a valid colour (use a name such as red, or a hex code like #1f77b4).")
    for name, lo, hi, scale in (("X", o["xmin"], o["xmax"], o["xscale"]), ("Y", o["ymin"], o["ymax"], o["yscale"])):
        if lo is not None and hi is not None and lo >= hi:
            raise ValueError(f"{name} min must be below {name} max.")
        if scale == "log" and ((lo is not None and lo <= 0) or (hi is not None and hi <= 0)):
            raise ValueError(f"{name} limits must be positive on a log axis.")
    ax.clear()
    ax.set_xscale(o["xscale"])
    ax.set_yscale(o["yscale"])
    size = o["font_size"]
    for order in ("line", "points"):
        for s in series:
            if s["kind"] != order or not _has_data(s):
                continue
            st = o["series"][s["name"]]
            if order == "line":
                ax.plot(s["x"], s["y"], color=st["color"], linewidth=st["width"], linestyle=st["style"],
                        label=st["label"], zorder=2)
            else:
                ax.plot(s["x"], s["y"], st["marker"], color=st["color"], markersize=st["size"], linestyle="none",
                        label=st["label"], zorder=3)
                if s.get("bounds") is not None:
                    keep = np.asarray(s["y"], dtype=float) > 0 if o["yscale"] == "log" else slice(None)
                    error_bars(ax, s["x"], s["y"], s["bounds"], keep, st["color"], o["xerr"], o["yerr"])
    xlim, ylim = auto_axis_limits(series, o["xscale"], o["yscale"], o["xerr"], o["yerr"])
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    apply_limits(ax, o["xmin"], o["xmax"], "x")
    apply_limits(ax, o["ymin"], o["ymax"], "y")
    ax.set_xlabel(o["xlabel"], fontsize=size)
    ax.set_ylabel(o["ylabel"], fontsize=size)
    if o["title"]:
        ax.set_title(o["title"], fontsize=size + 2)
    ax.tick_params(direction="in", top=True, right=True, which="both", labelsize=size - 1)
    if o["grid"]:
        ax.grid(True, alpha=0.3)
    else:
        ax.grid(False)
    if o["legend"] and ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, loc="best", fontsize=size - 1)


# ----------------------------------------------------------------------------------------------------------------
# Stage 1 UI: Clean & fit (EDB Plotter)
# ----------------------------------------------------------------------------------------------------------------

class CleanFitStage(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.files = []
        self.store = None
        self.fit = None
        self.fit_band = None
        self.var = {name: tk.StringVar(value=value) for name, value in (
            ("stride", "20"), ("fit_lo", ""), ("fit_hi", ""),
            ("x_min", ""), ("x_max", ""), ("y_min", ""), ("y_max", ""))}
        self.squared = tk.BooleanVar(value=False)
        self.use_err = tk.BooleanVar(value=False)
        self.compound = tk.StringVar(value="")
        self.n_true = tk.StringVar(value="")
        self.rho_true = tk.StringVar(value="")
        self.show_raw = tk.BooleanVar(value=True)
        self.show_corr = tk.BooleanVar(value=False)
        self.mode = tk.StringVar(value="fit")
        self.status = tk.StringVar(value="Add raw EDB files or a cleaned CSV (in acquisition order), then click "
                                         "Load / plot.")
        self.fit_text = tk.StringVar(value="Fit: none yet.")
        self.build()

    def build(self):
        nav_bar(self, next_=self.app.to_stage2, next_text="Next: MFS & gradients →")
        outer, content = scrollable(self)
        outer.pack(fill="both", expand=True)

        top = ttk.Frame(content, padding=6)
        top.pack(fill="x")

        files = ttk.LabelFrame(top, text="Files (acquisition order)", padding=6)
        files.pack(side="left", fill="y")
        self.listbox = tk.Listbox(files, height=5, width=32, selectmode=tk.EXTENDED)
        self.listbox.pack(side="left")
        buttons = ttk.Frame(files)
        buttons.pack(side="left", padx=6)
        for text, command in (("Add files...", self.add_files), ("Remove selected", self.remove_files),
                              ("Move up", lambda: self.move(-1)), ("Move down", lambda: self.move(1)),
                              ("Clear", self.clear_files)):
            ttk.Button(buttons, text=text, command=command).pack(fill="x", pady=1)

        opts = ttk.LabelFrame(top, text="Plot and fit", padding=6)
        opts.pack(side="left", fill="both", expand=True, padx=(8, 0))
        ttk.Checkbutton(opts, text="Plot radius² (otherwise radius)", variable=self.squared,
                        command=lambda: self.update_plot(rescale=True)).grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Checkbutton(opts, text="100nm uncertainty (error bars, fit band, CSV error columns)",
                        variable=self.use_err, command=self.update_plot).grid(
            row=1, column=0, columnspan=4, sticky="w")
        ttk.Label(opts, text="Plot every Nth point").grid(row=2, column=0, sticky="w")
        entry = ttk.Entry(opts, textvariable=self.var["stride"], width=7)
        entry.grid(row=2, column=1, sticky="w")
        entry.bind("<Return>", lambda _: self.update_plot())
        ttk.Label(opts, text="Fit from (s)").grid(row=3, column=0, sticky="w")
        ttk.Entry(opts, textvariable=self.var["fit_lo"], width=10).grid(row=3, column=1, sticky="w")
        ttk.Label(opts, text="to (s)").grid(row=3, column=2, sticky="e")
        ttk.Entry(opts, textvariable=self.var["fit_hi"], width=10).grid(row=3, column=3, sticky="w")
        ttk.Label(opts, text="(blank = whole trace; or drag in Fit region mode)").grid(
            row=4, column=0, columnspan=4, sticky="w")

        corr = ttk.LabelFrame(top, text="Radius correction (compounds.json)", padding=6)
        corr.pack(side="left", fill="y", padx=(8, 0))
        ttk.Label(corr, text="Compound").grid(row=0, column=0, sticky="w")
        combo = ttk.Combobox(corr, textvariable=self.compound, values=sorted(self.app.compounds),
                             state="readonly", width=18)
        combo.grid(row=0, column=1, sticky="w")
        combo.bind("<<ComboboxSelected>>", lambda _: self.on_compound())
        ttk.Label(corr, text="Refractive index").grid(row=1, column=0, sticky="w")
        ttk.Entry(corr, textvariable=self.n_true, width=10, state="readonly").grid(row=1, column=1, sticky="w")
        ttk.Label(corr, text="Density (g/mL)").grid(row=2, column=0, sticky="w")
        ttk.Entry(corr, textvariable=self.rho_true, width=10, state="readonly").grid(row=2, column=1, sticky="w")
        ttk.Checkbutton(corr, text="Plot raw radius", variable=self.show_raw,
                        command=self.on_basis_change).grid(row=3, column=0, sticky="w")
        ttk.Checkbutton(corr, text="Plot corrected radius", variable=self.show_corr,
                        command=self.on_basis_change).grid(row=3, column=1, sticky="w")
        self.factor_text = tk.StringVar(value=f"EDB assumed RI {EDB_ASSUMED_RI}; factor 1.0000")
        ttk.Label(corr, textvariable=self.factor_text).grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Label(corr, text="The displayed fit follows the plotted radius;\nstage 2 and exported fits always use\n"
                             "the corrected radius.",
                  foreground="#555").grid(row=5, column=0, columnspan=2, sticky="w")

        act = ttk.Frame(content, padding=(6, 0))
        act.pack(fill="x")
        for text, command in (("Load / plot", self.load), ("Fit", self.do_fit),
                              ("Clear fit region", self.clear_region), ("Save cleaned data", self.save_altered),
                              ("Export fit CSV", self.export_fit)):
            ttk.Button(act, text=text, command=command).pack(side="left", padx=2, pady=3)

        modes = ttk.LabelFrame(content, text="Plot mode (drag on the plot)", padding=4)
        modes.pack(fill="x", padx=6, pady=2)
        for text, value in (("Fit region", "fit"), ("Zoom", "zoom"), ("Cull", "cull"), ("Keep box", "keep")):
            ttk.Radiobutton(modes, text=text, value=value, variable=self.mode,
                            command=self.set_mode).pack(side="left", padx=6)
        for text, command in (("Undo cull", self.undo_cull), ("Restore all", self.restore_points),
                              ("Reset zoom", self.reset_zoom)):
            ttk.Button(modes, text=text, command=command).pack(side="left", padx=3)
        axes = ttk.Frame(modes)
        axes.pack(side="right")
        for label, key in (("X min", "x_min"), ("X max", "x_max"), ("Y min", "y_min"), ("Y max", "y_max")):
            ttk.Label(axes, text=label).pack(side="left", padx=(0, 2))
            box = ttk.Entry(axes, textvariable=self.var[key], width=7)
            box.pack(side="left", padx=(0, 6))
            box.bind("<Return>", lambda _: self.update_plot())
        ttk.Button(axes, text="Apply axes", command=self.update_plot).pack(side="left")

        ttk.Label(content, textvariable=self.status, wraplength=1150).pack(anchor="w", padx=8)
        ttk.Label(content, textvariable=self.fit_text, wraplength=1150, justify="left").pack(anchor="w", padx=8)

        self.figure = Figure(figsize=(8, 4.6))
        self.axis = self.figure.add_subplot(111)
        self.axis.grid(alpha=0.3)
        self.data_line, = self.axis.plot([], [], linestyle="none", marker="x", markersize=4,
                                         markeredgewidth=0.8, color="tab:blue", label="Raw")
        self.corr_line, = self.axis.plot([], [], linestyle="none", marker="x", markersize=4,
                                         markeredgewidth=0.8, color="tab:green", label="Corrected")
        self.err_bars, = self.axis.plot([], [], color="tab:blue", linewidth=0.6, alpha=0.6, zorder=1)
        self.fit_line, = self.axis.plot([], [], color="tab:red", linewidth=1.8, label="Monotone convex fit")
        plot = ttk.Frame(content)
        plot.pack(fill="both", expand=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot, pack_toolbar=False)
        self.toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.span = SpanSelector(self.axis, self.on_span, "horizontal", useblit=True, interactive=False,
                                 props={"alpha": 0.2, "facecolor": "crimson"})
        self.rect = RectangleSelector(self.axis, self.on_box, useblit=True, button=[1], minspanx=3, minspany=3,
                                      spancoords="pixels", interactive=False,
                                      props=dict(facecolor="orange", edgecolor="darkorange", alpha=0.25))
        self.rect.set_active(False)
        self.update_plot(rescale=True)

    # file list
    def refresh_list(self):
        self.listbox.delete(0, tk.END)
        for i, path in enumerate(self.files, 1):
            self.listbox.insert(tk.END, f"{i:02d}. {Path(path).name}")

    def add_files(self):
        for path in filedialog.askopenfilenames(title="Select EDB files or a cleaned CSV", filetypes=[
                ("Data files", "*.txt *.dat *.csv"), ("All files", "*")]):
            if path not in self.files:
                self.files.append(path)
        self.refresh_list()

    def remove_files(self):
        for i in reversed(self.listbox.curselection()):
            del self.files[i]
        self.refresh_list()

    def clear_files(self):
        self.files.clear()
        self.refresh_list()

    def move(self, direction):
        chosen = self.listbox.curselection()
        if len(chosen) != 1:
            return
        i, j = chosen[0], chosen[0] + direction
        if 0 <= j < len(self.files):
            self.files[i], self.files[j] = self.files[j], self.files[i]
            self.refresh_list()
            self.listbox.selection_set(j)

    def load(self):
        try:
            data, skipped = read_files(self.files)
        except Exception as error:
            messagebox.showerror("Load error", str(error), parent=self)
            return
        self.store = PointStore(data)
        self.fit = None
        self.status.set(f"Loaded {len(data):,} points from {len(self.files)} file(s); skipped {skipped:,} "
                        "non-data lines. Radius assumed in µm; time starts at 0 s.")
        self.fit_text.set("Fit: none yet. Drag in Fit region mode or click Fit for the whole trace.")
        self.update_plot(rescale=True)

    # modes and correction
    def set_mode(self):
        mode = self.mode.get()
        text = str(self.toolbar.mode).lower()
        if "zoom" in text:
            self.toolbar.zoom()
        elif "pan" in text:
            self.toolbar.pan()
        self.span.set_active(mode == "fit")
        self.rect.set_active(mode != "fit")
        self.status.set(MODE_HELP[mode])

    def on_compound(self):
        info = self.app.compounds.get(self.compound.get(), {})
        n, rho = info.get("refractive_index"), info.get("density_g_ml")
        self.n_true.set("" if n is None else f"{n:g}")
        self.rho_true.set("" if rho is None else f"{rho:g}")
        if n is None:
            self.status.set(f"No refractive_index for {self.compound.get()} in compounds.json; factor = 1.")
        self.on_basis_change()

    def on_basis_change(self):
        """The radius basis (raw or corrected) changed: refit so the fit always matches what is shown."""
        if self.store is not None and self.fit is not None:
            self.do_fit(redraw=False)
        self.update_plot(rescale=True)

    def factor(self):
        try:
            n = float(self.n_true.get())
            f = correction_factor(n) if n > 0 else 1.0
        except ValueError:
            f = 1.0
        self.factor_text.set(f"EDB assumed RI {EDB_ASSUMED_RI}; factor {f:.4f}")
        return f

    def fit_scale(self):
        return self.factor() if self.show_corr.get() else 1.0

    def require_correction(self, action):
        """Correction factor for outputs that must be on the corrected radius, or None if the user backs out."""
        name = self.compound.get()
        if name not in self.app.compounds:
            messagebox.showinfo("No compound", f"Choose the droplet compound (radius correction) before {action}: "
                                "stage 2 and the exported files always use the corrected radius.", parent=self)
            return None
        if self.app.compounds[name].get("refractive_index") is None and not messagebox.askyesno(
                "No refractive index", f"{name} has no refractive_index in compounds.json, so the corrected "
                "radius equals the raw radius (factor 1). Continue?", parent=self):
            return None
        return self.factor()

    def corrected_fit(self):
        """The current fit on the corrected radius: the displayed fit if it already is, otherwise a refit of the
        same retained points over the same time window."""
        factor = self.factor()
        if self.fit["scale"] == factor:
            return self.fit
        t, r = self.store.active()
        fit = fit_model(t, r * factor, self.fit["lo"], self.fit["hi"])
        fit["scale"] = factor
        return fit

    # plot interaction
    def on_span(self, lo, hi):
        if self.store is None or lo == hi:
            return
        lo, hi = sorted((lo, hi))
        self.var["fit_lo"].set(f"{lo:.8g}")
        self.var["fit_hi"].set(f"{hi:.8g}")
        self.do_fit()

    def on_box(self, press, release):
        if self.store is None or None in (press.xdata, press.ydata, release.xdata, release.ydata):
            return
        xmin, xmax = sorted((press.xdata, release.xdata))
        ymin, ymax = sorted((press.ydata, release.ydata))
        mode = self.mode.get()
        if mode == "zoom":
            self.axis.set_xlim(xmin, xmax)
            self.axis.set_ylim(ymin, ymax)
        elif mode in ("cull", "keep"):
            try:
                removed = self.store.cull_box((xmin, xmax, ymin, ymax), self.squared.get(), mode == "keep",
                                              self.cull_scale())
            except ValueError as error:
                messagebox.showinfo("Cull", str(error), parent=self)
                removed = 0
            if removed:
                self.after_edit()
        self.rect.clear()
        self.canvas.draw_idle()

    def cull_scale(self):
        # boxes are drawn on whichever series is visible; when both are shown the raw series sets the scale
        return self.factor() if self.show_corr.get() and not self.show_raw.get() else 1.0

    def after_edit(self):
        total, kept = len(self.store.data), int(self.store.keep.sum())
        self.status.set(f"{kept:,} retained / {total:,} loaded points; {total - kept:,} culled. "
                        "Source files unchanged.")
        if self.fit is not None:
            self.do_fit(redraw=False)
        self.update_plot()

    def undo_cull(self):
        if self.store is not None and self.store.undo():
            self.after_edit()

    def restore_points(self):
        if self.store is not None:
            self.store.restore()
            self.after_edit()

    def reset_zoom(self):
        self.update_plot(rescale=True)

    # outputs
    def save_altered(self):
        if self.store is None:
            messagebox.showinfo("No data", "Load data first.", parent=self)
            return
        if self.require_correction("saving") is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                            initialfile="edb_cleaned.csv", parent=self)
        if not path:
            return
        if any(Path(path).exists() and Path(path).samefile(p) for p in self.files):
            messagebox.showerror("Input selected", "Choose an output file different from the inputs.", parent=self)
            return
        try:
            count, factor = self.write_cleaned(path)
        except OSError as error:
            messagebox.showerror("Save failed", str(error), parent=self)
            return
        self.status.set(f"Saved {count:,} retained rows (correction factor {factor:.4f}).")

    def write_cleaned(self, path):
        t, r = self.store.active()
        factor = self.factor()
        rc = r * factor
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if self.use_err.get():
                sigma = RADIUS_UNCERTAINTY_UM * factor
                writer.writerow(["time_s", "radius_raw_um", "radius_corrected_um",
                                 "radius_corrected_err_um", "radius_corrected_squared_um2",
                                 "radius_corrected_squared_err_um2"])
                writer.writerows(zip(t, r, rc, np.full_like(rc, sigma), rc * rc, 2 * rc * sigma))
            else:
                writer.writerow(["time_s", "radius_raw_um", "radius_corrected_um"])
                writer.writerows(zip(t, r, rc))
        return len(t), factor

    def number(self, key, label):
        return parse_number(self.var[key].get(), label, optional=True)

    def clear_region(self):
        self.var["fit_lo"].set("")
        self.var["fit_hi"].set("")
        self.do_fit()

    def do_fit(self, redraw=True):
        if self.store is None:
            self.fit_text.set("Load data first.")
            return
        try:
            scale = self.fit_scale()
            t, r = self.store.active()
            self.fit = fit_model(t, r * scale, self.number("fit_lo", "Fit from"), self.number("fit_hi", "Fit to"))
            self.fit["scale"] = scale
        except ValueError as error:
            self.fit = None
            self.fit_text.set(f"Fit failed: {error}")
        else:
            f = self.fit
            basis = "corrected radius" if scale != 1.0 else "radius"
            self.fit_text.set(
                f"{f['label']} to {basis}, {f['lo']:.6g}-{f['hi']:.6g} s, {f['n']:,} retained points. "
                f"RMS residual {f['rms']:.4g} µm, R² {f['r2']:.6f}. {f['detail']}")
        if redraw:
            self.update_plot()

    def export_fit(self):
        if self.fit is None:
            messagebox.showinfo("No fit", "Run a fit first.", parent=self)
            return
        if self.require_correction("exporting the fit") is None:
            return
        try:
            fit = self.corrected_fit()
        except ValueError as error:
            messagebox.showerror("Fit error", f"Could not fit the corrected radius: {error}", parent=self)
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                            initialfile="edb_fit.csv", parent=self)
        if not path:
            return
        try:
            self.write_fit(path, fit)
        except OSError as error:
            messagebox.showerror("Export error", str(error), parent=self)
            return
        self.status.set(f"Corrected-radius fit exported ({FIT_EXPORT_POINTS} points, factor {fit['scale']:.4f}).")

    def write_fit(self, path, fit):
        x, radius, radius2 = fit_curve(fit, FIT_EXPORT_POINTS)
        header = ["time_s", "fitted_radius_corrected_um", "fitted_radius_corrected_squared_um2",
                  "radius_correction_factor"]
        columns = [x, radius, radius2, np.full_like(x, fit["scale"])]
        if self.use_err.get():
            sigma = RADIUS_UNCERTAINTY_UM * fit["scale"]
            header += ["fitted_radius_corrected_err_um", "fitted_radius_corrected_lower_um",
                       "fitted_radius_corrected_upper_um", "fitted_radius_corrected_squared_err_um2",
                       "fitted_radius_corrected_squared_lower_um2", "fitted_radius_corrected_squared_upper_um2"]
            columns += [np.full_like(radius, sigma), radius - sigma, radius + sigma, 2 * radius * sigma,
                        (radius - sigma) ** 2, (radius + sigma) ** 2]
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(zip(*columns))

    def handoff(self):
        """Data for stage 2: cleaned points and fit, both always on the corrected radius."""
        if self.store is None:
            messagebox.showinfo("No data", "Load EDB data first.", parent=self)
            return None
        if self.fit is None:
            messagebox.showinfo("No fit", "Fit the data before moving on (click Fit or drag a fit region).",
                                parent=self)
            return None
        factor = self.require_correction("moving on")
        if factor is None:
            return None
        try:
            fit = self.corrected_fit()
        except ValueError as error:
            messagebox.showerror("Fit error", f"Could not fit the corrected radius: {error}", parent=self)
            return None
        t, r = self.store.active()
        x, _, radius2 = fit_curve(fit, FIT_EXPORT_POINTS)
        return dict(edb_t=t.copy(), edb_r=r * factor, fit_t=x, fit_r2=radius2, factor=factor,
                    sigma=RADIUS_UNCERTAINTY_UM * factor, refitted=fit is not self.fit)

    def update_plot(self, rescale=False):
        ax = self.axis
        squared = self.squared.get()
        show_err = self.use_err.get()
        if not (self.show_raw.get() or self.show_corr.get()):
            self.show_raw.set(True)
        factor = self.factor()
        scale = self.fit_scale()
        ax.set_xlabel("Time since first point (s)")
        name = "Corrected radius" if self.show_corr.get() and not self.show_raw.get() else "Radius"
        ax.set_ylabel(f"{name}² (µm²)" if squared else f"{name} (µm)")
        try:
            stride = int(self.var["stride"].get())
            if stride < 1:
                raise ValueError
        except ValueError:
            stride = 1
            self.status.set("Plot every Nth point must be a positive whole number; using 1.")
        if self.store is not None:
            t, r = self.store.active()
            for line, s, on in ((self.data_line, 1.0, self.show_raw.get()),
                                (self.corr_line, factor, self.show_corr.get())):
                if on:
                    y = (r * s) ** 2 if squared else r * s
                    line.set_data(t[::stride], y[::stride])
                else:
                    line.set_data([], [])
            if show_err:
                step = max(1, len(t[::stride]) // MAX_ERROR_BARS) * stride
                ts, rs = t[::step], r[::step] * scale
                sigma = RADIUS_UNCERTAINTY_UM * scale
                lower, upper = np.clip(rs - sigma, 0, None), rs + sigma
                if squared:
                    lower, upper = lower ** 2, upper ** 2
                x_bars = np.repeat(ts, 3)
                y_bars = np.empty(3 * len(ts))
                y_bars[0::3], y_bars[1::3], y_bars[2::3] = lower, upper, np.nan
                x_bars[2::3] = np.nan
                self.err_bars.set_data(x_bars, y_bars)
            else:
                self.err_bars.set_data([], [])
            ax.set_title(f"EDB evaporation — {len(t):,} retained; "
                         f"{len(self.store.data) - len(t):,} culled; {len(t[::stride]):,} plotted")
        else:
            for line in (self.data_line, self.corr_line, self.err_bars):
                line.set_data([], [])
        if self.fit_band is not None:
            self.fit_band.remove()
            self.fit_band = None
        if self.fit is not None:
            x, radius, radius2 = fit_curve(self.fit)
            self.fit_line.set_data(x, radius2 if squared else radius)
            if show_err:
                sigma = RADIUS_UNCERTAINTY_UM * self.fit["scale"]
                lower, upper = np.clip(radius - sigma, 0, None), radius + sigma
                if squared:
                    lower, upper = lower ** 2, upper ** 2
                self.fit_band = ax.fill_between(x, lower, upper, color="tab:red", alpha=0.25, linewidth=0, zorder=2)
        else:
            self.fit_line.set_data([], [])
        if rescale:
            ax.autoscale(enable=True)
            ax.relim()
            ax.autoscale_view()
        try:
            b = parse_limits([self.var[k].get() for k in ("x_min", "x_max", "y_min", "y_max")])
            apply_limits(ax, b[0], b[1], "x")
            apply_limits(ax, b[2], b[3], "y")
        except ValueError as error:
            self.status.set(f"Axis limits ignored: {error}")
        ax.legend(loc="best", fontsize=8)
        self.canvas.draw_idle()


# ----------------------------------------------------------------------------------------------------------------
# Stage 2 UI: MFS & gradients (EDB Fit Processing)
# ----------------------------------------------------------------------------------------------------------------

class MfsStage(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.t = self.r2 = None
        self.edb_t = self.edb_r2 = None
        self.mfs = self.mfs_minus = self.mfs_plus = None
        self.sigma = 0.0
        self.rows = []
        self.edge_lines = []
        self.band = None
        self.v = {}
        self.source = tk.StringVar(value="No data yet: press Next on stage 1, or load a fit CSV and a cleaned "
                                         "EDB CSV.")

        nav_bar(self, back=lambda: self.app.select(0), next_=self.app.to_stage3,
                next_text="Next: activity coefficients →")
        outer, root = scrollable(self)
        outer.pack(fill="both", expand=True)

        bar = ttk.Frame(root)
        bar.pack(fill="x", padx=4, pady=4)
        for label, command in (("Load fit CSV", self.load_fit), ("Load cleaned EDB CSV", self.load_raw),
                               ("Calculate intervals", self.calculate), ("Export CSV", self.export),
                               ("Clear results", self.clear)):
            ttk.Button(bar, text=label, command=command).pack(side="left", padx=2)
        ttk.Label(root, textvariable=self.source, foreground="#555", wraplength=1150).pack(anchor="w", padx=6)

        panel = ttk.LabelFrame(root, text="Initial (wet) droplet composition", padding=6)
        panel.pack(fill="x", padx=4)
        self.boxes = []
        for col, (key, default_mass) in enumerate((("A: evaporating", "1.04"), ("B: conserved", "1.36"),
                                                   ("Solvent", "12.0"))):
            ttk.Label(panel, text=key).grid(row=0, column=2 * col, sticky="w", padx=3)
            var = tk.StringVar()
            self.v[key] = var
            box = ttk.Combobox(panel, textvariable=var, state="readonly", width=20)
            box.grid(row=0, column=2 * col + 1, padx=3, pady=2)
            box.bind("<<ComboboxSelected>>", lambda _: self.update_model())
            self.boxes.append(box)
            ttk.Label(panel, text=f"Mass {key.split(':')[0]} (g)").grid(row=1, column=2 * col, sticky="w", padx=3)
            mass = tk.StringVar(value=default_mass)
            self.v["mass" + key] = mass
            ttk.Entry(panel, textvariable=mass, width=22).grid(row=1, column=2 * col + 1, padx=3, pady=2)
        self.field(panel, 2, 0, "Initial radius (um)", "")
        self.field(panel, 2, 1, "Interval start (s)", "")
        self.field(panel, 2, 2, "Interval end (s)", "")
        self.field(panel, 3, 0, "Bin width (s, 0 = one interval)", "1000")
        self.field(panel, 3, 1, "Plot every Nth EDB point", "20")
        self.field(panel, 3, 2, "Radius uncertainty (um)", f"{RADIUS_UNCERTAINTY_UM:g}").bind(
            "<Return>", lambda _: self.update_model())
        r0 = ttk.Frame(panel)
        r0.grid(row=4, column=0, columnspan=6, sticky="w")
        ttk.Button(r0, text="r₀ = first EDB point", command=self.r0_from_edb).pack(side="left", padx=2)
        ttk.Button(r0, text="r₀ = fit at start", command=self.r0_from_fit).pack(side="left", padx=2)
        ttk.Label(r0, text="Uncertainty (corrected radius, 0 = off) shifts the whole fit and r₀ down / up together; "
                           "the spread gives the error bars in stage 3.", foreground="#555").pack(side="left", padx=8)

        limits = ttk.LabelFrame(root, text="Plot axis limits (blank = automatic)", padding=6)
        limits.pack(fill="x", padx=4, pady=(4, 0))
        self.limit_vars = []
        for label, _ in LIMIT_KEYS:
            ttk.Label(limits, text=label).pack(side="left", padx=(0, 2))
            var = tk.StringVar()
            entry = ttk.Entry(limits, textvariable=var, width=10)
            entry.pack(side="left", padx=(0, 8))
            entry.bind("<Return>", lambda _: self.draw(rescale=False))
            self.limit_vars.append(var)
        ttk.Button(limits, text="Apply", command=lambda: self.draw(rescale=False)).pack(side="left", padx=2)
        ttk.Button(limits, text="Auto", command=self.auto_limits).pack(side="left", padx=2)

        self.status = tk.StringVar(value="")
        self.info = tk.StringVar(value="")
        ttk.Label(root, textvariable=self.status, wraplength=1150).pack(anchor="w", padx=6, pady=(4, 0))
        ttk.Label(root, textvariable=self.info, wraplength=1150).pack(anchor="w", padx=6)

        modes = ttk.Frame(root)
        modes.pack(fill="x", padx=6)
        self.mode = tk.StringVar(value="select")
        for text, value in (("Select interval (drag)", "select"), ("Zoom (drag)", "zoom")):
            ttk.Radiobutton(modes, text=text, value=value, variable=self.mode,
                            command=self.set_mode).pack(side="left", padx=6)
        ttk.Button(modes, text="Reset view", command=self.reset_view).pack(side="left", padx=6)

        self.fig = Figure(figsize=(9, 4.2))
        self.ax = self.fig.add_subplot(111)
        self.line_r2, = self.ax.plot([], [], color="tab:red", linewidth=2.2, zorder=2, label="Fit")
        self.line_edb, = self.ax.plot([], [], linestyle="none", marker="x", markersize=3, markeredgewidth=0.6,
                                      color="tab:blue", alpha=0.7, zorder=3, label="EDB data (cleaned)")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Radius² (µm²)")
        self.ax.grid(alpha=0.25)
        plot = ttk.Frame(root)
        plot.pack(fill="both", expand=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot, pack_toolbar=False)
        self.toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.span = SpanSelector(self.ax, self.on_span, "horizontal", useblit=True, interactive=False,
                                 props={"alpha": 0.2, "facecolor": "orange"})
        self.stride_entry.bind("<Return>", lambda _: self.draw(rescale=False))

        frame = ttk.Frame(root)
        frame.pack(fill="x", padx=4, pady=4)
        self.table = ttk.Treeview(frame, columns=CSV_COLUMNS, show="headings", height=6)
        for column in CSV_COLUMNS:
            self.table.heading(column, text=column)
            self.table.column(column, width=190)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.pack(side="left", fill="x", expand=True)
        scroll.pack(side="left", fill="y")
        self.set_compounds()
        self.draw(rescale=True)

    def field(self, panel, row, col, key, default):
        ttk.Label(panel, text=key).grid(row=row, column=2 * col, sticky="w", padx=3)
        var = tk.StringVar(value=default)
        self.v[key] = var
        entry = ttk.Entry(panel, textvariable=var, width=22)
        entry.grid(row=row, column=2 * col + 1, padx=3, pady=2)
        if key.startswith("Plot every"):
            self.stride_entry = entry
        return entry

    def get(self, key):
        return self.v[key].get().strip()

    def num(self, key, label):
        return parse_number(self.get(key), label)

    def set_compounds(self):
        names = list(self.app.compounds)
        if not names:
            self.info.set(f"No compounds loaded: place a valid {COMPOUNDS_PATH.name} beside this script.")
            return
        defaults = {"A: evaporating": "Malonic acid", "B: conserved": "Tween 20", "Solvent": "Water"}
        for box, key in zip(self.boxes, ("A: evaporating", "B: conserved", "Solvent")):
            box["values"] = names
            self.v[key].set(defaults[key] if defaults[key] in names else names[0])

    # data in
    def receive(self, data):
        """Data handed over from stage 1, always on the corrected radius."""
        self.t, self.r2 = clean_fit_arrays(data["fit_t"], data["fit_r2"])
        self.edb_t, self.edb_r2 = data["edb_t"], data["edb_r"] ** 2
        self.after_fit_loaded()
        self.v["Initial radius (um)"].set(f"{data['edb_r'][0]:.8g}")
        self.v["Radius uncertainty (um)"].set(f"{data['sigma']:.6g}")
        refit = " (refitted on the corrected radius, same window)" if data["refitted"] else ""
        self.source.set(f"From stage 1: {len(self.edb_t):,} cleaned points and a {len(self.t):,}-point fit{refit}, "
                        f"refractive-index corrected radius (factor {data['factor']:.4f}), uncertainty "
                        f"±{data['sigma']:.4g} um. Initial radius set from the first cleaned point (editable).")
        self.clear()
        self.update_model(rescale=True)

    def after_fit_loaded(self):
        self.v["Interval start (s)"].set(f"{self.t[0]:.8g}")
        self.v["Interval end (s)"].set(f"{self.t[-1]:.8g}")
        self.status.set(f"Fit: {len(self.t):,} points, {self.t[0]:.6g}-{self.t[-1]:.6g} s.")

    def load_fit(self):
        path = filedialog.askopenfilename(title="Fit CSV", filetypes=[("CSV", "*.csv"), ("All files", "*")],
                                          parent=self)
        if not path:
            return
        try:
            self.t, self.r2, sigma = read_fit_csv(path)
        except Exception as error:
            messagebox.showerror("Fit CSV error", str(error), parent=self)
            return
        self.after_fit_loaded()
        self.source.set(f"Corrected-radius fit loaded from {Path(path).name}." + self.take_sigma(sigma))
        self.clear()
        self.update_model(rescale=True)

    def load_raw(self):
        path = filedialog.askopenfilename(title="Cleaned EDB CSV", filetypes=[("CSV", "*.csv"), ("All files", "*")],
                                          parent=self)
        if not path:
            return
        try:
            t, r, sigma = read_edb_csv(path)
        except Exception as error:
            messagebox.showerror("EDB CSV error", str(error), parent=self)
            return
        self.edb_t, self.edb_r2 = t, r * r
        self.v["Initial radius (um)"].set(f"{r[0]:.8g}")
        self.status.set(f"EDB file: {len(t):,} points. First point t = {t[0]:.6g} s, corrected radius = "
                        f"{r[0]:.6g} um (set as initial radius; editable). Read from radius_corrected_um saved by "
                        "stage 1." + self.take_sigma(sigma))
        self.clear()
        self.update_model(rescale=True)

    def take_sigma(self, sigma):
        """Use the radius uncertainty stored in a loaded CSV; returns a note for the status line."""
        if sigma is None:
            return f" No uncertainty stored; keeping ±{self.get('Radius uncertainty (um)')} um."
        self.v["Radius uncertainty (um)"].set(f"{sigma:.6g}")
        return f" Radius uncertainty ±{sigma:.4g} um taken from the file."

    def r0_from_edb(self):
        if self.edb_r2 is None:
            messagebox.showinfo("No EDB data", "No cleaned EDB data loaded.", parent=self)
            return
        self.v["Initial radius (um)"].set(f"{math.sqrt(self.edb_r2[0]):.8g}")
        self.update_model()

    def r0_from_fit(self):
        if self.t is None:
            messagebox.showinfo("No fit", "No fit loaded.", parent=self)
            return
        self.v["Initial radius (um)"].set(f"{math.sqrt(self.r2[0]):.8g}")
        self.update_model()

    def update_model(self, rescale=False):
        self.mfs = self.mfs_minus = self.mfs_plus = None
        try:
            args = (self.app.compounds, self.get("A: evaporating"), self.get("B: conserved"), self.get("Solvent"),
                    self.num("massA: evaporating", "Mass A"), self.num("massB: conserved", "Mass B"),
                    self.num("massSolvent", "Solvent mass"))
            r0 = self.num("Initial radius (um)", "Initial radius")
            sigma = self.num("Radius uncertainty (um)", "Radius uncertainty")
            if not 0 <= sigma < r0:
                raise ValueError("Radius uncertainty must be zero or positive and below the initial radius.")
            mfs, info = make_mfs_function(*args, r0)
            if sigma:
                # a radius offset applies to every measured radius, including r0
                self.mfs_minus = make_mfs_function(*args, r0 - sigma)[0]
                self.mfs_plus = make_mfs_function(*args, r0 + sigma)[0]
            self.mfs, self.sigma = mfs, sigma
            self.info.set(f"Initial dry x(A) = {info['x_a0']:.6g}; calculated dry (A+B) radius = "
                          f"{info['r_dry']:.6g} um. Below this radius solvent is gone and A is lost. "
                          + (f"Radius uncertainty ±{sigma:.4g} um propagated." if sigma else
                             "No radius uncertainty propagated."))
        except ValueError as error:
            self.info.set(f"Composition: {error}")
        self.draw(rescale=rescale)

    def scenarios(self):
        """(r2, mfs) for the nominal fit, then with the radius shifted down and up by the uncertainty."""
        if not self.sigma:
            return [(self.r2, self.mfs)] * 3
        r = np.sqrt(self.r2)
        lower = np.where(r > self.sigma, (r - self.sigma) ** 2, np.nan)
        return [(self.r2, self.mfs), (lower, self.mfs_minus), ((r + self.sigma) ** 2, self.mfs_plus)]

    def set_mode(self):
        text = str(self.toolbar.mode).lower()
        if "zoom" in text:
            self.toolbar.zoom()
        elif "pan" in text:
            self.toolbar.pan()
        zoom = self.mode.get() == "zoom"
        self.span.set_active(not zoom)
        if zoom:
            self.toolbar.zoom()

    def auto_limits(self):
        for var in self.limit_vars:
            var.set("")
        self.draw(rescale=True)

    def reset_view(self):
        self.draw(rescale=True)

    def on_span(self, lo, hi):
        if self.t is None or lo == hi:
            return
        lo, hi = sorted((lo, hi))
        self.v["Interval start (s)"].set(f"{max(lo, self.t[0]):.8g}")
        self.v["Interval end (s)"].set(f"{min(hi, self.t[-1]):.8g}")
        self.draw(rescale=False)

    def draw(self, rescale=False):
        ax = self.ax
        try:
            stride = int(self.get("Plot every Nth EDB point"))
            if stride < 1:
                raise ValueError
        except ValueError:
            stride = 1
            self.status.set("Plot every Nth point must be a positive whole number; using 1.")
        if self.edb_t is not None:
            self.line_edb.set_data(self.edb_t[::stride], self.edb_r2[::stride])
        else:
            self.line_edb.set_data([], [])
        if self.band is not None:
            self.band.remove()
            self.band = None
        if self.t is not None:
            self.line_r2.set_data(self.t, self.r2)
            if self.sigma:
                r = np.sqrt(self.r2)
                self.band = ax.fill_between(self.t, np.clip(r - self.sigma, 0, None) ** 2, (r + self.sigma) ** 2,
                                            color="tab:red", alpha=0.2, linewidth=0, zorder=1,
                                            label=f"Fit ± {self.sigma:.3g} µm radius")
        else:
            self.line_r2.set_data([], [])
        for line in self.edge_lines:
            line.remove()
        self.edge_lines = []
        if self.t is not None:
            try:
                lo, hi = self.num("Interval start (s)", "start"), self.num("Interval end (s)", "end")
                width = self.num("Bin width (s, 0 = one interval)", "bin width")
                self.edge_lines = [ax.axvline(e, color="orange", alpha=0.4, linewidth=0.8, zorder=1)
                                   for e in bin_edges(lo, hi, width)]
            except ValueError:
                pass
        if rescale:
            ax.autoscale(True)
            ax.relim()
            ax.autoscale_view()
        try:
            b = parse_limits([var.get() for var in self.limit_vars], [label for label, _ in LIMIT_KEYS])
            apply_limits(ax, b[0], b[1], "x")
            apply_limits(ax, b[2], b[3], "y")
        except ValueError as error:
            self.status.set(f"Axis limits ignored: {error}")
        if self.t is not None or self.edb_t is not None:
            ax.legend(loc="best", fontsize=8)
        self.canvas.draw_idle()

    def calculate(self, quiet=False):
        if self.t is None:
            messagebox.showinfo("No fit", "Press Next on stage 1, or load a fit CSV.", parent=self)
            return False
        self.update_model()
        if self.mfs is None:
            messagebox.showerror("Composition error", self.info.get(), parent=self)
            return False
        try:
            lo = self.num("Interval start (s)", "Interval start")
            hi = self.num("Interval end (s)", "Interval end")
            width = self.num("Bin width (s, 0 = one interval)", "Bin width")
            lo, hi = max(lo, self.t[0]), min(hi, self.t[-1])
            rows, skipped = interval_results(self.t, self.scenarios(), lo, hi, width)
        except ValueError as error:
            messagebox.showerror("Calculation error", str(error), parent=self)
            return False
        self.clear()
        self.rows = rows
        for row in rows:
            self.table.insert("", "end", values=tuple("" if math.isnan(v) else f"{v:.6g}" for v in row))
        message = f"{len(rows)} interval(s) calculated"
        message += f" with ±{self.sigma:.4g} um radius uncertainty." if self.sigma else " (no radius uncertainty)."
        if skipped:
            message += (f" {skipped} skipped (fewer than {MIN_POINTS} fit points, or A would be negative "
                        "i.e. radius below the B-only radius).")
        partial = sum(any(math.isnan(v) for v in row) for row in rows)
        if partial:
            message += f" {partial} interval(s) have no value for a shifted radius (A would be negative there)."
        self.status.set(message)
        return True

    def clear(self):
        self.rows = []
        for item in self.table.get_children():
            self.table.delete(item)

    def export(self):
        if not self.rows:
            messagebox.showinfo("Nothing to export", "Calculate intervals first.", parent=self)
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                            initialfile="edb_intervals.csv", parent=self)
        if not path:
            return
        try:
            self.write_rows(path)
        except OSError as error:
            messagebox.showerror("Export error", str(error), parent=self)
            return
        self.status.set(f"Exported {len(self.rows)} row(s).")

    def write_rows(self, path):
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_COLUMNS)
            writer.writerows(self.rows)

    def handoff(self):
        if not self.rows:
            if not messagebox.askyesno("No intervals", "No intervals calculated yet. Calculate them now with the "
                                       "current settings?", parent=self):
                return None
            if not self.calculate() or not self.rows:
                return None
        rows = np.asarray(self.rows, dtype=float)
        return sort_mfs_gradient(rows[:, 0], rows[:, 1], rows[:, 2:])


# ----------------------------------------------------------------------------------------------------------------
# Stage 3 UI: Activity coefficients and publication plots (EDB Activity Coefficients)
# ----------------------------------------------------------------------------------------------------------------

class PlotStudio(tk.Toplevel):
    def __init__(self, master, series, xlabel, ylabel, filename="figure.png", show_xerr=True, show_yerr=True):
        super().__init__(master)
        self.title("Publication plot")
        self.geometry("1280x900")
        self.series = series
        self.filename = filename
        self.last_good = None
        self.pending = None
        self.v = {}

        form = ttk.Frame(self, padding=8)
        form.pack(side="left", fill="y")
        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True)
        row = [0]

        def header(text):
            ttk.Label(form, text=text, font=("TkDefaultFont", 10, "bold")).grid(
                row=row[0], column=0, columnspan=2, sticky="w", pady=(8, 2))
            row[0] += 1

        def entry(key, text, default, values=None, width=24):
            ttk.Label(form, text=text).grid(row=row[0], column=0, sticky="w", pady=1)
            var = tk.StringVar(value=str(default))
            widget = (ttk.Combobox(form, textvariable=var, values=values, width=width - 3) if values
                      else ttk.Entry(form, textvariable=var, width=width))
            widget.grid(row=row[0], column=1, sticky="w", padx=4, pady=1)
            var.trace_add("write", lambda *_: self.schedule())
            self.v[key] = var
            row[0] += 1

        header("Text (use $...$ for maths, e.g. $\\gamma_{\\mathrm{A}}$)")
        entry("title", "Title", "")
        entry("xlabel", "X label", xlabel)
        entry("ylabel", "Y label", ylabel)
        entry("font_size", "Font size (pt)", "12")
        header("Axes (limits blank = automatic)")
        entry("xscale", "X axis", "linear", ("linear", "log"))
        entry("yscale", "Y axis", "log", ("linear", "log"))
        entry("xmin", "X min", "")
        entry("xmax", "X max", "")
        entry("ymin", "Y min", "")
        entry("ymax", "Y max", "")
        for s in series:
            header(f"{s['name']} ({'curve' if s['kind'] == 'line' else 'points'})")
            entry(s["name"] + ".label", "Legend label", s["label"])
            entry(s["name"] + ".color", "Colour (name or #hex)", s["color"], COLOURS)
            if s["kind"] == "line":
                entry(s["name"] + ".style", "Line style", s.get("style", "solid"), tuple(LINE_STYLES))
                entry(s["name"] + ".width", "Line width", s.get("width", 1.2))
            else:
                entry(s["name"] + ".marker", "Marker", s.get("marker", "square"), tuple(MARKERS))
                entry(s["name"] + ".size", "Marker size", s.get("size", 5))
        header("Figure")
        entry("width", "Width (in)", "6.5")
        entry("height", "Height (in)", "5")
        self.legend = tk.BooleanVar(value=True)
        self.grid_on = tk.BooleanVar(value=False)
        self.xerr = tk.BooleanVar(value=show_xerr)
        self.yerr = tk.BooleanVar(value=show_yerr)
        for text, var in (("Show legend", self.legend), ("Show grid", self.grid_on),
                          ("Show x error bars (radius uncertainty)", self.xerr),
                          ("Show y error bars (radius uncertainty)", self.yerr)):
            ttk.Checkbutton(form, text=text, variable=var, command=self.schedule).grid(
                row=row[0], column=0, columnspan=2, sticky="w")
            row[0] += 1
        ttk.Button(form, text=f"Save figure ({SAVE_DPI} dpi)...", command=self.save).grid(
            row=row[0], column=0, columnspan=2, sticky="ew", pady=(10, 2))
        self.status = tk.StringVar(value="Edits update the preview as you type.")
        ttk.Label(form, textvariable=self.status, wraplength=300, foreground="#444").grid(
            row=row[0] + 1, column=0, columnspan=2, sticky="w")

        self.fig = Figure(figsize=(6.5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        holder = ttk.Frame(right)
        holder.pack(expand=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=holder)
        self.canvas.get_tk_widget().pack()
        self.update_plot()

    def set_data(self, data):
        """data: {series name: (x, y), (x, y, bounds) or None}. Style edits are kept."""
        for s in self.series:
            xy = data.get(s["name"]) or (None, None)
            s["x"], s["y"] = xy[:2]
            s["bounds"] = xy[2] if len(xy) > 2 else None
        self.schedule()

    def schedule(self):
        if self.pending is not None:
            self.after_cancel(self.pending)
        self.pending = self.after(40, self.update_plot)

    def number(self, key, label, optional=False, positive=False):
        value = parse_number(self.v[key].get(), label, optional)
        if value is not None and positive and value <= 0:
            raise ValueError(f"{label} must be positive and finite.")
        return value

    def options(self):
        get = lambda k: self.v[k].get()
        styles = {}
        for s in self.series:
            n = s["name"]
            marker = get(n + ".marker").strip() if n + ".marker" in self.v else "square"
            style = get(n + ".style").strip() if n + ".style" in self.v else "solid"
            styles[n] = dict(
                label=get(n + ".label"), color=get(n + ".color").strip(),
                marker=MARKERS.get(marker, marker if marker in MARKERS.values() else "s"),
                size=self.number(n + ".size", n + " marker size", positive=True) if n + ".size" in self.v else 5.0,
                style=LINE_STYLES.get(style, style if style in LINE_STYLES.values() else "-"),
                width=self.number(n + ".width", n + " line width", positive=True) if n + ".width" in self.v else 1.2)
        return dict(
            title=get("title"), xlabel=get("xlabel"), ylabel=get("ylabel"),
            font_size=self.number("font_size", "Font size", positive=True),
            xscale=get("xscale") if get("xscale") in ("linear", "log") else "linear",
            yscale=get("yscale") if get("yscale") in ("linear", "log") else "linear",
            xmin=self.number("xmin", "X min", True), xmax=self.number("xmax", "X max", True),
            ymin=self.number("ymin", "Y min", True), ymax=self.number("ymax", "Y max", True),
            legend=self.legend.get(), grid=self.grid_on.get(), xerr=self.xerr.get(), yerr=self.yerr.get(),
            series=styles,
            width=self.number("width", "Width", positive=True), height=self.number("height", "Height", positive=True))

    def render(self, o):
        width, height = o["width"], o["height"]
        dpi = min(100.0, 760.0 / width, 640.0 / height)
        self.fig.set_size_inches(width, height)
        self.fig.set_dpi(dpi)
        self.canvas.get_tk_widget().config(width=int(width * dpi), height=int(height * dpi))
        draw_publication(self.ax, self.series, o)
        self.fig.tight_layout()
        self.canvas.draw()  # synchronous, so unfinished maths such as "$x_{$" is caught here

    def update_plot(self):
        self.pending = None
        try:
            o = self.options()
            self.render(o)
            self.last_good = o
            self.status.set("Preview up to date.")
        except Exception as error:
            self.status.set(f"Not updated: {first_line(error)}")
            if self.last_good is not None:
                try:
                    self.render(self.last_good)
                except Exception:
                    pass

    def save(self):
        try:
            o = self.options()
            self.render(o)
        except Exception as error:
            messagebox.showerror("Cannot save", first_line(error), parent=self)
            return
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension=".png", initialfile=self.filename,
            filetypes=[("PNG", "*.png"), ("TIFF", "*.tiff *.tif"), ("PDF (vector)", "*.pdf"),
                       ("SVG (vector)", "*.svg"), ("EPS (vector)", "*.eps")])
        if not path:
            return
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            self.fig.set_size_inches(o["width"], o["height"])
            self.fig.savefig(path, dpi=SAVE_DPI, facecolor="white")
            self.status.set(f"Saved at {SAVE_DPI} dpi: {path}")
        except Exception as error:
            messagebox.showerror("Save failed", str(error), parent=self)
        finally:
            self.config(cursor="")
            self.render(o)


class ActivityStage(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.theory = self.experiment = self.tables = None
        self.points = self.curve = self.bounds = None
        self.studio = None
        self.source = tk.StringVar(value="Experimental data: none yet (press Next on stage 2, or load a CSV).")

        nav_bar(self, back=lambda: self.app.select(1))
        outer, root = scrollable(self)
        outer.pack(fill="both", expand=True)

        bar = ttk.Frame(root)
        bar.pack(fill="x", padx=4, pady=4)
        for text, command in (("Load theoretical CSV", self.load_theory),
                              ("Load AIOMFAC file", self.load_aiomfac),
                              ("Load experimental CSV (override)", self.load_experiment),
                              ("Export activity coefficients CSV", self.export),
                              ("Publication plot...", self.open_studio)):
            ttk.Button(bar, text=text, command=command).pack(side="left", padx=2)
        ttk.Label(root, textvariable=self.source, foreground="#555", wraplength=950).pack(anchor="w", padx=6)

        opts = ttk.Frame(root)
        opts.pack(fill="x", padx=4)
        ttk.Label(opts, text="Label").pack(side="left", padx=(2, 3))
        self.name = tk.StringVar(value="A")
        entry = ttk.Entry(opts, textvariable=self.name, width=8)
        entry.pack(side="left")
        entry.bind("<KeyRelease>", lambda _: self.refresh())
        ttk.Label(opts, text="AIOMFAC component (x_i, a_x)").pack(side="left", padx=(14, 3))
        self.component = tk.StringVar(value="01")
        self.component_box = ttk.Combobox(opts, textvariable=self.component, values=["01"], width=5,
                                          state="readonly")
        self.component_box.pack(side="left")
        self.component_box.bind("<<ComboboxSelected>>", lambda _: self.refresh())
        self.xerr = tk.BooleanVar(value=True)
        self.yerr = tk.BooleanVar(value=True)
        for text, var in (("x error bars", self.xerr), ("y error bars", self.yerr)):
            ttk.Checkbutton(opts, text=text, variable=var, command=self.refresh).pack(side="left", padx=(14, 0))

        lim = ttk.Frame(root)
        lim.pack(fill="x", padx=4, pady=3)
        self.limit_vars = []
        for text in ("X min", "X max", "Y min", "Y max"):
            ttk.Label(lim, text=text).pack(side="left", padx=(2, 3))
            var = tk.StringVar()
            box = ttk.Entry(lim, textvariable=var, width=9)
            box.pack(side="left", padx=(0, 8))
            box.bind("<Return>", lambda _: self.refresh())
            box.bind("<FocusOut>", lambda _: self.refresh())
            self.limit_vars.append(var)
        ttk.Button(lim, text="Apply limits", command=self.refresh).pack(side="left", padx=2)
        ttk.Button(lim, text="Auto limits", command=self.auto_limits).pack(side="left", padx=2)
        ttk.Label(lim, text="(blank = automatic)").pack(side="left", padx=6)

        self.info = tk.StringVar(value="")
        ttk.Label(root, textvariable=self.info, wraplength=950).pack(anchor="w", padx=6, pady=2)
        self.fig = Figure(figsize=(6, 5.0))
        self.ax = self.fig.add_subplot(111)
        plot = ttk.Frame(root)
        plot.pack(fill="both", expand=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot)
        toolbar = NavigationToolbar2Tk(self.canvas, plot, pack_toolbar=False)
        toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.refresh()

    def receive(self, experiment):
        self.experiment = experiment
        self.source.set(f"Experimental data: {len(experiment[0])} interval(s) from stage 2"
                        f"{self.uncertainty_note()}. Load your theoretical (DropletLab) CSV and AIOMFAC file.")
        self.refresh()

    def pick(self, title, types):
        return filedialog.askopenfilename(title=title, filetypes=types, parent=self)

    def load_theory(self):
        path = self.pick("Theoretical (DropletLab) CSV", [("CSV", "*.csv"), ("All files", "*")])
        if path:
            self.safe(lambda: setattr(self, "theory", read_mfs_gradient(path)[:2]))

    def load_experiment(self):
        path = self.pick("Experimental CSV", [("CSV", "*.csv"), ("All files", "*")])
        if path:
            def work():
                self.experiment = read_mfs_gradient(path)
                self.source.set(f"Experimental data: {len(self.experiment[0])} row(s) from {Path(path).name} "
                                f"(overrides stage 2){self.uncertainty_note()}.")
            self.safe(work)

    def uncertainty_note(self):
        if self.experiment[2] is None:
            return ", no radius-uncertainty columns (no error bars)"
        return ", with radius-uncertainty columns (error bars)"

    def load_aiomfac(self):
        path = self.pick("AIOMFAC output file", [("Text", "*.txt"), ("All files", "*")])
        if path:
            self.safe(lambda: self.set_tables(parse_aiomfac(path)))

    def set_tables(self, tables):
        self.tables = tables
        self.component_box["values"] = list(tables)
        if self.component.get() not in tables:
            self.component.set(next(iter(tables)))

    def safe(self, action):
        try:
            action()
        except Exception as error:
            messagebox.showerror("Load error", str(error), parent=self)
        self.refresh()

    def studio_data(self):
        points = None if self.points is None else (*self.points, self.bounds)
        return {"AIOMFAC": self.curve, "Experiment": points}

    def open_studio(self):
        if self.studio is not None and self.studio.winfo_exists():
            self.studio.lift()
            return
        label = self.name.get().strip() or "A"
        series = [dict(name="AIOMFAC", kind="line", label="AIOMFAC", color="black", style="solid", width=1.2),
                  dict(name="Experiment", kind="points", label="Experiment", color="black", marker="square", size=5)]
        self.studio = PlotStudio(self.winfo_toplevel(), series, rf"$x_{{\mathrm{{{label}}}}}$",
                                 rf"$\gamma_{{\mathrm{{{label}}}}}$", "activity_coefficient.png",
                                 self.xerr.get(), self.yerr.get())
        self.studio.set_data(self.studio_data())

    def auto_limits(self):
        for var in self.limit_vars:
            var.set("")
        self.refresh()

    def refresh(self):
        notes = []
        self.points = self.curve = self.bounds = None
        if self.experiment is not None and self.theory is None:
            notes.append(f"{len(self.experiment[0])} experimental interval(s) ready; load the theoretical CSV.")
        if self.theory is not None and self.experiment is not None:
            x_exp, g_exp, extra = self.experiment
            x, gamma, outside, zero, self.bounds, truncated = activity_coefficients(x_exp, g_exp, *self.theory,
                                                                                    extra)
            self.points = (x, gamma)
            notes.append(f"{len(x)} experimental activity coefficients calculated"
                         + (" with radius-uncertainty error bars." if self.bounds is not None else "."))
            if truncated:
                notes.append(f"{truncated} error bar(s) truncated (shifted MFS outside the theoretical range).")
            if outside:
                notes.append(f"{outside} point(s) outside the theoretical MFS range skipped.")
            if zero:
                notes.append(f"{zero} skipped (theoretical gradient ~ 0).")
            if np.any(gamma <= 0):
                notes.append(f"{int(np.sum(gamma <= 0))} non-positive value(s) not shown on log axis.")
        if self.tables is not None:
            try:
                x, y, flagged = aiomfac_curve(self.tables, self.component.get())
                self.curve = (x, y)
                if flagged:
                    notes.append(f"{flagged} AIOMFAC point(s) with non-zero flag excluded.")
            except Exception as error:
                notes.append(str(error))
        if not notes:
            notes.append("Load the theoretical CSV and the AIOMFAC output file.")
        try:
            limits = parse_limits([var.get() for var in self.limit_vars], positive_y=True)
        except ValueError as error:
            limits = None
            notes.append(f"Limits ignored: {error}")
        self.info.set(" ".join(notes))
        draw_activity(self.ax, self.name.get().strip() or "A", self.points, self.curve, limits, self.bounds,
                      self.xerr.get(), self.yerr.get())
        self.fig.tight_layout()
        self.canvas.draw_idle()
        if self.studio is not None and self.studio.winfo_exists():
            self.studio.set_data(self.studio_data())

    def export(self):
        if self.points is None or not len(self.points[0]):
            messagebox.showinfo("Nothing to export", "Need experimental data and the theoretical CSV first.",
                                parent=self)
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                            initialfile="activity_coefficients.csv", parent=self)
        if not path:
            return
        try:
            columns = {"MFS": self.points[0], "Activity coefficient": self.points[1]}
            if self.bounds is not None:
                columns.update(zip(("MFS min", "MFS max", "Activity coefficient min", "Activity coefficient max"),
                                   self.bounds))
            pd.DataFrame(columns).to_csv(path, index=False, float_format="%.10g")
        except OSError as error:
            messagebox.showerror("Export failed", str(error), parent=self)
            return
        self.info.set(f"Exported {len(self.points[0])} activity coefficients to {path}")


# ----------------------------------------------------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------------------------------------------------

class PipelineApp(tk.Tk):
    STAGES = ("1 · Clean & fit", "2 · MFS & gradients", "3 · Activity coefficients")

    def __init__(self):
        super().__init__()
        self.title("EDB Pipeline: evaporation data → activity coefficients")
        self.geometry("1250x940")
        self.compounds = {}
        try:
            self.compounds = load_compounds()
        except Exception as error:
            messagebox.showerror("compounds.json", f"Place a valid {COMPOUNDS_PATH.name} beside this script.\n"
                                                   f"{error}")
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)
        self.stage1 = CleanFitStage(self.notebook, self)
        self.stage2 = MfsStage(self.notebook, self)
        self.stage3 = ActivityStage(self.notebook, self)
        for frame, text in zip((self.stage1, self.stage2, self.stage3), self.STAGES):
            self.notebook.add(frame, text=text)

    def select(self, index):
        self.notebook.select(index)

    def to_stage2(self):
        data = self.stage1.handoff()
        if data is None:
            return
        try:
            self.stage2.receive(data)
        except ValueError as error:
            messagebox.showerror("Hand-over failed", str(error), parent=self)
            return
        self.select(1)

    def to_stage3(self):
        experiment = self.stage2.handoff()
        if experiment is None:
            return
        self.stage3.receive(experiment)
        self.select(2)


if __name__ == "__main__":
    PipelineApp().mainloop()
