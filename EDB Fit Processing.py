import csv
import json
import math
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

JSON_NAME = "compounds.json"
MIN_POINTS = 3
MAX_BINS = 20000
CSV_COLUMNS = ["Average MFS", "Gradient (um2/s)"]
LIMIT_KEYS = (("X min (s)", "x_min"), ("X max (s)", "x_max"), ("Y min", "y_min"), ("Y max", "y_max"))


def load_compounds(path):
    with open(path, encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "compounds" in data:
        data = data["compounds"]
    if isinstance(data, list):
        data = {record["name"]: record for record in data}
    for name, record in data.items():
        for key in ("density_g_ml", "molar_mass_g_mol"):
            value = record[key]
            value = float(value["value"] if isinstance(value, dict) else value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name}: {key} must be positive.")
            record[key] = value
    return data


def read_fit_csv(path):
    """Return time (s) and fitted radius squared (um^2) from the polynomial-fit CSV."""
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    cols = {c.strip().lower(): c for c in df.columns}
    time_col = cols.get("time_s", df.columns[0])
    if "fitted_radius_squared_um2" in cols:
        r2 = pd.to_numeric(df[cols["fitted_radius_squared_um2"]], errors="coerce")
    elif "fitted_radius_um" in cols:
        r2 = pd.to_numeric(df[cols["fitted_radius_um"]], errors="coerce") ** 2
    elif df.shape[1] >= 3:
        r2 = pd.to_numeric(df.iloc[:, 2], errors="coerce")
    else:
        raise ValueError("Could not find the fitted radius squared column.")
    t = pd.to_numeric(df[time_col], errors="coerce")
    good = t.notna() & r2.notna() & (r2 > 0)
    t, r2 = t[good].to_numpy(float), r2[good].to_numpy(float)
    if len(t) < MIN_POINTS:
        raise ValueError("The fit CSV needs at least three valid rows.")
    order = np.argsort(t, kind="stable")
    t, r2 = t[order], r2[order]
    if np.any(np.diff(t) <= 0):
        raise ValueError("Fit CSV times must be strictly increasing.")
    return t, r2


def read_edb_csv(path):
    with open(path, encoding="utf-8-sig", errors="replace") as handle:
        sample = "".join(handle.readline() for _ in range(5))
    sep = "," if "," in sample else r"\s+"
    df = pd.read_csv(path, sep=sep, header=None, usecols=[0, 1], dtype=str, encoding="utf-8-sig",
                     encoding_errors="replace", on_bad_lines="skip")
    t = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(float)
    r = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(float)
    good = np.isfinite(t) & np.isfinite(r) & (r > 0)
    if not good.any():
        raise ValueError("No valid time/radius rows found in the EDB file.")
    return t[good], r[good]


def make_mfs_function(records, name_a, name_b, name_s, mass_a, mass_b, mass_s, r0):
    if len({name_a, name_b, name_s}) < 3:
        raise ValueError("Choose three different compounds for A, B and the solvent.")
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


def interval_results(t, r2, mfs, lo, hi, width):
    x_all = mfs(r2)
    edges = bin_edges(lo, hi, width)
    rows, skipped = [], 0
    for j, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        last = j == len(edges) - 2
        mask = (t >= a) & ((t <= b) if last else (t < b))
        tt, yy, xx = t[mask], r2[mask], x_all[mask]
        if len(tt) < MIN_POINTS or np.any(~np.isfinite(xx)):
            skipped += 1
            continue
        span = tt[-1] - tt[0]
        average = float(np.sum(0.5 * (xx[1:] + xx[:-1]) * np.diff(tt)) / span)
        slope = float(np.polyfit(tt - tt.mean(), yy, 1)[0])
        rows.append((average, slope))
    return rows, skipped


def parse_limits(texts):
    values = []
    for (label, _), text in zip(LIMIT_KEYS, texts):
        text = text.strip()
        if not text:
            values.append(None)
            continue
        try:
            value = float(text)
        except ValueError:
            raise ValueError(f"{label} must be a number or blank.")
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite.")
        values.append(value)
    for i, name in ((0, "X"), (2, "Y")):
        if values[i] is not None and values[i + 1] is not None and values[i] >= values[i + 1]:
            raise ValueError(f"{name} min must be below {name} max.")
    return values


def apply_limits(ax, lo, hi, which):
    current = ax.get_xlim() if which == "x" else ax.get_ylim()
    new = (current[0] if lo is None else lo, current[1] if hi is None else hi)
    (ax.set_xlim if which == "x" else ax.set_ylim)(*new)


class App:
    def __init__(self, root):
        self.root = root
        root.title("EDB timed intervals - MFS and gradient from polynomial fit")
        root.geometry("1250x940")
        self.t = self.r2 = None
        self.edb_t = self.edb_r2 = None
        self.mfs = None
        self.rows = []
        self.records = {}
        self.edge_lines = []
        self.v = {}

        bar = ttk.Frame(root)
        bar.pack(fill="x", padx=4, pady=4)
        for label, command in (("Load fit CSV", self.load_fit), ("Load Cleaned EDB CSV", self.load_raw),
                               ("Calculate intervals", self.calculate), ("Export CSV", self.export),
                               ("Clear results", self.clear)):
            ttk.Button(bar, text=label, command=command).pack(side="left", padx=2)

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

        self.status = tk.StringVar(value="Load the polynomial-fit CSV and the altered EDB CSV.")
        self.info = tk.StringVar(value="")
        ttk.Label(root, textvariable=self.status, wraplength=1200).pack(anchor="w", padx=6, pady=(4, 0))
        ttk.Label(root, textvariable=self.info, wraplength=1200).pack(anchor="w", padx=6)

        modes = ttk.Frame(root)
        modes.pack(fill="x", padx=6)
        self.mode = tk.StringVar(value="select")
        for text, value in (("Select interval (drag)", "select"), ("Zoom (drag)", "zoom")):
            ttk.Radiobutton(modes, text=text, value=value, variable=self.mode, command=self.set_mode).pack(side="left", padx=6)
        ttk.Button(modes, text="Reset view", command=self.reset_view).pack(side="left", padx=6)

        self.fig = Figure(figsize=(9, 4.8))
        self.ax = self.fig.add_subplot(111)
        self.line_r2, = self.ax.plot([], [], color="tab:red", linewidth=2.2, zorder=2, label="Polynomial fit")
        self.line_edb, = self.ax.plot([], [], linestyle="none", marker="x", markersize=3, markeredgewidth=0.6,
                                      color="tab:blue", alpha=0.7, zorder=3, label="EDB data (altered)")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Radius\u00b2 (\u00b5m\u00b2)")
        self.ax.grid(alpha=0.25)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, root)
        self.span = SpanSelector(self.ax, self.on_span, "horizontal", useblit=True, interactive=False,
                                 props={"alpha": 0.2, "facecolor": "orange"})
        self.stride_entry.bind("<Return>", lambda _: self.draw(rescale=False))

        frame = ttk.Frame(root)
        frame.pack(fill="x", padx=4, pady=4)
        self.table = ttk.Treeview(frame, columns=CSV_COLUMNS, show="headings", height=6)
        for column in CSV_COLUMNS:
            self.table.heading(column, text=column)
            self.table.column(column, width=220)
        self.table.pack(side="left", fill="x", expand=True)
        self.load_json()
        self.draw(rescale=True)

    def field(self, panel, row, col, key, default):
        ttk.Label(panel, text=key).grid(row=row, column=2 * col, sticky="w", padx=3)
        var = tk.StringVar(value=default)
        self.v[key] = var
        entry = ttk.Entry(panel, textvariable=var, width=22)
        entry.grid(row=row, column=2 * col + 1, padx=3, pady=2)
        if key.startswith("Plot every"):
            self.stride_entry = entry

    def get(self, key):
        return self.v[key].get().strip()

    def num(self, key, label):
        text = self.get(key)
        if not text:
            raise ValueError(f"{label} is required.")
        try:
            value = float(text)
        except ValueError:
            raise ValueError(f"{label} must be a number.")
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite.")
        return value

    def load_json(self):
        path = Path(__file__).resolve().parent / JSON_NAME
        try:
            self.records = load_compounds(path)
        except Exception as error:
            messagebox.showerror("JSON error", f"Place {JSON_NAME} beside this script.\n{error}")
            return
        names = list(self.records)
        defaults = {"A: evaporating": "Malonic acid", "B: conserved": "Tween 20", "Solvent": "Water"}
        for box, key in zip(self.boxes, ("A: evaporating", "B: conserved", "Solvent")):
            box["values"] = names
            self.v[key].set(defaults[key] if defaults[key] in names else names[0])

    def load_fit(self):
        path = filedialog.askopenfilename(title="Polynomial-fit CSV", filetypes=[("CSV", "*.csv"), ("All files", "*")])
        if not path:
            return
        try:
            self.t, self.r2 = read_fit_csv(path)
        except Exception as error:
            messagebox.showerror("Fit CSV error", str(error))
            return
        self.v["Interval start (s)"].set(f"{self.t[0]:.8g}")
        self.v["Interval end (s)"].set(f"{self.t[-1]:.8g}")
        self.status.set(f"Fit CSV: {len(self.t):,} points, {self.t[0]:.6g}-{self.t[-1]:.6g} s.")
        self.clear()
        self.update_model(rescale=True)

    def load_raw(self):
        path = filedialog.askopenfilename(title="Altered EDB CSV", filetypes=[("CSV", "*.csv"), ("All files", "*")])
        if not path:
            return
        self.status.set("Reading EDB file...")
        self.root.update_idletasks()
        try:
            t, r = read_edb_csv(path)
        except Exception as error:
            messagebox.showerror("EDB CSV error", str(error))
            return
        self.edb_t, self.edb_r2 = t, r * r
        self.v["Initial radius (um)"].set(f"{r[0]:.8g}")
        self.status.set(f"EDB file: {len(t):,} points. First point t = {t[0]:.6g} s, radius = {r[0]:.6g} um "
                        "(set as initial radius; editable).")
        self.clear()
        self.update_model(rescale=True)

    def update_model(self, rescale=False):
        self.mfs = None
        try:
            self.mfs, info = make_mfs_function(
                self.records, self.get("A: evaporating"), self.get("B: conserved"), self.get("Solvent"),
                self.num("massA: evaporating", "Mass A"), self.num("massB: conserved", "Mass B"),
                self.num("massSolvent", "Solvent mass"), self.num("Initial radius (um)", "Initial radius"))
            self.info.set(f"Initial dry x(A) = {info['x_a0']:.6g}; calculated dry (A+B) radius = "
                          f"{info['r_dry']:.6g} um. Below this radius solvent is gone and A is lost.")
        except ValueError as error:
            self.info.set(f"Composition: {error}")
        self.draw(rescale=rescale)

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
        if self.t is not None:
            self.line_r2.set_data(self.t, self.r2)
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
            b = parse_limits([var.get() for var in self.limit_vars])
            apply_limits(ax, b[0], b[1], "x")
            apply_limits(ax, b[2], b[3], "y")
        except ValueError as error:
            self.status.set(f"Axis limits ignored: {error}")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def calculate(self):
        if self.t is None:
            messagebox.showinfo("No fit", "Load the polynomial-fit CSV first.")
            return
        self.update_model()
        if self.mfs is None:
            messagebox.showerror("Composition error", self.info.get())
            return
        try:
            lo = self.num("Interval start (s)", "Interval start")
            hi = self.num("Interval end (s)", "Interval end")
            width = self.num("Bin width (s, 0 = one interval)", "Bin width")
            lo, hi = max(lo, self.t[0]), min(hi, self.t[-1])
            rows, skipped = interval_results(self.t, self.r2, self.mfs, lo, hi, width)
        except ValueError as error:
            messagebox.showerror("Calculation error", str(error))
            return
        self.clear()
        self.rows = rows
        for average, gradient in rows:
            self.table.insert("", "end", values=(f"{average:.6g}", f"{gradient:.6g}"))
        message = f"{len(rows)} interval(s) calculated."
        if skipped:
            message += (f" {skipped} skipped (fewer than {MIN_POINTS} fit points, or A would be negative "
                        "i.e. radius below the B-only radius).")
        self.status.set(message)

    def clear(self):
        self.rows = []
        for item in self.table.get_children():
            self.table.delete(item)

    def export(self):
        if not self.rows:
            messagebox.showinfo("Nothing to export", "Calculate intervals first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                            initialfile="edb_intervals.csv")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(CSV_COLUMNS)
                writer.writerows(self.rows)
        except OSError as error:
            messagebox.showerror("Export error", str(error))
            return
        self.status.set(f"Exported {len(self.rows)} row(s).")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
