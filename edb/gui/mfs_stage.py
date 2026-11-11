"""Stage 2 UI: MFS & gradients (EDB Fit Processing)."""
import csv
import math
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from ..activity import sort_mfs_gradient
from ..config import COMPOUNDS_PATH, CSV_COLUMNS, LIMIT_KEYS, RADIUS_UNCERTAINTY_UM
from ..fit_io import clean_fit_arrays, read_edb_csv, read_fit_csv
from ..mfs import auto_bin_count, bin_edges, interval_results, make_mfs_function
from ..parsing import parse_bin_width, parse_limits, parse_number, parse_stride
from ..plotting import apply_limits
from ..trace import Trace
from .widgets import PointView, Progress, nav_bar, scrollable


class MfsStage(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.t = self.r2 = None
        self.edb, self.edb_scale = None, 1.0  # cleaned points (a Trace) and the factor that corrects their radius
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
        self.file_buttons = []  # disabled while a file is being read
        for label, command in (("Load fit CSV", self.load_fit), ("Load cleaned EDB data", self.load_raw),
                               ("Calculate intervals", self.calculate), ("Export CSV", self.export),
                               ("Clear results", self.clear)):
            button = ttk.Button(bar, text=label, command=command)
            button.pack(side="left", padx=2)
            if command == self.load_raw:
                self.file_buttons.append(button)
        self.progress = Progress(bar)
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
        self.field(panel, 3, 0, "Bin width (s; auto ≈ 1000, 0 = one)", "auto").bind(
            "<Return>", lambda _: self.draw(rescale=False))
        self.field(panel, 3, 1, "Plot every Nth EDB point", "auto")
        self.field(panel, 3, 2, "Radius uncertainty (um)", f"{RADIUS_UNCERTAINTY_UM:g}").bind(
            "<Return>", lambda _: self.update_model())
        r0 = ttk.Frame(panel)
        r0.grid(row=4, column=0, columnspan=6, sticky="w")
        ttk.Button(r0, text="r₀ = first EDB point", command=self.r0_from_edb).pack(side="left", padx=2)
        ttk.Button(r0, text="r₀ = fit at start", command=self.r0_from_fit).pack(side="left", padx=2)
        ttk.Button(r0, text="r₀ from dry radius", command=self.r0_from_dry).pack(side="left", padx=2)
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
        self.view = PointView(self, self.ax, self.on_view)

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
        self.edb, self.edb_scale = data["edb"], data["factor"]
        self.after_fit_loaded()
        self.v["Initial radius (um)"].set(f"{self.edb.first()[1] * self.edb_scale:.8g}")
        self.v["Radius uncertainty (um)"].set(f"{data['sigma']:.6g}")
        refit = " (refitted on the corrected radius, same window)" if data["refitted"] else ""
        self.source.set(f"From stage 1: {self.edb.count:,} cleaned points and a {len(self.t):,}-point fit{refit}, "
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
        """Read cleaned EDB data (CSV or .npz saved by stage 1) on a background thread."""
        path = filedialog.askopenfilename(title="Cleaned EDB data", parent=self, filetypes=[
            ("Cleaned data", "*.csv *.npz"), ("All files", "*")])
        if not path:
            return
        started = time.perf_counter()

        def work(progress, cancel):
            t, r, sigma = read_edb_csv(path, progress, cancel)
            return Trace(t, r), sigma

        def done(result):
            self.edb, sigma = result
            self.edb_scale = 1.0
            t0, r0 = self.edb.first()
            self.v["Initial radius (um)"].set(f"{r0:.8g}")
            self.status.set(f"EDB file: {len(self.edb):,} points read in {time.perf_counter() - started:.1f} s. "
                            f"First point t = {t0:.6g} s, corrected radius = {r0:.6g} um (set as initial radius; "
                            "editable). Read from the corrected radius saved by stage 1." + self.take_sigma(sigma))
            self.clear()
            self.update_model(rescale=True)

        self.progress.run("Loading", work, done,
                          lambda error: messagebox.showerror("EDB data error", str(error) or repr(error), parent=self),
                          lambda: self.status.set("Load cancelled."), lock=self.file_buttons)

    def take_sigma(self, sigma):
        """Use the radius uncertainty stored in a loaded CSV; returns a note for the status line."""
        if sigma is None:
            return f" No uncertainty stored; keeping ±{self.get('Radius uncertainty (um)')} um."
        self.v["Radius uncertainty (um)"].set(f"{sigma:.6g}")
        return f" Radius uncertainty ±{sigma:.4g} um taken from the file."

    def r0_from_edb(self):
        if self.edb is None:
            messagebox.showinfo("No EDB data", "No cleaned EDB data loaded.", parent=self)
            return
        self.v["Initial radius (um)"].set(f"{self.edb.first()[1] * self.edb_scale:.8g}")
        self.update_model()

    def r0_from_fit(self):
        if self.t is None:
            messagebox.showinfo("No fit", "No fit loaded.", parent=self)
            return
        self.v["Initial radius (um)"].set(f"{math.sqrt(self.r2[0]):.8g}")
        self.update_model()

    def r0_from_dry(self):
        """Effective r0 that makes the fitted radius at the interval start the dry (A+B) radius."""
        if self.t is None:
            messagebox.showinfo("No fit", "No fit loaded.", parent=self)
            return
        try:
            start = self.num("Interval start (s)", "Interval start")
            a = self.app.compounds[self.get("A: evaporating")]
            b = self.app.compounds[self.get("B: conserved")]
            s = self.app.compounds[self.get("Solvent")]
            v_a = self.num("massA: evaporating", "Mass A") / a["density_g_ml"]
            v_b = self.num("massB: conserved", "Mass B") / b["density_g_ml"]
            v_s = self.num("massSolvent", "Solvent mass") / s["density_g_ml"]
        except (ValueError, KeyError) as error:
            messagebox.showerror("Dry-radius anchor", str(error), parent=self)
            return
        r_dry = math.sqrt(np.interp(start, self.t, self.r2))
        r0 = r_dry * ((v_a + v_b + v_s) / (v_a + v_b)) ** (1 / 3)
        self.v["Initial radius (um)"].set(f"{r0:.8g}")
        self.update_model()
        self.status.set(f"r0 set to {r0:.6g} um so the fit radius at {start:g} s ({r_dry:.6g} um) "
                        "is the dry A+B radius.")

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

    def draw_points(self, whole=False, force=True):
        """Put the EDB points to draw on the plot: those in the current x range, or of the whole trace. Returns
        False (and leaves the plot alone) if force is off and they are the ones already drawn."""
        if self.edb is None:
            self.line_edb.set_data([], [])
            return True
        try:
            stride = parse_stride(self.get("Plot every Nth EDB point"))
        except ValueError:
            stride = None
            self.status.set("Plot every Nth point must be 'auto' or a positive whole number; using auto.")
        t, r, changed = self.view.get(self.edb, stride, whole)
        if changed or force:
            self.line_edb.set_data(t, (r * self.edb_scale) ** 2)
        return changed or force

    def on_view(self):
        """The x range has settled after a zoom or pan: draw the points of the new range."""
        if self.edb is not None and self.draw_points(force=False):
            self.canvas.draw_idle()

    def draw(self, rescale=False):
        ax = self.ax
        self.draw_points(whole=rescale)
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
                width = parse_bin_width(self.get("Bin width (s; auto ≈ 1000, 0 = one)"))
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
        if self.t is not None or self.edb is not None:
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
            width = parse_bin_width(self.get("Bin width (s; auto ≈ 1000, 0 = one)"))
            lo, hi = max(lo, self.t[0]), min(hi, self.t[-1])
            rows, skipped, resampled = interval_results(self.t, self.scenarios(), lo, hi, width)
        except ValueError as error:
            messagebox.showerror("Calculation error", str(error), parent=self)
            return False
        self.clear()
        self.rows = rows
        for row in rows:
            self.table.insert("", "end", values=tuple("" if math.isnan(v) else f"{v:.6g}" for v in row))
        message = f"{len(rows)} interval(s) calculated"
        if width is None:
            message += f" (automatic: {auto_bin_count(lo, hi)} equal bin(s) of {(hi - lo) / auto_bin_count(lo, hi):.6g} s)"
        message += f" with ±{self.sigma:.4g} um radius uncertainty." if self.sigma else " (no radius uncertainty)."
        if skipped:
            message += (f" {skipped} skipped: A would be negative there (fitted radius below the B-only radius "
                        "for this composition and initial radius).")
        if resampled:
            message += (f" {resampled} bin(s) are narrower than the spacing of the fit points, so the fit was "
                        "interpolated inside them.")
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
                                            initialfile="Experimental Data.csv", parent=self)
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
        if self.progress.busy:
            messagebox.showinfo("Busy", "Wait for the file to finish loading first.", parent=self)
            return None
        if not self.rows:
            if not messagebox.askyesno("No intervals", "No intervals calculated yet. Calculate them now with the "
                                       "current settings?", parent=self):
                return None
            if not self.calculate() or not self.rows:
                return None
        rows = np.asarray(self.rows, dtype=float)
        return sort_mfs_gradient(rows[:, 0], rows[:, 1], rows[:, 2:])
