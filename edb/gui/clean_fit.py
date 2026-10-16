"""Stage 1 UI: Clean & fit (EDB Plotter)."""
import csv
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector, SpanSelector

from ..config import EDB_ASSUMED_RI, FIT_EXPORT_POINTS, MAX_ERROR_BARS, RADIUS_UNCERTAINTY_UM
from ..culling import PointStore
from ..fitting import fit_curve, fit_model
from ..optics import correction_factor
from ..parsing import parse_limits, parse_number
from ..plotting import apply_limits
from ..raw_data import read_files
from .widgets import nav_bar, scrollable


MODE_HELP = {
    "fit": "Fit region: drag horizontally to fit that time region.",
    "zoom": "Zoom: drag a rectangle to zoom immediately.",
    "cull": "Cull: drag a rectangle to remove all points inside it immediately.",
    "keep": "Keep box: drag a rectangle to remove all points outside it.",
}


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
                                            initialfile="Cleaned EDB Data.csv", parent=self)
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
                                            initialfile="Fitted EDB Data.csv", parent=self)
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
