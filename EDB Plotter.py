import csv
import math
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector, SpanSelector

RADIUS_UNCERTAINTY_UM = 0.1
MAX_FIT_BINS = 3000
DICT_TAUS = np.geomspace(0.003, 30.0, 70)


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

    def cull_box(self, box, squared, keep_inside=False):
        xmin, xmax, ymin, ymax = box
        ids = self.ids()
        t, r = self.data[ids, 0], self.data[ids, 1]
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

    terms = " + ".join(f"{a:.6g}\u00b7exp(-t/{g * span:.6g} s)" for a, g in zip(amps, taus))
    detail = f"r(t) = {offset:.8g} + {terms}  (plateau and amplitudes constrained \u2265 0)"
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


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EDB radius plotter - monotone convex fit")
        self.geometry("1150x900")
        self.files = []
        self.store = None
        self.fit = None
        self.fit_band = None
        self.var = {name: tk.StringVar(value=value) for name, value in (
            ("stride", "20"), ("fit_lo", ""), ("fit_hi", ""),
            ("x_min", ""), ("x_max", ""), ("y_min", ""), ("y_max", ""))}
        self.squared = tk.BooleanVar(value=False)
        self.use_err = tk.BooleanVar(value=False)
        self.mode = tk.StringVar(value="fit")
        self.status = tk.StringVar(value="Add EDB files in acquisition order, then click Load / plot.")
        self.fit_text = tk.StringVar(value="Fit: none yet.")
        self.build()

    def build(self):
        viewport = ttk.Frame(self)
        viewport.pack(fill="both", expand=True)
        self.scroll_canvas = tk.Canvas(viewport, highlightthickness=0)
        scrollbar = ttk.Scrollbar(viewport, orient="vertical", command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=scrollbar.set)
        self.scroll_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        content = ttk.Frame(self.scroll_canvas)
        content_window = self.scroll_canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind(
            "<Configure>",
            lambda _: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")),
        )
        self.scroll_canvas.bind(
            "<Configure>",
            lambda event: self.scroll_canvas.itemconfigure(content_window, width=event.width),
        )

        top = ttk.Frame(content, padding=6)
        top.pack(fill="x")

        files = ttk.LabelFrame(top, text="Files (acquisition order)", padding=6)
        files.pack(side="left", fill="y")
        self.listbox = tk.Listbox(files, height=5, width=38, selectmode=tk.EXTENDED)
        self.listbox.pack(side="left")
        buttons = ttk.Frame(files)
        buttons.pack(side="left", padx=6)
        for text, command in (("Add files...", self.add_files), ("Remove selected", self.remove_files),
                              ("Move up", lambda: self.move(-1)), ("Move down", lambda: self.move(1)),
                              ("Clear", self.clear_files)):
            ttk.Button(buttons, text=text, command=command).pack(fill="x", pady=1)

        opts = ttk.LabelFrame(top, text="Plot and fit", padding=6)
        opts.pack(side="left", fill="both", expand=True, padx=(8, 0))
        ttk.Checkbutton(opts, text="Plot radius\u00b2 (otherwise radius)", variable=self.squared,
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
        axes = ttk.Frame(opts)
        axes.grid(row=5, column=0, columnspan=4, sticky="w", pady=(4, 0))
        for label, key in (("X min", "x_min"), ("X max", "x_max"), ("Y min", "y_min"), ("Y max", "y_max")):
            ttk.Label(axes, text=label).pack(side="left", padx=(0, 2))
            box = ttk.Entry(axes, textvariable=self.var[key], width=8)
            box.pack(side="left", padx=(0, 8))
            box.bind("<Return>", lambda _: self.update_plot())
        ttk.Button(axes, text="Apply axes", command=self.update_plot).pack(side="left")

        act = ttk.Frame(content, padding=(6, 0))
        act.pack(fill="x")
        for text, command in (("Load / plot", self.load), ("Fit", self.do_fit),
                              ("Clear fit region", self.clear_region), ("Export fit CSV", self.export_fit)):
            ttk.Button(act, text=text, command=command).pack(side="left", padx=2, pady=3)

        modes = ttk.LabelFrame(content, text="Plot mode (drag on the plot)", padding=4)
        modes.pack(fill="x", padx=6, pady=2)
        for text, value in (("Fit region", "fit"), ("Zoom", "zoom"), ("Cull", "cull"), ("Keep box", "keep")):
            ttk.Radiobutton(modes, text=text, value=value, variable=self.mode,
                            command=self.set_mode).pack(side="left", padx=6)
        for text, command in (("Undo cull", self.undo_cull), ("Restore all", self.restore_points),
                              ("Reset zoom", self.reset_zoom), ("Save altered data", self.save_altered)):
            ttk.Button(modes, text=text, command=command).pack(side="left", padx=3)

        ttk.Label(content, textvariable=self.status).pack(anchor="w", padx=8)
        ttk.Label(content, textvariable=self.fit_text, wraplength=1100, justify="left").pack(anchor="w", padx=8)

        self.figure = Figure(figsize=(8, 5))
        self.axis = self.figure.add_subplot(111)
        self.axis.grid(alpha=0.3)
        self.data_line, = self.axis.plot([], [], linestyle="none", marker="x", markersize=4,
                                         markeredgewidth=0.8, color="tab:blue", label="Data")
        self.err_bars = LineCollection([], colors="tab:blue", linewidths=0.6, alpha=0.6, zorder=1)
        self.axis.add_collection(self.err_bars)
        self.fit_line, = self.axis.plot([], [], color="tab:red", linewidth=1.8, label="Monotone convex fit")
        self.canvas = FigureCanvasTkAgg(self.figure, master=content)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, content)
        self.span = SpanSelector(self.axis, self.on_span, "horizontal", useblit=True, interactive=False,
                                 props={"alpha": 0.2, "facecolor": "crimson"})
        self.rect = RectangleSelector(self.axis, self.on_box, useblit=True, button=[1], minspanx=3, minspany=3,
                                      spancoords="pixels", interactive=False,
                                      props=dict(facecolor="orange", edgecolor="darkorange", alpha=0.25))
        self.rect.set_active(False)
        self.update_plot(rescale=True)

    def refresh_list(self):
        self.listbox.delete(0, tk.END)
        for i, path in enumerate(self.files, 1):
            self.listbox.insert(tk.END, f"{i:02d}. {Path(path).name}")

    def add_files(self):
        for path in filedialog.askopenfilenames(title="Select EDB files", filetypes=[
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
            messagebox.showerror("Load error", str(error))
            return
        self.store = PointStore(data)
        self.fit = None
        self.status.set(f"Loaded {len(data):,} points from {len(self.files)} file(s); skipped {skipped:,} "
                        "non-data lines. Radius assumed in \u00b5m; time starts at 0 s.")
        self.fit_text.set("Fit: none yet. Drag in Fit region mode or click Fit for the whole trace.")
        self.update_plot(rescale=True)

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
                removed = self.store.cull_box((xmin, xmax, ymin, ymax), self.squared.get(), mode == "keep")
            except ValueError as error:
                messagebox.showinfo("Cull", str(error))
                removed = 0
            if removed:
                self.after_edit()
        self.rect.clear()
        self.canvas.draw_idle()

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

    def save_altered(self):
        if self.store is None:
            messagebox.showinfo("No data", "Load data first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                            initialfile="edb_altered.csv")
        if not path:
            return
        if any(Path(path).exists() and Path(path).samefile(p) for p in self.files):
            messagebox.showerror("Input selected", "Choose an output file different from the inputs.")
            return
        t, r = self.store.active()
        try:
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if self.use_err.get():
                    sigma = np.full_like(r, RADIUS_UNCERTAINTY_UM)
                    writer.writerow(["time_s", "radius_um", "radius_err_um", "radius_squared_um2",
                                     "radius_squared_err_um2"])
                    writer.writerows(zip(t, r, sigma, r * r, 2 * r * sigma))
                else:
                    writer.writerow(["time_s", "radius_um"])
                    writer.writerows(zip(t, r))
        except OSError as error:
            messagebox.showerror("Save failed", str(error))
            return
        self.status.set(f"Saved {len(t):,} retained rows.")

    def number(self, key, label):
        text = self.var[key].get().strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            raise ValueError(f"{label} must be a number.")
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite.")
        return value

    def clear_region(self):
        self.var["fit_lo"].set("")
        self.var["fit_hi"].set("")
        self.do_fit()

    def do_fit(self, redraw=True):
        if self.store is None:
            self.fit_text.set("Load data first.")
            return
        try:
            t, r = self.store.active()
            self.fit = fit_model(t, r, self.number("fit_lo", "Fit from"), self.number("fit_hi", "Fit to"))
        except ValueError as error:
            self.fit = None
            self.fit_text.set(f"Fit failed: {error}")
        else:
            f = self.fit
            self.fit_text.set(
                f"{f['label']} to radius, {f['lo']:.6g}-{f['hi']:.6g} s, {f['n']:,} retained points. "
                f"RMS residual {f['rms']:.4g} \u00b5m, R\u00b2 {f['r2']:.6f}. {f['detail']}")
        if redraw:
            self.update_plot()

    def export_fit(self):
        if self.fit is None:
            messagebox.showinfo("No fit", "Run a fit first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                            initialfile="edb_fit.csv")
        if not path:
            return
        x, radius, radius2 = fit_curve(self.fit, 1000)
        try:
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if self.use_err.get():
                    sigma = RADIUS_UNCERTAINTY_UM
                    sigma2 = 2 * radius * sigma
                    writer.writerow(["time_s", "fitted_radius_um", "fitted_radius_err_um",
                                     "fitted_radius_lower_um", "fitted_radius_upper_um",
                                     "fitted_radius_squared_um2", "fitted_radius_squared_err_um2",
                                     "fitted_radius_squared_lower_um2", "fitted_radius_squared_upper_um2"])
                    writer.writerows(zip(x, radius, np.full_like(radius, sigma), radius - sigma, radius + sigma,
                                         radius2, sigma2, radius2 - sigma2, radius2 + sigma2))
                else:
                    writer.writerow(["time_s", "fitted_radius_um", "fitted_radius_squared_um2"])
                    writer.writerows(zip(x, radius, radius2))
        except OSError as error:
            messagebox.showerror("Export error", str(error))

    def update_plot(self, rescale=False):
        ax = self.axis
        squared = self.squared.get()
        show_err = self.use_err.get()
        ax.set_xlabel("Time since first point (s)")
        ax.set_ylabel("Radius\u00b2 (\u00b5m\u00b2)" if squared else "Radius (\u00b5m)")
        try:
            stride = int(self.var["stride"].get())
            if stride < 1:
                raise ValueError
        except ValueError:
            stride = 1
            self.status.set("Plot every Nth point must be a positive whole number; using 1.")
        if self.store is not None:
            t, r = self.store.active()
            y = r * r if squared else r
            self.data_line.set_data(t[::stride], y[::stride])
            if show_err:
                ts, ys, rs = t[::stride], y[::stride], r[::stride]
                err = RADIUS_UNCERTAINTY_UM * (2 * rs if squared else np.ones_like(rs))
                self.err_bars.set_segments(np.stack([np.column_stack([ts, ys - err]),
                                                     np.column_stack([ts, ys + err])], axis=1))
            else:
                self.err_bars.set_segments([])
            ax.set_title(f"EDB evaporation \u2014 {len(t):,} retained; "
                         f"{len(self.store.data) - len(t):,} culled; {len(t[::stride]):,} plotted")
        else:
            self.data_line.set_data([], [])
            self.err_bars.set_segments([])
        if self.fit_band is not None:
            self.fit_band.remove()
            self.fit_band = None
        if self.fit is not None:
            x, radius, radius2 = fit_curve(self.fit)
            self.fit_line.set_data(x, radius2 if squared else radius)
            if show_err:
                centre = radius2 if squared else radius
                half = RADIUS_UNCERTAINTY_UM * (2 * radius if squared else 1.0)
                self.fit_band = ax.fill_between(x, centre - half, centre + half, color="tab:red", alpha=0.25,
                                                linewidth=0, zorder=2)
        else:
            self.fit_line.set_data([], [])
        if rescale:
            ax.autoscale(enable=True)
            ax.relim()
            segments = self.err_bars.get_segments()
            if segments:
                ax.update_datalim(np.vstack(segments))
            ax.autoscale_view()
        try:
            b = [self.number(k, k.replace("_", " ")) for k in ("x_min", "x_max", "y_min", "y_max")]
            for low, high, name in ((b[0], b[1], "X"), (b[2], b[3], "Y")):
                if low is not None and high is not None and low >= high:
                    raise ValueError(f"{name} min must be below {name} max.")
            if b[0] is not None or b[1] is not None:
                left, right = ax.get_xlim()
                ax.set_xlim(left if b[0] is None else b[0], right if b[1] is None else b[1])
            if b[2] is not None or b[3] is not None:
                bottom, top = ax.get_ylim()
                ax.set_ylim(bottom if b[2] is None else b[2], top if b[3] is None else b[3])
        except ValueError as error:
            self.status.set(f"Axis limits ignored: {error}")
        self.canvas.draw_idle()


if __name__ == "__main__":
    App().mainloop()
