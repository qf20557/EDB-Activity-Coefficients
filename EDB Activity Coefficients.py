import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import is_color_like
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

AIOMFAC_COLUMN = "a_x"


def read_mfs_gradient(path):
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    if df.shape[1] < 2:
        raise ValueError("The CSV needs two columns: Average MFS and Gradient.")
    x = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(float)
    g = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(float)
    good = np.isfinite(x) & np.isfinite(g)
    if not good.any():
        raise ValueError("No numeric rows found.")
    order = np.argsort(x[good], kind="stable")
    return x[good][order], g[good][order]


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


def activity_coefficients(x_exp, g_exp, x_theory, g_theory):
    ux, idx = np.unique(x_theory, return_index=True)
    gt = g_theory[idx]
    inside = (x_exp >= ux[0]) & (x_exp <= ux[-1])
    theory_at = np.full(len(x_exp), np.nan)
    theory_at[inside] = np.interp(x_exp[inside], ux, gt)
    nonzero = np.abs(np.nan_to_num(theory_at)) > 1e-12 * max(np.abs(gt).max(), 1e-300)
    good = inside & nonzero
    return x_exp[good], g_exp[good] / theory_at[good], int((~inside).sum()), int((inside & ~nonzero).sum())


def parse_limits(texts):
    values = []
    for name, text in zip(("X min", "X max", "Y min", "Y max"), texts):
        text = text.strip()
        if not text:
            values.append(None)
            continue
        try:
            value = float(text)
        except ValueError:
            raise ValueError(f"{name} must be a number or blank.")
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
        values.append(value)
    xmin, xmax, ymin, ymax = values
    for low, high, axis in ((xmin, xmax, "X"), (ymin, ymax, "Y")):
        if low is not None and high is not None and low >= high:
            raise ValueError(f"{axis} min must be below {axis} max.")
    if (ymin is not None and ymin <= 0) or (ymax is not None and ymax <= 0):
        raise ValueError("Y limits must be positive (log axis).")
    return xmin, xmax, ymin, ymax


def draw(ax, label, points=None, curve=None, limits=None):
    ax.clear()
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
    if has_points:
        lo, hi = points[0].min(), points[0].max()
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
    values = [v[np.isfinite(v) & (v > 0)] for v in values]
    values = np.concatenate(values) if values else np.array([])
    if values.size:
        ax.set_ylim(10 ** np.floor(np.log10(values.min()) - 0.05), 10 ** np.ceil(np.log10(values.max()) + 0.05))
    xmin, xmax, ymin, ymax = limits or (None, None, None, None)
    if xmin is not None or xmax is not None:
        left, right = ax.get_xlim()
        ax.set_xlim(left if xmin is None else xmin, right if xmax is None else xmax)
    if ymin is not None or ymax is not None:
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom if ymin is None else ymin, top if ymax is None else ymax)
    if curve is not None or has_points:
        ax.legend(frameon=False, loc="upper left")


SAVE_DPI = 1200
matplotlib.rcParams["savefig.dpi"] = SAVE_DPI  # every save (including the toolbar save button) is 1200 dpi
COLOURS = ("black", "red", "blue", "green", "orange", "purple", "grey", "brown", "teal", "magenta")
MARKERS = {"square": "s", "circle": "o", "triangle": "^", "diamond": "D", "cross (x)": "x", "plus": "+"}
LINE_STYLES = {"solid": "-", "dashed": "--", "dash-dot": "-.", "dotted": ":"}


def first_line(error):
    text = next((line.strip() for line in str(error).splitlines() if line.strip()), "invalid input")
    if "end_group" in str(error) or "Expected" in str(error):
        text += "  (check that every $...$ and { } in your labels is complete)"
    return text[:200]


def _has_data(series):
    return series.get("x") is not None and len(series["x"]) > 0


def auto_axis_limits(series, xscale, yscale):
    """Default (xlim, ylim) from the data. Point series set the x range; lines only if there are no points."""
    series = [s for s in series if _has_data(s)]
    points = [s for s in series if s["kind"] == "points"]
    lines = [s for s in series if s["kind"] == "line"]
    source = points or lines
    if not source:
        return None, None
    xs = np.concatenate([np.asarray(s["x"], dtype=float) for s in source])
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


def default_options(series, xlabel, ylabel):
    return dict(
        title="", xlabel=xlabel, ylabel=ylabel, font_size=12, xscale="linear", yscale="log",
        xmin=None, xmax=None, ymin=None, ymax=None, legend=True, grid=False, width=6.5, height=5.0,
        series={s["name"]: dict(label=s["label"], color=s["color"], marker=MARKERS.get(s.get("marker", "square"), "s"),
                                size=s.get("size", 5.0), style=LINE_STYLES.get(s.get("style", "solid"), "-"),
                                width=s.get("width", 1.2)) for s in series})


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
    xlim, ylim = auto_axis_limits(series, o["xscale"], o["yscale"])
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    if o["xmin"] is not None or o["xmax"] is not None:
        left, right = ax.get_xlim()
        ax.set_xlim(left if o["xmin"] is None else o["xmin"], right if o["xmax"] is None else o["xmax"])
    if o["ymin"] is not None or o["ymax"] is not None:
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom if o["ymin"] is None else o["ymin"], top if o["ymax"] is None else o["ymax"])
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


class PlotStudio(tk.Toplevel):
    def __init__(self, master, series, xlabel, ylabel, filename="figure.png"):
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
        for text, var in (("Show legend", self.legend), ("Show grid", self.grid_on)):
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
        """data: {series name: (x, y) or None}. Style edits are kept."""
        for s in self.series:
            xy = data.get(s["name"])
            s["x"], s["y"] = (xy if xy is not None else (None, None))
        self.schedule()

    def schedule(self):
        if self.pending is not None:
            self.after_cancel(self.pending)
        self.pending = self.after(40, self.update_plot)

    def number(self, key, label, optional=False, positive=False):
        text = self.v[key].get().strip()
        if not text:
            if optional:
                return None
            raise ValueError(f"{label} is required.")
        try:
            value = float(text)
        except ValueError:
            raise ValueError(f"{label} must be a number.")
        if not np.isfinite(value) or (positive and value <= 0):
            raise ValueError(f"{label} must be {'positive and ' if positive else ''}finite.")
        return value

    def options(self):
        get = lambda k: self.v[k].get()
        styles = {}
        for s in self.series:
            n = s["name"]
            marker, style = get(n + ".marker").strip() if n + ".marker" in self.v else "square", \
                get(n + ".style").strip() if n + ".style" in self.v else "solid"
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
            legend=self.legend.get(), grid=self.grid_on.get(), series=styles,
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


class App:
    def __init__(self, root):
        self.root = root
        root.title("Activity coefficient: experiment vs AIOMFAC")
        root.geometry("1000x830")
        self.theory = self.experiment = self.tables = None
        self.points = self.curve = None
        self.studio = None

        bar = ttk.Frame(root)
        bar.pack(fill="x", padx=4, pady=4)
        for text, command in (("Load theoretical CSV", self.load_theory),
                              ("Load experimental CSV", self.load_experiment),
                              ("Load AIOMFAC file", self.load_aiomfac),
                              ("Export activity coefficients CSV", self.export),
                              ("Publication plot...", self.open_studio)):
            ttk.Button(bar, text=text, command=command).pack(side="left", padx=2)

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

        self.info = tk.StringVar(value="Load the theoretical CSV, experimental CSV and AIOMFAC output file.")
        ttk.Label(root, textvariable=self.info, wraplength=950).pack(anchor="w", padx=6, pady=2)
        self.fig = Figure(figsize=(6, 5.5))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, root)
        self.refresh()

    def pick(self, title, types):
        return filedialog.askopenfilename(title=title, filetypes=types)

    def load_theory(self):
        path = self.pick("Theoretical (DropletLab) CSV", [("CSV", "*.csv"), ("All files", "*")])
        if path:
            self.safe(lambda: setattr(self, "theory", read_mfs_gradient(path)))

    def load_experiment(self):
        path = self.pick("Experimental CSV", [("CSV", "*.csv"), ("All files", "*")])
        if path:
            self.safe(lambda: setattr(self, "experiment", read_mfs_gradient(path)))

    def load_aiomfac(self):
        path = self.pick("AIOMFAC output file", [("Text", "*.txt"), ("All files", "*")])
        if path:
            def work():
                self.tables = parse_aiomfac(path)
                self.component_box["values"] = list(self.tables)
                if self.component.get() not in self.tables:
                    self.component.set(next(iter(self.tables)))
            self.safe(work)

    def safe(self, action):
        try:
            action()
        except Exception as error:
            messagebox.showerror("Load error", str(error))
        self.refresh()

    def studio_data(self):
        return {"AIOMFAC": self.curve, "Experiment": self.points}

    def open_studio(self):
        if self.studio is not None and self.studio.winfo_exists():
            self.studio.lift()
            return
        label = self.name.get().strip() or "A"
        series = [dict(name="AIOMFAC", kind="line", label="AIOMFAC", color="black", style="solid", width=1.2),
                  dict(name="Experiment", kind="points", label="Experiment", color="black", marker="square", size=5)]
        self.studio = PlotStudio(self.root, series, rf"$x_{{\mathrm{{{label}}}}}$", rf"$\gamma_{{\mathrm{{{label}}}}}$",
                                 "activity_coefficient.png")
        self.studio.set_data(self.studio_data())

    def auto_limits(self):
        for var in self.limit_vars:
            var.set("")
        self.refresh()

    def refresh(self):
        notes = []
        self.points = self.curve = None
        if self.theory is not None and self.experiment is not None:
            x, gamma, outside, zero = activity_coefficients(*self.experiment, *self.theory)
            self.points = (x, gamma)
            notes.append(f"{len(x)} experimental activity coefficients calculated.")
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
            notes.append("Load the theoretical CSV, experimental CSV and AIOMFAC output file.")
        try:
            limits = parse_limits([var.get() for var in self.limit_vars])
        except ValueError as error:
            limits = None
            notes.append(f"Limits ignored: {error}")
        self.info.set(" ".join(notes))
        draw(self.ax, self.name.get().strip() or "A", self.points, self.curve, limits)
        self.fig.tight_layout()
        self.canvas.draw_idle()
        if self.studio is not None and self.studio.winfo_exists():
            self.studio.set_data(self.studio_data())

    def export(self):
        if self.points is None or not len(self.points[0]):
            messagebox.showinfo("Nothing to export", "Load the theoretical and experimental CSVs first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                            initialfile="activity_coefficients.csv")
        if not path:
            return
        try:
            pd.DataFrame({"MFS": self.points[0], "Activity coefficient": self.points[1]}).to_csv(
                path, index=False, float_format="%.10g")
        except OSError as error:
            messagebox.showerror("Export failed", str(error))
            return
        self.info.set(f"Exported {len(self.points[0])} activity coefficients to {path}")


if __name__ == "__main__":
    window = tk.Tk()
    App(window)
    window.mainloop()
