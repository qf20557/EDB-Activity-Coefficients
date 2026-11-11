"""Stage 3 publication-plot window."""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ..config import COLOURS, LINE_STYLES, MARKERS, SAVE_DPI
from ..parsing import first_line, parse_number
from ..plotting import draw_publication
from .widgets import scroll_column


class PlotStudio(tk.Toplevel):
    def __init__(self, master, series, xlabel, ylabel, filename="figure.png", show_xerr=True, show_yerr=True,
                 show_band=True, show_combined=False):
        super().__init__(master)
        self.title("Publication plot")
        self.geometry("1280x900")
        self.series = series
        self.filename = filename
        self.last_good = None
        self.pending = None
        self.v = {}

        column, form, wheel = scroll_column(self)
        column.pack(side="left", fill="y")
        form.configure(padding=8)
        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True)
        row = [0]
        self.sections = []  # (header label, text, expanded var)
        self.rows = []  # (widgets, expanded var of their section, shown() or None)

        def header(text, expanded=True):
            open_ = tk.BooleanVar(value=expanded)
            label = ttk.Label(form, font=("TkDefaultFont", 10, "bold"), cursor="hand2")
            label.grid(row=row[0], column=0, columnspan=2, sticky="w", pady=(8, 2))
            label.bind("<Button-1>", lambda _: (open_.set(not open_.get()), self.refresh()))
            self.sections.append((label, text, open_))
            row[0] += 1

        def place(widgets, shown=None):
            self.rows.append((widgets, self.sections[-1][2], shown))
            row[0] += 1

        def note(text, shown=None):
            label = ttk.Label(form, text=text, font=("TkDefaultFont", 9, "italic"))
            label.grid(row=row[0], column=0, columnspan=2, sticky="w", pady=(4, 0))
            place((label,), shown)

        def entry(key, text, default, values=None, width=24, shown=None):
            label = ttk.Label(form, text=text)
            label.grid(row=row[0], column=0, sticky="w", pady=1)
            var = tk.StringVar(value=str(default))
            widget = (ttk.Combobox(form, textvariable=var, values=values, width=width - 3) if values
                      else ttk.Entry(form, textvariable=var, width=width))
            widget.grid(row=row[0], column=1, sticky="w", padx=4, pady=1)
            if values:
                for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                    widget.bind(sequence, wheel)  # scroll the column, not the combobox value
            var.trace_add("write", lambda *_: self.schedule())
            self.v[key] = var
            place((label, widget), shown)

        def check(text, var, shown=None):
            box = ttk.Checkbutton(form, text=text, variable=var, command=self.schedule)
            box.grid(row=row[0], column=0, columnspan=2, sticky="w")
            place((box,), shown)

        has = lambda key: lambda: any(s.get(key) is not None for s in self.series)
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
        header("Plot elements")
        self.legend = tk.BooleanVar(value=True)
        self.grid_on = tk.BooleanVar(value=False)
        self.xerr = tk.BooleanVar(value=show_xerr)
        self.yerr = tk.BooleanVar(value=show_yerr)
        self.band = tk.BooleanVar(value=show_band)
        self.combined = tk.BooleanVar(value=show_combined)
        self.advanced = tk.BooleanVar(value=False)
        check("Legend", self.legend)
        check("Grid", self.grid_on)
        check("x error (radius uncertainty)", self.xerr, has("bounds"))
        check("y error (radius uncertainty)", self.yerr, has("bounds"))
        check("p_vap band (systematic)", self.band, has("pvap"))
        check("Combined y error bars (radius + p_vap)", self.combined, has("pvap"))
        check("Advanced style options (line widths, caps)", self.advanced)
        for s in series:
            n = s["name"]
            header(f"{n} ({'curve' if s['kind'] == 'line' else 'points'})")
            entry(n + ".label", "Legend label", s["label"])
            entry(n + ".color", "Colour (name or #hex)", s["color"], COLOURS)
            if s["kind"] == "line":
                entry(n + ".style", "Line style", s.get("style", "solid"), tuple(LINE_STYLES))
                entry(n + ".width", "Line width", s.get("width", 1.2))
                continue
            entry(n + ".marker", "Marker", s.get("marker", "square"), tuple(MARKERS))
            entry(n + ".size", "Marker size", s.get("size", 5))
            shown = self.visibility(s)
            entry(n + ".yerr_mode", "Radius y error as", "bars", ("bars", "band"), shown=shown["yerr"])
            note("Radius band", shown["rband"])
            entry(n + ".rband_label", "Legend label", "radius range", shown=shown["rband"])
            entry(n + ".rband_color", "Colour", s["color"], COLOURS, shown=shown["rband"])
            entry(n + ".rband_alpha", "Alpha (0-1)", 0.25, shown=shown["rband"])
            for prefix, text, color, alpha in (("bar", "Radius error bars", s["color"], 1.0),
                                               ("comb", "Combined error bars", "red", 0.7)):
                basic, advanced = shown[prefix], shown[prefix + "_advanced"]
                note(text, basic)
                entry(f"{n}.{prefix}_color", "Colour", color, COLOURS, shown=basic)
                entry(f"{n}.{prefix}_alpha", "Alpha (0-1)", alpha, shown=basic)
                entry(f"{n}.{prefix}_width", "Line width", 0.8, shown=advanced)
                entry(f"{n}.{prefix}_style", "Line style", "solid", tuple(LINE_STYLES), shown=advanced)
                entry(f"{n}.{prefix}_capsize", "Cap size", 2, shown=advanced)
                entry(f"{n}.{prefix}_capthick", "Cap thickness", 1.0, shown=advanced)
            note("p_vap band", shown["band"])
            entry(n + ".band_label", "Legend label", "p_vap range", shown=shown["band"])
            entry(n + ".band_color", "Colour", "grey", COLOURS, shown=shown["band"])
            entry(n + ".band_alpha", "Alpha (0-1)", 0.25, shown=shown["band"])
        header("Figure size", expanded=False)
        entry("width", "Width (in)", "6.5")
        entry("height", "Height (in)", "5")
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
        self.refresh()
        self.update_plot()

    def visibility(self, s):
        """shown() predicates for the uncertainty settings of points series s: only what is currently drawn."""
        n = s["name"]
        bounds = lambda: s.get("bounds") is not None
        pvap = lambda: s.get("pvap") is not None
        y_band = lambda: self.v[n + ".yerr_mode"].get().strip() == "band"
        bars = lambda: bounds() and (self.xerr.get() or (self.yerr.get() and not y_band()))
        comb = lambda: pvap() and self.combined.get()
        return dict(yerr=lambda: bounds() and self.yerr.get(),
                    rband=lambda: bounds() and self.yerr.get() and y_band(),
                    bar=bars, bar_advanced=lambda: bars() and self.advanced.get(),
                    comb=comb, comb_advanced=lambda: comb() and self.advanced.get(),
                    band=lambda: pvap() and self.band.get())

    def refresh(self):
        """Show only the settings that apply now; collapsed sections hide their rows."""
        for label, text, open_ in self.sections:
            label.configure(text=f"{'▾' if open_.get() else '▸'} {text}")
        for widgets, open_, shown in self.rows:
            visible = open_.get() and (shown is None or shown())
            for widget in widgets:
                widget.grid() if visible else widget.grid_remove()

    def set_data(self, data):
        """data: {series name: (x, y), (x, y, bounds), (x, y, bounds, pvap) or None}; pvap is pvap_bounds output.
        Style edits are kept."""
        for s in self.series:
            xy = data.get(s["name"]) or (None, None)
            s["x"], s["y"] = xy[:2]
            s["bounds"] = xy[2] if len(xy) > 2 else None
            s["pvap"] = xy[3] if len(xy) > 3 else None
        self.schedule()

    def schedule(self):
        self.refresh()
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
            color = get(n + ".color").strip()
            text = lambda key, default: get(n + key).strip() if n + key in self.v else default
            alpha = lambda key, default: self.number(n + key, f"{n} alpha") if n + key in self.v else default
            bars = {}
            for prefix, text_label in (("bar", "radius bar"), ("comb", "combined bar")):
                number = lambda key, label, default, positive=True: (
                    self.number(f"{n}.{prefix}_{key}", f"{n} {text_label} {label}", positive=positive)
                    if f"{n}.{prefix}_{key}" in self.v else default)
                bar_style = text(f".{prefix}_style", "solid")
                bars.update({
                    f"{prefix}_width": number("width", "line width", 0.8),
                    f"{prefix}_capsize": number("capsize", "cap size", 2.0, positive=False),
                    f"{prefix}_capthick": number("capthick", "cap thickness", 1.0),
                    f"{prefix}_style": LINE_STYLES.get(bar_style, bar_style if bar_style in LINE_STYLES.values() else "-")})
            styles[n] = dict(
                bar_color=text(".bar_color", "") or color, bar_alpha=alpha(".bar_alpha", 1.0),
                band_label=text(".band_label", "p_vap range"), band_color=text(".band_color", "grey"),
                band_alpha=alpha(".band_alpha", 0.25), comb_color=text(".comb_color", "") or "red",
                comb_alpha=alpha(".comb_alpha", 0.7), **bars,
                yerr_mode="band" if text(".yerr_mode", "bars") == "band" else "bars",
                rband_label=text(".rband_label", "radius range"), rband_color=text(".rband_color", "") or color,
                rband_alpha=alpha(".rband_alpha", 0.25),
                label=get(n + ".label"), color=color,
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
            band=self.band.get(), combined=self.combined.get(),
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
