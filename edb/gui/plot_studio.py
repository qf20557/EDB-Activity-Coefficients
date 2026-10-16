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
            if values:
                for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                    widget.bind(sequence, wheel)  # scroll the column, not the combobox value
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
                entry(s["name"] + ".bar_color", "Radius bar colour (blank = marker)", "", COLOURS)
                entry(s["name"] + ".bar_alpha", "Radius bar alpha (0-1)", 1.0)
                entry(s["name"] + ".band_label", "p_vap band legend label", "p_vap range")
                entry(s["name"] + ".band_color", "p_vap band colour", "grey", COLOURS)
                entry(s["name"] + ".band_alpha", "p_vap band alpha (0-1)", 0.25)
                entry(s["name"] + ".comb_color", "Combined bar colour", "red", COLOURS)
                entry(s["name"] + ".comb_alpha", "Combined bar alpha (0-1)", 0.7)
        header("Figure")
        entry("width", "Width (in)", "6.5")
        entry("height", "Height (in)", "5")
        self.legend = tk.BooleanVar(value=True)
        self.grid_on = tk.BooleanVar(value=False)
        self.xerr = tk.BooleanVar(value=show_xerr)
        self.yerr = tk.BooleanVar(value=show_yerr)
        self.band = tk.BooleanVar(value=show_band)
        self.combined = tk.BooleanVar(value=show_combined)
        for text, var in (("Show legend", self.legend), ("Show grid", self.grid_on),
                          ("Show x error bars (radius uncertainty)", self.xerr),
                          ("Show y error bars (radius uncertainty)", self.yerr),
                          ("Show p_vap band (systematic)", self.band),
                          ("Show combined y error bars (radius + p_vap)", self.combined)):
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
        """data: {series name: (x, y), (x, y, bounds), (x, y, bounds, pvap) or None}; pvap is pvap_bounds output.
        Style edits are kept."""
        for s in self.series:
            xy = data.get(s["name"]) or (None, None)
            s["x"], s["y"] = xy[:2]
            s["bounds"] = xy[2] if len(xy) > 2 else None
            s["pvap"] = xy[3] if len(xy) > 3 else None
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
            color = get(n + ".color").strip()
            text = lambda key, default: get(n + key).strip() if n + key in self.v else default
            alpha = lambda key, default: self.number(n + key, f"{n} alpha") if n + key in self.v else default
            styles[n] = dict(
                bar_color=text(".bar_color", "") or color, bar_alpha=alpha(".bar_alpha", 1.0),
                band_label=text(".band_label", "p_vap range"), band_color=text(".band_color", "grey"),
                band_alpha=alpha(".band_alpha", 0.25), comb_color=text(".comb_color", "red"),
                comb_alpha=alpha(".comb_alpha", 0.7),
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
