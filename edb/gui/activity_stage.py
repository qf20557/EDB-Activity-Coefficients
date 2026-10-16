"""Stage 3 UI: Activity coefficients (EDB Activity Coefficients)."""
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from ..activity import (activity_coefficients, activity_frame, pvap_bounds, pvap_factors, pvap_note, read_mfs_gradient,
                        read_theory_csv)
from ..aiomfac import aiomfac_curve, aiomfac_simulations, parse_aiomfac
from ..config import AIOMFAC_DIR
from ..parsing import parse_limits
from ..plotting import draw_activity
from .plot_studio import PlotStudio
from .widgets import nav_bar, scrollable


class ActivityStage(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.theory = self.theory_pvap = self.experiment = self.tables = None
        self.points = self.curve = self.bounds = self.unc = None
        self.studio = None
        self.source = tk.StringVar(value="Experimental data: none yet (press Next on stage 2, or load a CSV).")

        nav_bar(self, back=lambda: self.app.select(1))
        outer, root = scrollable(self)
        outer.pack(fill="both", expand=True)

        bar = ttk.Frame(root)
        bar.pack(fill="x", padx=4, pady=4)
        for text, command in (("Load theoretical CSV", self.load_theory),
                              ("Load experimental CSV (override)", self.load_experiment),
                              ("Export activity coefficients CSV", self.export),
                              ("Publication plot...", self.open_studio)):
            ttk.Button(bar, text=text, command=command).pack(side="left", padx=2)
        ttk.Label(root, textvariable=self.source, foreground="#555", wraplength=950).pack(anchor="w", padx=6)

        sim = ttk.Frame(root)
        sim.pack(fill="x", padx=4, pady=(3, 0))
        ttk.Label(sim, text="AIOMFAC simulation").pack(side="left", padx=(2, 3))
        self.simulation = tk.StringVar()
        self.simulation_box = ttk.Combobox(sim, textvariable=self.simulation, width=70, state="readonly",
                                           values=aiomfac_simulations(), postcommand=self.list_simulations)
        self.simulation_box.pack(side="left")
        self.simulation_box.bind("<<ComboboxSelected>>", lambda _: self.load_aiomfac())

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
        self.band = tk.BooleanVar(value=True)
        self.combined = tk.BooleanVar(value=False)
        for text, var in (("x error bars (radius)", self.xerr), ("y error bars (radius)", self.yerr),
                          ("p_vap band", self.band), ("combined y bars", self.combined)):
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
                        f"{self.uncertainty_note()}. Load your theoretical (DropletLab) CSV and choose an AIOMFAC simulation.")
        self.refresh()

    def pick(self, title, types):
        return filedialog.askopenfilename(title=title, filetypes=types, parent=self)

    def load_theory(self):
        path = self.pick("Theoretical (DropletLab) CSV", [("CSV", "*.csv"), ("All files", "*")])
        if path:
            def work():
                x, g, self.theory_pvap = read_theory_csv(path)
                self.theory = (x, g)
            self.safe(work)

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

    def list_simulations(self):
        self.simulation_box["values"] = aiomfac_simulations()

    def load_aiomfac(self):
        name = self.simulation.get()
        if name:
            self.safe(lambda: self.set_tables(parse_aiomfac(AIOMFAC_DIR / name)))

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
        points = None if self.points is None else (*self.points, self.bounds, self.unc)
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
                                 self.xerr.get(), self.yerr.get(), self.band.get(), self.combined.get())
        self.studio.set_data(self.studio_data())

    def auto_limits(self):
        for var in self.limit_vars:
            var.set("")
        self.refresh()

    def refresh(self):
        notes = []
        self.points = self.curve = self.bounds = self.unc = None
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
            factors = pvap_factors(x, *self.theory, self.theory_pvap)
            if factors is not None:
                self.unc = pvap_bounds(gamma, self.bounds, factors, self.theory_pvap)
                notes.append(pvap_note(self.unc))
        if self.tables is not None:
            try:
                x, y, flagged = aiomfac_curve(self.tables, self.component.get())
                self.curve = (x, y)
                if flagged:
                    notes.append(f"{flagged} AIOMFAC point(s) with non-zero flag excluded.")
            except Exception as error:
                notes.append(str(error))
        if not notes:
            notes.append("Load the theoretical CSV and choose an AIOMFAC simulation.")
        try:
            limits = parse_limits([var.get() for var in self.limit_vars], positive_y=True)
        except ValueError as error:
            limits = None
            notes.append(f"Limits ignored: {error}")
        self.info.set(" ".join(notes))
        draw_activity(self.ax, self.name.get().strip() or "A", self.points, self.curve, limits, self.bounds,
                      self.xerr.get(), self.yerr.get(), self.unc, self.band.get(), self.combined.get())
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
            activity_frame(*self.points, self.bounds, self.unc).to_csv(path, index=False, float_format="%.10g")
        except OSError as error:
            messagebox.showerror("Export failed", str(error), parent=self)
            return
        self.info.set(f"Exported {len(self.points[0])} activity coefficients to {path}")
