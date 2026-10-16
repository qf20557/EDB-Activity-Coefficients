"""Matplotlib drawing shared by the stages (no Tk): axis limits, error bars, activity and publication plots."""
import numpy as np
from matplotlib.colors import is_color_like


def apply_limits(ax, lo, hi, which):
    if lo is None and hi is None:
        return
    current = ax.get_xlim() if which == "x" else ax.get_ylim()
    new = (current[0] if lo is None else lo, current[1] if hi is None else hi)
    (ax.set_xlim if which == "x" else ax.set_ylim)(*new)


def error_bars(ax, x, y, bounds, keep, color, show_x=True, show_y=True, zorder=3, alpha=1.0):
    """Asymmetric error bars from bounds = (x_lo, x_hi, y_lo, y_hi) for the points selected by keep."""
    if not (show_x or show_y):
        return
    x_lo, x_hi, y_lo, y_hi = (np.asarray(b, dtype=float)[keep] for b in bounds)
    x, y = np.asarray(x, dtype=float)[keep], np.asarray(y, dtype=float)[keep]
    ax.errorbar(x, y, xerr=[x - x_lo, x_hi - x] if show_x else None, yerr=[y - y_lo, y_hi - y] if show_y else None,
                fmt="none", ecolor=color, elinewidth=0.8, capsize=2, zorder=zorder, alpha=alpha)


def _open_to_top(ax, values):
    """An open (infinite) upper bound is drawn to the top of the axes; call after the limits are set."""
    values = np.asarray(values, dtype=float)
    return np.where(np.isposinf(values), ax.get_ylim()[1], values)


def pvap_band(ax, x, unc, keep, color="grey", alpha=0.25, label="p_vap range"):
    """Systematic p_vap range as a light band behind the points (sorted by MFS)."""
    x = np.asarray(x, dtype=float)
    keep = np.asarray(keep, dtype=bool) & np.isfinite(unc["lower"]) & ~np.isnan(unc["upper"])
    if not keep.any():
        return
    order = np.argsort(x[keep], kind="stable")
    lo, hi = unc["lower"][keep][order], _open_to_top(ax, unc["upper"][keep][order])
    ax.fill_between(x[keep][order], lo, hi, color=color, alpha=alpha, linewidth=0, zorder=1, label=label)


def combined_bars(ax, x, y, unc, keep, color, alpha=1.0):
    keep = np.asarray(keep, dtype=bool) & np.isfinite(unc["combined_lower"]) & ~np.isnan(unc["combined_upper"])
    if keep.any():
        # an open bound on a point above the axes would end below the point: zero-length bar instead
        upper = np.maximum(_open_to_top(ax, unc["combined_upper"]), np.asarray(y, dtype=float))
        bounds = (x, x, unc["combined_lower"], upper)
        error_bars(ax, x, y, bounds, keep, color, show_x=False, zorder=2.5, alpha=alpha)


def pvap_values(unc, band, combined):
    """y values a shown band / combined bars add to the automatic limits (finite only)."""
    if unc is None:
        return []
    keys = (("lower", "upper") if band else ()) + (("combined_lower", "combined_upper") if combined else ())
    return [np.asarray(unc[k], dtype=float) for k in keys]


def draw_activity(ax, label, points=None, curve=None, limits=None, bounds=None, show_x=True, show_y=True,
                  unc=None, show_band=True, show_combined=False):
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
            values += pvap_values(unc, show_band, show_combined)
    values = [v[np.isfinite(v) & (v > 0)] for v in values]
    values = np.concatenate(values) if values else np.array([])
    if values.size:
        ax.set_ylim(10 ** np.floor(np.log10(values.min()) - 0.05), 10 ** np.ceil(np.log10(values.max()) + 0.05))
    xmin, xmax, ymin, ymax = limits or (None, None, None, None)
    apply_limits(ax, xmin, xmax, "x")
    apply_limits(ax, ymin, ymax, "y")
    if has_points and unc is not None:
        keep = points[1] > 0
        if show_band:
            pvap_band(ax, points[0], unc, keep)
        if show_combined:
            combined_bars(ax, *points, unc, keep, "tab:red", 0.7)
    if curve is not None or has_points:
        ax.legend(frameon=False, loc="upper left")


def _has_data(series):
    return series.get("x") is not None and len(series["x"]) > 0


def auto_axis_limits(series, xscale, yscale, xerr=True, yerr=True, band=False, combined=False):
    """Default (xlim, ylim) from the data. Point series set the x range; lines only if there are no points.
    Shown error bars (xerr / yerr), p_vap band and combined bars are included in the range."""
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
        values += pvap_values(s.get("pvap"), band, combined)
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
        for key in ("color", "bar_color", "band_color", "comb_color"):
            if not is_color_like(style[key]):
                raise ValueError(f"'{style[key]}' is not a valid colour (use a name such as red, or a hex code like #1f77b4).")
        for key in ("bar_alpha", "band_alpha", "comb_alpha"):
            if not 0 <= style[key] <= 1:
                raise ValueError(f"{name} alpha values must be between 0 and 1.")
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
                    error_bars(ax, s["x"], s["y"], s["bounds"], keep, st["bar_color"], o["xerr"], o["yerr"],
                               alpha=st["bar_alpha"])
    xlim, ylim = auto_axis_limits(series, o["xscale"], o["yscale"], o["xerr"], o["yerr"], o["band"], o["combined"])
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    apply_limits(ax, o["xmin"], o["xmax"], "x")
    apply_limits(ax, o["ymin"], o["ymax"], "y")
    for s in series:  # drawn once the limits are set, so an open upper bound reaches the top of the axes
        if s["kind"] != "points" or not _has_data(s) or s.get("pvap") is None:
            continue
        st = o["series"][s["name"]]
        y = np.asarray(s["y"], dtype=float)
        keep = y > 0 if o["yscale"] == "log" else np.ones(len(y), dtype=bool)
        if o["band"]:
            pvap_band(ax, s["x"], s["pvap"], keep, st["band_color"], st["band_alpha"], st["band_label"])
        if o["combined"]:
            combined_bars(ax, s["x"], y, s["pvap"], keep, st["comb_color"], st["comb_alpha"])
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
