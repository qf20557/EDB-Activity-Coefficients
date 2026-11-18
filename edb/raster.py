"""Server-side all-points rasterizer and interactive Plotly figure builders for Stages 1 and 2.

Every retained point in a Trace (whether 100 or 100,000,000 points) is projected onto a high-DPI RGBA canvas with
zero stride or decimation, then embedded as a crisp data-coordinate layer inside an interactive Plotly figure.
"""
from collections import OrderedDict
import base64
import io
import math

import numpy as np
from PIL import Image
import plotly.graph_objects as go

from . import config
from .fast import error_envelope_fast, project_points_fast
from .fitting import fit_curve
from .mfs import bin_edges

_PNG_CACHE = OrderedDict()
_MAX_CACHE = 8

# Standard Matplotlib tab10 RGB tuples matching the pipeline's colour conventions
RGB_BLUE = (31, 119, 180)
RGB_GREEN = (44, 160, 44)


def _stamp_grid(grid, vis_count):
    """Apply an adaptive crisp marker stamp on the (H, W) count grid so sparse views show sharp 'x' markers and
    multi-million-point views keep isolated 1-point spikes clearly visible without bloating dense regions."""
    h, w = grid.shape
    if vis_count == 0:
        return grid.astype(np.float32)
    src = grid.astype(np.float32)
    if vis_count <= 30_000:
        # Crisp 5x5 'x' cross marker
        pad = np.pad(src, 2, mode="constant")
        out = pad[2:2 + h, 2:2 + w].copy()
        for d in (-2, -1, 1, 2):
            wgt = 0.85 if abs(d) == 1 else 0.65
            out = np.maximum(out, pad[2 + d:2 + d + h, 2 + d:2 + d + w] * wgt)
            out = np.maximum(out, pad[2 + d:2 + d + h, 2 - d:2 - d + w] * wgt)
        return out
    if vis_count <= 400_000:
        # Crisp 3x3 'x' cross marker
        pad = np.pad(src, 1, mode="constant")
        out = pad[1:1 + h, 1:1 + w].copy()
        for dy, dx in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            out = np.maximum(out, pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] * 0.8)
        return out
    # Millions of points: 3x3 '+' soft halo so 1-point outlier spikes remain unmistakable while dense core stays sharp
    pad = np.pad(src, 1, mode="constant")
    out = pad[1:1 + h, 1:1 + w].copy()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        out = np.maximum(out, pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] * 0.6)
    return out


def _composite_over(dst, rgb, alpha_u8):
    """In-place Porter-Duff source-over compositing of a solid RGB layer with uint8 alpha onto RGBA uint8 `dst`."""
    mask = alpha_u8 > 0
    if not mask.any():
        return
    sa = alpha_u8[mask].astype(np.float32) / 255.0
    da = dst[mask, 3].astype(np.float32) / 255.0
    out_a = sa + da * (1.0 - sa)
    safe_a = np.where(out_a > 0, out_a, 1.0)
    for c in range(3):
        dst[mask, c] = np.clip((rgb[c] * sa + dst[mask, c] * da * (1.0 - sa)) / safe_a, 0, 255).astype(np.uint8)
    dst[mask, 3] = np.clip(out_a * 255.0, 0, 255).astype(np.uint8)


def _density_alpha(stamped):
    """Map positive stamped density counts to crisp uint8 alpha [195..255]."""
    alpha = np.zeros(stamped.shape, dtype=np.uint8)
    pos = stamped > 0
    if not pos.any():
        return alpha
    vals = stamped[pos]
    hi = float(np.log1p(vals.max()))
    if hi <= 0.7:
        alpha[pos] = np.clip(vals * 230.0, 150, 235).astype(np.uint8)
    else:
        norm = np.log1p(vals) / hi
        alpha[pos] = np.clip(185.0 + 70.0 * norm, 140, 255).astype(np.uint8)
    return alpha


def render_trace_png(trace, x_lo, x_hi, y_lo, y_hi, layers, squared=False, err_spec=None,
                     width=config.RASTER_WIDTH, height=config.RASTER_HEIGHT):
    """Render ALL retained points of `trace` for `layers` = [(scale, rgb), ...] into a base64 PNG data URI.
    Returns (data_uri, visible_points_count)."""
    cache_key = (
        id(trace),
        trace.version,
        round(float(x_lo), 9),
        round(float(x_hi), 9),
        round(float(y_lo), 9),
        round(float(y_hi), 9),
        bool(squared),
        tuple((round(float(s), 8), rgb) for s, rgb in layers),
        None if err_spec is None else (round(float(err_spec[0]), 8), round(float(err_spec[1]), 8)),
        int(width),
        int(height),
    )
    cached = _PNG_CACHE.get(cache_key)
    if cached is not None:
        _PNG_CACHE.move_to_end(cache_key)
        return cached

    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    if err_spec is not None:
        err_scale, sigma = err_spec
        env = error_envelope_fast(trace, x_lo, x_hi, y_lo, y_hi, err_scale, sigma, squared, width, height)
        if env is not None:
            col_top, col_bot = env
            err_alpha = np.zeros((height, width), dtype=np.uint8)
            valid_cols = np.flatnonzero(col_bot >= col_top)
            for px in valid_cols:
                err_alpha[col_top[px]:col_bot[px] + 1, px] = 75
            _composite_over(rgba, RGB_BLUE, err_alpha)

    vis_max = 0
    for scale, rgb in layers:
        grid, vis = project_points_fast(trace, x_lo, x_hi, y_lo, y_hi, scale=scale, squared=squared,
                                        width=width, height=height)
        vis_max = max(vis_max, vis)
        if vis > 0:
            stamped = _stamp_grid(grid, vis)
            alpha = _density_alpha(stamped)
            _composite_over(rgba, rgb, alpha)

    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", compress_level=1)
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    _PNG_CACHE[cache_key] = (uri, vis_max)
    if len(_PNG_CACHE) > _MAX_CACHE:
        _PNG_CACHE.popitem(last=False)
    return uri, vis_max


def compute_auto_bounds(trace, scales, squared=False, extra_xy=(), sigma=0.0, user_limits=(None, None, None, None)):
    """Compute (x_lo, x_hi, y_lo, y_hi) spanning all retained points + optional extra curves, respecting user limits."""
    ux_lo, ux_hi, uy_lo, uy_hi = user_limits
    xs_lo, xs_hi, ys_lo, ys_hi = [], [], [], []

    if trace is not None and trace.count > 0:
        cnt, t_min, t_max, r_min, r_max = trace.bounds(ux_lo, ux_hi)
        if cnt == 0:
            cnt, t_min, t_max, r_min, r_max = trace.bounds()
        if cnt > 0:
            xs_lo.append(t_min)
            xs_hi.append(t_max)
            for s in scales:
                lo_r = max(0.0, r_min * s - sigma) if sigma > 0 else r_min * s
                hi_r = r_max * s + sigma if sigma > 0 else r_max * s
                ys_lo.append(lo_r * lo_r if squared else lo_r)
                ys_hi.append(hi_r * hi_r if squared else hi_r)

    for ex_x, ex_y in extra_xy:
        if ex_x is not None and len(ex_x):
            ex_x = np.asarray(ex_x, dtype=float)
            ex_y = np.asarray(ex_y, dtype=float)
            finite = np.isfinite(ex_x) & np.isfinite(ex_y)
            if ux_lo is not None:
                finite &= ex_x >= ux_lo
            if ux_hi is not None:
                finite &= ex_x <= ux_hi
            if finite.any():
                xs_lo.append(float(ex_x[finite].min()))
                xs_hi.append(float(ex_x[finite].max()))
                ys_lo.append(float(ex_y[finite].min()))
                ys_hi.append(float(ex_y[finite].max()))

    x_lo = min(xs_lo) if xs_lo else 0.0
    x_hi = max(xs_hi) if xs_hi else 1.0
    y_lo = min(ys_lo) if ys_lo else 0.0
    y_hi = max(ys_hi) if ys_hi else 1.0

    x_span = max(x_hi - x_lo, 1e-9)
    y_span = max(y_hi - y_lo, 1e-9)
    x_lo, x_hi = x_lo - 0.02 * x_span, x_hi + 0.02 * x_span
    y_lo, y_hi = y_lo - 0.04 * y_span, y_hi + 0.04 * y_span

    if ux_lo is not None:
        x_lo = ux_lo
    if ux_hi is not None:
        x_hi = ux_hi
    if uy_lo is not None:
        y_lo = uy_lo
    if uy_hi is not None:
        y_hi = uy_hi

    if x_hi <= x_lo:
        x_hi = x_lo + 1.0
    if y_hi <= y_lo:
        y_hi = y_lo + 1.0
    return float(x_lo), float(x_hi), float(y_lo), float(y_hi)


def _base_figure(x_lo, x_hi, y_lo, y_hi, xlabel, ylabel, title, select_horizontal=False, nonce=0):
    fig = go.Figure()
    # Invisible 2-point anchor trace so Plotly box/horizontal selection always emits box coordinates with ~0 payload
    fig.add_trace(
        go.Scatter(
            x=[x_lo, x_hi],
            y=[y_lo, y_hi],
            mode="markers",
            marker=dict(size=1, opacity=0.0, color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
            name="_anchor",
            meta=nonce,
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis=dict(
            title=xlabel,
            range=[x_lo, x_hi],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.12)",
            zeroline=False,
            mirror=True,
            ticks="outside",
            showline=True,
            linecolor="rgba(0,0,0,0.45)",
        ),
        yaxis=dict(
            title=ylabel,
            range=[y_lo, y_hi],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.12)",
            zeroline=False,
            mirror=True,
            ticks="outside",
            showline=True,
            linecolor="rgba(0,0,0,0.45)",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        dragmode="select",
        selectdirection="h" if select_horizontal else "any",
        margin=dict(l=65, r=20, t=42, b=52),
        height=520,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0.8)",
            font=dict(size=12),
        ),
    )
    return fig


def build_stage1_figure(store, fit, squared, use_err, show_raw, show_corr, factor, fit_scale, mode, limits, nonce=0):
    """Build the interactive Stage 1 Plotly figure with all retained points rasterized onto the plot."""
    if not (show_raw or show_corr):
        show_raw = True
    layers = []
    scales = []
    if show_raw:
        layers.append((1.0, RGB_BLUE))
        scales.append(1.0)
    if show_corr:
        layers.append((factor, RGB_GREEN))
        scales.append(factor)

    extra_xy = []
    fit_x = fit_y = fit_lo_band = fit_hi_band = None
    if fit is not None:
        fit_x, fit_r, fit_r2 = fit_curve(fit, points=800)
        fit_y = fit_r2 if squared else fit_r
        extra_xy.append((fit_x, fit_y))
        if use_err:
            sig = config.RADIUS_UNCERTAINTY_UM * fit["scale"]
            lo_r, hi_r = np.clip(fit_r - sig, 0.0, None), fit_r + sig
            fit_lo_band = lo_r ** 2 if squared else lo_r
            fit_hi_band = hi_r ** 2 if squared else hi_r
            extra_xy.extend([(fit_x, fit_lo_band), (fit_x, fit_hi_band)])

    err_sigma = config.RADIUS_UNCERTAINTY_UM * fit_scale if use_err else 0.0
    x_lo, x_hi, y_lo, y_hi = compute_auto_bounds(
        store, scales, squared=squared, extra_xy=extra_xy, sigma=err_sigma, user_limits=limits
    )

    name = "Corrected radius" if show_corr and not show_raw else "Radius"
    ylabel = f"{name}² (µm²)" if squared else f"{name} (µm)"
    if store is not None:
        kept = store.count
        culled = len(store) - kept
        title = f"EDB evaporation — {kept:,} retained; {culled:,} culled (all {kept:,} points plotted)"
    else:
        title = "EDB evaporation — no data loaded"

    fig = _base_figure(
        x_lo, x_hi, y_lo, y_hi,
        xlabel="Time since first point (s)",
        ylabel=ylabel,
        title=title,
        select_horizontal=(mode == "fit"),
        nonce=nonce,
    )

    if store is not None and store.count > 0:
        err_spec = (fit_scale, config.RADIUS_UNCERTAINTY_UM * fit_scale) if use_err else None
        uri, _ = render_trace_png(store, x_lo, x_hi, y_lo, y_hi, layers, squared=squared, err_spec=err_spec)
        fig.add_layout_image(
            dict(
                source=uri,
                xref="x",
                yref="y",
                x=x_lo,
                y=y_hi,
                sizex=x_hi - x_lo,
                sizey=y_hi - y_lo,
                sizing="stretch",
                layer="below",
            )
        )
        if show_raw:
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(symbol="x", size=8, color="#1f77b4"),
                    name="Raw",
                )
            )
        if show_corr:
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(symbol="x", size=8, color="#2ca02c"),
                    name="Corrected",
                )
            )

    if fit is not None:
        if use_err and fit_lo_band is not None:
            fig.add_trace(
                go.Scatter(
                    x=fit_x, y=fit_hi_band, mode="lines",
                    line=dict(width=0), hoverinfo="skip", showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=fit_x, y=fit_lo_band, mode="lines",
                    line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(214, 39, 40, 0.22)",
                    hoverinfo="skip", name="Fit ± uncertainty",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=fit_x,
                y=fit_y,
                mode="lines",
                line=dict(color="#d62728", width=2.2),
                name="Monotone convex fit",
            )
        )

    return fig


def build_stage2_figure(edb, edb_scale, fit_t, fit_r2, sigma, interval_lo, interval_hi, bin_width, mode, limits,
                        nonce=0):
    """Build the interactive Stage 2 Plotly figure with all cleaned EDB points and fit + bin edges."""
    extra_xy = []
    fit_lo_band = fit_hi_band = None
    if fit_t is not None and len(fit_t):
        extra_xy.append((fit_t, fit_r2))
        if sigma and sigma > 0:
            r = np.sqrt(fit_r2)
            fit_lo_band = np.clip(r - sigma, 0.0, None) ** 2
            fit_hi_band = (r + sigma) ** 2
            extra_xy.extend([(fit_t, fit_lo_band), (fit_t, fit_hi_band)])

    x_lo, x_hi, y_lo, y_hi = compute_auto_bounds(
        edb, [edb_scale] if edb is not None else [], squared=True, extra_xy=extra_xy, user_limits=limits
    )

    if edb is not None:
        title = f"Radius² vs time — {edb.count:,} cleaned EDB points (all plotted)"
    elif fit_t is not None:
        title = f"Radius² vs time — {len(fit_t):,}-point fit"
    else:
        title = "Radius² vs time — load fit or press Next on stage 1"

    fig = _base_figure(
        x_lo, x_hi, y_lo, y_hi,
        xlabel="Time (s)",
        ylabel="Radius² (µm²)",
        title=title,
        select_horizontal=(mode == "select"),
        nonce=nonce,
    )

    if edb is not None and edb.count > 0:
        uri, _ = render_trace_png(edb, x_lo, x_hi, y_lo, y_hi, [(edb_scale, RGB_BLUE)], squared=True)
        fig.add_layout_image(
            dict(
                source=uri,
                xref="x",
                yref="y",
                x=x_lo,
                y=y_hi,
                sizex=x_hi - x_lo,
                sizey=y_hi - y_lo,
                sizing="stretch",
                layer="below",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(symbol="x", size=7, color="#1f77b4"),
                name="EDB data (cleaned)",
            )
        )

    # Bin edge vertical lines drawn as a single NaN-separated vector trace (instant even for thousands of bins)
    if fit_t is not None and interval_lo is not None and interval_hi is not None:
        try:
            edges = bin_edges(interval_lo, interval_hi, bin_width)
            vis_edges = edges[(edges >= x_lo) & (edges <= x_hi)]
            if len(vis_edges):
                ex = np.repeat(vis_edges, 3).astype(float)
                ey = np.empty(3 * len(vis_edges), dtype=float)
                ey[0::3], ey[1::3], ey[2::3] = y_lo, y_hi, math.nan
                ex[2::3] = math.nan
                fig.add_trace(
                    go.Scatter(
                        x=ex, y=ey, mode="lines",
                        line=dict(color="rgba(255, 140, 0, 0.45)", width=1.0),
                        hoverinfo="skip", name="Bin edges",
                    )
                )
        except ValueError:
            pass

    if fit_t is not None and len(fit_t):
        if sigma and fit_lo_band is not None:
            fig.add_trace(
                go.Scatter(
                    x=fit_t, y=fit_hi_band, mode="lines",
                    line=dict(width=0), hoverinfo="skip", showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=fit_t, y=fit_lo_band, mode="lines",
                    line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(214, 39, 40, 0.20)",
                    hoverinfo="skip", name=f"Fit ± {sigma:.3g} µm radius",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=fit_t, y=fit_r2, mode="lines",
                line=dict(color="#d62728", width=2.4),
                name="Fit",
            )
        )

    return fig
