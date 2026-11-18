"""Stage 3 Publication Plot Studio for Streamlit."""
import io
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import streamlit as st

from ..config import COLOURS, LINE_STYLES, MARKERS, SAVE_DPI
from ..parsing import first_line, parse_number
from ..plotting import draw_publication
from .widgets import ask_save_file


def init_studio_state(default_label="A", show_xerr=True, show_yerr=True, show_band=True, show_combined=False):
    defaults = {
        "ps_title": "",
        "ps_xlabel": rf"$x_{{\mathrm{{{default_label}}}}}$",
        "ps_ylabel": rf"$\gamma_{{\mathrm{{{default_label}}}}}$",
        "ps_font_size": "12",
        "ps_xscale": "linear",
        "ps_yscale": "log",
        "ps_xmin": "",
        "ps_xmax": "",
        "ps_ymin": "",
        "ps_ymax": "",
        "ps_legend": True,
        "ps_grid": False,
        "ps_xerr": show_xerr,
        "ps_yerr": show_yerr,
        "ps_band": show_band,
        "ps_combined": show_combined,
        "ps_advanced": False,
        # AIOMFAC curve
        "ps_aiomfac_label": "AIOMFAC",
        "ps_aiomfac_color": "black",
        "ps_aiomfac_style": "solid",
        "ps_aiomfac_width": "1.2",
        # Experiment points
        "ps_exp_label": "Experiment",
        "ps_exp_color": "black",
        "ps_exp_marker": "square",
        "ps_exp_size": "5",
        "ps_exp_yerr_mode": "bars",
        "ps_exp_rband_label": "radius range",
        "ps_exp_rband_color": "black",
        "ps_exp_rband_alpha": "0.25",
        # Radius error bars
        "ps_exp_bar_color": "black",
        "ps_exp_bar_alpha": "1.0",
        "ps_exp_bar_width": "0.8",
        "ps_exp_bar_style": "solid",
        "ps_exp_bar_capsize": "2",
        "ps_exp_bar_capthick": "1.0",
        # Combined error bars
        "ps_exp_comb_color": "red",
        "ps_exp_comb_alpha": "0.7",
        "ps_exp_comb_width": "0.8",
        "ps_exp_comb_style": "solid",
        "ps_exp_comb_capsize": "2",
        "ps_exp_comb_capthick": "1.0",
        # p_vap band
        "ps_exp_band_label": "p_vap range",
        "ps_exp_band_color": "grey",
        "ps_exp_band_alpha": "0.25",
        # Figure size
        "ps_width": "6.5",
        "ps_height": "5",
        "ps_last_dir": None,
        "ps_last_good": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _num(key, label, optional=False, positive=False):
    val = parse_number(st.session_state[key], label, optional=optional)
    if val is not None and positive and val <= 0:
        raise ValueError(f"{label} must be positive and finite.")
    return val


def _collect_options():
    g = lambda k: str(st.session_state[k]).strip()

    def bar_opts(prefix, label_prefix):
        bs = g(f"ps_exp_{prefix}_style")
        return {
            f"{prefix}_width": _num(f"ps_exp_{prefix}_width", f"Experiment {label_prefix} line width", positive=True),
            f"{prefix}_capsize": _num(f"ps_exp_{prefix}_capsize", f"Experiment {label_prefix} cap size"),
            f"{prefix}_capthick": _num(
                f"ps_exp_{prefix}_capthick", f"Experiment {label_prefix} cap thickness", positive=True
            ),
            f"{prefix}_style": LINE_STYLES.get(bs, bs if bs in LINE_STYLES.values() else "-"),
        }

    aiomfac_style = g("ps_aiomfac_style")
    exp_marker = g("ps_exp_marker")
    exp_color = g("ps_exp_color") or "black"

    styles = {
        "AIOMFAC": dict(
            label=st.session_state["ps_aiomfac_label"],
            color=g("ps_aiomfac_color") or "black",
            style=LINE_STYLES.get(aiomfac_style, aiomfac_style if aiomfac_style in LINE_STYLES.values() else "-"),
            width=_num("ps_aiomfac_width", "AIOMFAC line width", positive=True),
            marker="s",
            size=5.0,
            bar_color="black",
            bar_alpha=1.0,
            band_label="p_vap range",
            band_color="grey",
            band_alpha=0.25,
            comb_color="red",
            comb_alpha=0.7,
            yerr_mode="bars",
            rband_label="radius range",
            rband_color="black",
            rband_alpha=0.25,
            bar_width=0.8,
            bar_capsize=2.0,
            bar_capthick=1.0,
            bar_style="-",
            comb_width=0.8,
            comb_capsize=2.0,
            comb_capthick=1.0,
            comb_style="-",
        ),
        "Experiment": dict(
            label=st.session_state["ps_exp_label"],
            color=exp_color,
            marker=MARKERS.get(exp_marker, exp_marker if exp_marker in MARKERS.values() else "s"),
            size=_num("ps_exp_size", "Experiment marker size", positive=True),
            style="-",
            width=1.2,
            yerr_mode="band" if g("ps_exp_yerr_mode") == "band" else "bars",
            rband_label=g("ps_exp_rband_label") or "radius range",
            rband_color=g("ps_exp_rband_color") or exp_color,
            rband_alpha=_num("ps_exp_rband_alpha", "Experiment radius band alpha"),
            bar_color=g("ps_exp_bar_color") or exp_color,
            bar_alpha=_num("ps_exp_bar_alpha", "Experiment error bar alpha"),
            comb_color=g("ps_exp_comb_color") or "red",
            comb_alpha=_num("ps_exp_comb_alpha", "Experiment combined bar alpha"),
            band_label=g("ps_exp_band_label") or "p_vap range",
            band_color=g("ps_exp_band_color") or "grey",
            band_alpha=_num("ps_exp_band_alpha", "Experiment p_vap band alpha"),
            **bar_opts("bar", "radius bar"),
            **bar_opts("comb", "combined bar"),
        ),
    }

    return dict(
        title=st.session_state["ps_title"],
        xlabel=st.session_state["ps_xlabel"],
        ylabel=st.session_state["ps_ylabel"],
        font_size=_num("ps_font_size", "Font size", positive=True),
        xscale=g("ps_xscale") if g("ps_xscale") in ("linear", "log") else "linear",
        yscale=g("ps_yscale") if g("ps_yscale") in ("linear", "log") else "linear",
        xmin=_num("ps_xmin", "X min", optional=True),
        xmax=_num("ps_xmax", "X max", optional=True),
        ymin=_num("ps_ymin", "Y min", optional=True),
        ymax=_num("ps_ymax", "Y max", optional=True),
        legend=st.session_state["ps_legend"],
        grid=st.session_state["ps_grid"],
        xerr=st.session_state["ps_xerr"],
        yerr=st.session_state["ps_yerr"],
        band=st.session_state["ps_band"],
        combined=st.session_state["ps_combined"],
        series=styles,
        width=_num("ps_width", "Width", positive=True),
        height=_num("ps_height", "Height", positive=True),
    )


def render_publication_bytes(series, options, fmt="png", dpi=130):
    fig = Figure(figsize=(options["width"], options["height"]), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    draw_publication(ax, series, options)
    fig.tight_layout()
    canvas.draw()
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, facecolor="white")
    return buf.getvalue()


def _save_publication_file(path, series, options):
    fig = Figure(figsize=(options["width"], options["height"]), dpi=SAVE_DPI)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    draw_publication(ax, series, options)
    fig.tight_layout()
    canvas.draw()
    fig.savefig(path, dpi=SAVE_DPI, facecolor="white")


def render_plot_studio(curve, points, bounds, unc, default_label="A"):
    init_studio_state(default_label=default_label)

    has_bounds = bounds is not None
    has_pvap = unc is not None

    series = [
        dict(
            name="AIOMFAC",
            kind="line",
            x=None if curve is None else curve[0],
            y=None if curve is None else curve[1],
        ),
        dict(
            name="Experiment",
            kind="points",
            x=None if points is None else points[0],
            y=None if points is None else points[1],
            bounds=bounds,
            pvap=unc,
        ),
    ]

    col_settings, col_preview = st.columns([1.05, 1.35], gap="large")

    with col_settings:
        with st.expander("Text (use $...$ for maths, e.g. $\\gamma_{\\mathrm{A}}$)", expanded=True):
            st.text_input("Title", key="ps_title")
            t1, t2, t3 = st.columns([1.2, 1.2, 0.8])
            t1.text_input("X label", key="ps_xlabel")
            t2.text_input("Y label", key="ps_ylabel")
            t3.text_input("Font size (pt)", key="ps_font_size")

        with st.expander("Axes (limits blank = automatic)", expanded=True):
            s1, s2 = st.columns(2)
            s1.selectbox("X axis", ("linear", "log"), key="ps_xscale")
            s2.selectbox("Y axis", ("log", "linear"), key="ps_yscale")
            l1, l2, l3, l4 = st.columns(4)
            l1.text_input("X min", key="ps_xmin", placeholder="auto")
            l2.text_input("X max", key="ps_xmax", placeholder="auto")
            l3.text_input("Y min", key="ps_ymin", placeholder="auto")
            l4.text_input("Y max", key="ps_ymax", placeholder="auto")

        with st.expander("Plot elements", expanded=True):
            e1, e2 = st.columns(2)
            e1.checkbox("Legend", key="ps_legend")
            e2.checkbox("Grid", key="ps_grid")
            if has_bounds:
                b1, b2 = st.columns(2)
                b1.checkbox("x error (radius uncertainty)", key="ps_xerr")
                b2.checkbox("y error (radius uncertainty)", key="ps_yerr")
            if has_pvap:
                p1, p2 = st.columns(2)
                p1.checkbox("p_vap band (systematic)", key="ps_band")
                p2.checkbox("Combined y error bars (radius + p_vap)", key="ps_combined")
            st.checkbox("Advanced style options (line widths, caps)", key="ps_advanced")

        with st.expander("AIOMFAC (curve)", expanded=False):
            st.text_input("Legend label", key="ps_aiomfac_label")
            st.text_input(f"Colour ({', '.join(COLOURS[:5])}... or #hex)", key="ps_aiomfac_color")
            a1, a2 = st.columns(2)
            a1.selectbox("Line style", tuple(LINE_STYLES), key="ps_aiomfac_style")
            a2.text_input("Line width", key="ps_aiomfac_width")

        with st.expander("Experiment (points)", expanded=False):
            st.text_input("Legend label", key="ps_exp_label")
            st.text_input(f"Colour ({', '.join(COLOURS[:5])}... or #hex)", key="ps_exp_color")
            m1, m2 = st.columns(2)
            m1.selectbox("Marker", tuple(MARKERS), key="ps_exp_marker")
            m2.text_input("Marker size", key="ps_exp_size")

            y_band = st.session_state["ps_exp_yerr_mode"] == "band"
            if has_bounds and st.session_state["ps_yerr"]:
                st.selectbox("Radius y error as", ("bars", "band"), key="ps_exp_yerr_mode")
                y_band = st.session_state["ps_exp_yerr_mode"] == "band"
                if y_band:
                    st.markdown("*Radius band*")
                    rb1, rb2, rb3 = st.columns(3)
                    rb1.text_input("Band label", key="ps_exp_rband_label")
                    rb2.text_input("Band colour", key="ps_exp_rband_color")
                    rb3.text_input("Alpha (0-1)", key="ps_exp_rband_alpha")

            show_bars = has_bounds and (st.session_state["ps_xerr"] or (st.session_state["ps_yerr"] and not y_band))
            if show_bars:
                st.markdown("*Radius error bars*")
                eb1, eb2 = st.columns(2)
                eb1.text_input("Bar colour", key="ps_exp_bar_color")
                eb2.text_input("Bar alpha (0-1)", key="ps_exp_bar_alpha")
                if st.session_state["ps_advanced"]:
                    adv1, adv2, adv3, adv4 = st.columns(4)
                    adv1.text_input("Width", key="ps_exp_bar_width")
                    adv2.selectbox("Style", tuple(LINE_STYLES), key="ps_exp_bar_style")
                    adv3.text_input("Cap size", key="ps_exp_bar_capsize")
                    adv4.text_input("Cap thick", key="ps_exp_bar_capthick")

            if has_pvap and st.session_state["ps_combined"]:
                st.markdown("*Combined error bars*")
                cb1, cb2 = st.columns(2)
                cb1.text_input("Combined colour", key="ps_exp_comb_color")
                cb2.text_input("Combined alpha (0-1)", key="ps_exp_comb_alpha")
                if st.session_state["ps_advanced"]:
                    ca1, ca2, ca3, ca4 = st.columns(4)
                    ca1.text_input("Width", key="ps_exp_comb_width")
                    ca2.selectbox("Style", tuple(LINE_STYLES), key="ps_exp_comb_style")
                    ca3.text_input("Cap size", key="ps_exp_comb_capsize")
                    ca4.text_input("Cap thick", key="ps_exp_comb_capthick")

            if has_pvap and st.session_state["ps_band"]:
                st.markdown("*p_vap band*")
                pb1, pb2, pb3 = st.columns(3)
                pb1.text_input("p_vap label", key="ps_exp_band_label")
                pb2.text_input("p_vap colour", key="ps_exp_band_color")
                pb3.text_input("p_vap alpha (0-1)", key="ps_exp_band_alpha")

        with st.expander("Figure size", expanded=False):
            wh1, wh2 = st.columns(2)
            wh1.text_input("Width (in)", key="ps_width")
            wh2.text_input("Height (in)", key="ps_height")

        save_fig_clicked = st.button(f"Save figure ({SAVE_DPI} dpi)...", type="primary", key="ps_save_btn",
                                     width="stretch")

    with col_preview:
        valid_opts = None
        try:
            opts = _collect_options()
            preview_png = render_publication_bytes(series, opts, fmt="png", dpi=130)
            st.session_state["ps_last_good"] = opts
            valid_opts = opts
            st.caption("Preview up to date.")
        except Exception as err:
            st.warning(f"Not updated: {first_line(err)}")
            if st.session_state["ps_last_good"] is not None:
                try:
                    preview_png = render_publication_bytes(series, st.session_state["ps_last_good"], fmt="png", dpi=130)
                except Exception:
                    preview_png = None
            else:
                preview_png = None

        if preview_png is not None:
            st.image(preview_png, width="content")

        if save_fig_clicked:
            if valid_opts is None:
                st.error("Fix the plot settings error before saving.")
            else:
                out_path = ask_save_file(
                    title="Save publication figure",
                    filetypes=[
                        ("PNG (*.png)", "*.png"),
                        ("TIFF (*.tiff;*.tif)", "*.tiff;*.tif"),
                        ("PDF (vector) (*.pdf)", "*.pdf"),
                        ("SVG (vector) (*.svg)", "*.svg"),
                        ("EPS (vector) (*.eps)", "*.eps"),
                    ],
                    initial_file="activity_coefficient.png",
                    default_ext=".png",
                    initial_dir=st.session_state["ps_last_dir"],
                )
                if out_path:
                    st.session_state["ps_last_dir"] = str(Path(out_path).parent)
                    try:
                        _save_publication_file(out_path, series, valid_opts)
                        st.success(f"Saved at {SAVE_DPI} dpi: {out_path}")
                    except Exception as exc:
                        st.error(f"Save failed: {exc}")
