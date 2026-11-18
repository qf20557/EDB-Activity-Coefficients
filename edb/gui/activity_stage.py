"""Stage 3 Streamlit UI: Activity coefficients."""
import io
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import streamlit as st

from ..activity import (
    activity_coefficients,
    activity_frame,
    pvap_bounds,
    pvap_factors,
    pvap_note,
    read_mfs_gradient,
    read_theory_csv,
)
from ..aiomfac import aiomfac_curve, aiomfac_simulations, parse_aiomfac
from ..config import AIOMFAC_DIR
from ..parsing import parse_limits
from ..plotting import draw_activity
from .plot_studio import render_plot_studio
from .widgets import ask_open_files, ask_save_file


def init_stage3_state():
    defaults = {
        "s3_theory": None,
        "s3_theory_pvap": None,
        "s3_experiment": None,
        "s3_tables": None,
        "s3_last_dir": None,
        "s3_sim_name": "",
        "s3_label": "A",
        "s3_xerr": True,
        "s3_yerr": True,
        "s3_band": True,
        "s3_combined": False,
        "s3_xmin": "",
        "s3_xmax": "",
        "s3_ymin": "",
        "s3_ymax": "",
        "s3_source": "Experimental data: none yet (press Next on stage 2, or load a CSV).",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _unc_note(exp):
    if exp[2] is None:
        return ", no radius-uncertainty columns (no error bars)"
    return ", with radius-uncertainty columns (error bars)"


def receive_stage3_data(experiment):
    init_stage3_state()
    st.session_state["s3_experiment"] = experiment
    st.session_state["s3_source"] = (
        f"Experimental data: {len(experiment[0])} interval(s) from stage 2{_unc_note(experiment)}. "
        "Load your theoretical (DropletLab) CSV and choose an AIOMFAC simulation."
    )


def _reset_s3_limits():
    st.session_state["s3_xmin"] = ""
    st.session_state["s3_xmax"] = ""
    st.session_state["s3_ymin"] = ""
    st.session_state["s3_ymax"] = ""


def _set_aiomfac_tables(tables):
    st.session_state["s3_tables"] = tables


def _on_sim_selected():
    sim = st.session_state["s3_sim_name"]
    if sim:
        try:
            _set_aiomfac_tables(parse_aiomfac(AIOMFAC_DIR / sim))
        except Exception as err:
            st.session_state["s3_tables"] = None
            st.error(f"AIOMFAC error: {err}")
    else:
        st.session_state["s3_tables"] = None


def _compute_activity_state():
    notes = []
    points = curve = bounds = unc = None
    experiment = st.session_state["s3_experiment"]
    theory = st.session_state["s3_theory"]
    theory_pvap = st.session_state["s3_theory_pvap"]
    tables = st.session_state["s3_tables"]

    if experiment is not None and theory is None:
        notes.append(f"{len(experiment[0])} experimental interval(s) ready; load the theoretical CSV.")

    if theory is not None and experiment is not None:
        x_exp, g_exp, extra = experiment
        x, gamma, outside, zero, bounds, truncated = activity_coefficients(x_exp, g_exp, *theory, extra)
        points = (x, gamma)
        notes.append(
            f"{len(x)} experimental activity coefficients calculated"
            + (" with radius-uncertainty error bars." if bounds is not None else ".")
        )
        if truncated:
            notes.append(f"{truncated} error bar(s) truncated (shifted MFS outside the theoretical range).")
        if outside:
            notes.append(f"{outside} point(s) outside the theoretical MFS range skipped.")
        if zero:
            notes.append(f"{zero} skipped (theoretical gradient ~ 0).")
        if np.any(gamma <= 0):
            notes.append(f"{int(np.sum(gamma <= 0))} non-positive value(s) not shown on log axis.")
        factors = pvap_factors(x, *theory, theory_pvap)
        if factors is not None:
            unc = pvap_bounds(gamma, bounds, factors, theory_pvap)
            notes.append(pvap_note(unc))

    if tables is not None:
        try:
            cx, cy, flagged = aiomfac_curve(tables, "01", "a_coeff_x")
            curve = (cx, cy)
            if flagged:
                notes.append(f"{flagged} AIOMFAC point(s) with non-zero flag excluded.")
        except Exception as err:
            notes.append(str(err))

    if not notes:
        notes.append("Load the theoretical CSV and choose an AIOMFAC simulation.")

    return points, curve, bounds, unc, " ".join(notes)


def _render_activity_figure(label, points, curve, limits, bounds, xerr, yerr, unc, band, combined):
    fig = Figure(figsize=(6.4, 5.0), dpi=130)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    draw_activity(ax, label, points, curve, limits, bounds, xerr, yerr, unc, band, combined)
    fig.tight_layout()
    canvas.draw()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, facecolor="white")
    return buf.getvalue()


def render_activity_stage(on_back_stage):
    init_stage3_state()

    # Top action bar matching the original desktop layout
    b1, b2, b3, b4 = st.columns([1.2, 1.45, 1.45, 0.8])
    th_clicked = b1.button("Load theoretical CSV", key="s3_load_th_btn", width="stretch")
    exp_clicked = b2.button("Load experimental CSV (override)", key="s3_load_exp_btn", width="stretch")
    export_clicked = b3.button("Export activity coefficients CSV", key="s3_export_btn", width="stretch")
    back_clicked = b4.button("← Back", key="s3_back_btn", width="stretch")

    if th_clicked:
        path = ask_open_files(
            title="Theoretical (DropletLab) CSV",
            filetypes=[("CSV (*.csv)", "*.csv"), ("All files (*.*)", "*.*")],
            multiple=False,
            initial_dir=st.session_state["s3_last_dir"],
        )
        if path:
            st.session_state["s3_last_dir"] = str(Path(path).parent)
            try:
                x, g, pvap = read_theory_csv(path)
                st.session_state["s3_theory"] = (x, g)
                st.session_state["s3_theory_pvap"] = pvap
            except Exception as err:
                st.error(f"Load error: {err}")

    if exp_clicked:
        path = ask_open_files(
            title="Experimental CSV",
            filetypes=[("CSV (*.csv)", "*.csv"), ("All files (*.*)", "*.*")],
            multiple=False,
            initial_dir=st.session_state["s3_last_dir"],
        )
        if path:
            st.session_state["s3_last_dir"] = str(Path(path).parent)
            try:
                exp = read_mfs_gradient(path)
                st.session_state["s3_experiment"] = exp
                st.session_state["s3_source"] = (
                    f"Experimental data: {len(exp[0])} row(s) from {Path(path).name} "
                    f"(overrides stage 2){_unc_note(exp)}."
                )
            except Exception as err:
                st.error(f"Load error: {err}")

    if export_clicked:
        points, _, bounds, unc, _ = _compute_activity_state()
        if points is None or not len(points[0]):
            st.warning("Need experimental data and the theoretical CSV first.")
        else:
            out_path = ask_save_file(
                title="Export activity coefficients CSV",
                filetypes=[("CSV (*.csv)", "*.csv")],
                initial_file="activity_coefficients.csv",
                default_ext=".csv",
                initial_dir=st.session_state["s3_last_dir"],
            )
            if out_path:
                st.session_state["s3_last_dir"] = str(Path(out_path).parent)
                try:
                    activity_frame(*points, bounds, unc).to_csv(out_path, index=False, float_format="%.10g")
                    st.success(f"Exported {len(points[0])} activity coefficients to {out_path}")
                except OSError as err:
                    st.error(f"Export failed: {err}")

    if back_clicked:
        on_back_stage()

    st.caption(st.session_state["s3_source"])

    sims = [""] + aiomfac_simulations()
    st.selectbox(
        "AIOMFAC simulation",
        options=sims,
        format_func=lambda s: "(select AIOMFAC simulation...)" if s == "" else s,
        key="s3_sim_name",
        on_change=_on_sim_selected,
    )

    _render_stage3_interactive()


@st.fragment
def _render_stage3_interactive():
    with st.container(border=True):
        o1, o2, o3, o4, o5 = st.columns([0.85, 1.15, 1.15, 0.9, 1.1], vertical_alignment="center")
        o1.text_input("Label", key="s3_label")
        o2.checkbox("x error bars (radius)", key="s3_xerr")
        o3.checkbox("y error bars (radius)", key="s3_yerr")
        o4.checkbox("p_vap band", key="s3_band")
        o5.checkbox("combined y bars", key="s3_combined")

        l1, l2, l3, l4, l5 = st.columns([1, 1, 1, 1, 0.8], vertical_alignment="bottom")
        l1.text_input("X min", key="s3_xmin", placeholder="auto")
        l2.text_input("X max", key="s3_xmax", placeholder="auto")
        l3.text_input("Y min", key="s3_ymin", placeholder="auto")
        l4.text_input("Y max", key="s3_ymax", placeholder="auto")
        l5.button("Auto limits", key="s3_auto_lim", on_click=_reset_s3_limits, width="stretch")

    points, curve, bounds, unc, status_text = _compute_activity_state()

    try:
        limits = parse_limits(
            [st.session_state[k] for k in ("s3_xmin", "s3_xmax", "s3_ymin", "s3_ymax")],
            positive_y=True,
        )
    except ValueError as err:
        limits = None
        status_text += f" Limits ignored: {err}"

    st.markdown(f"**{status_text}**")
    label = st.session_state["s3_label"].strip() or "A"

    tab_quick, tab_pub = st.tabs(["Activity plot", "Publication plot..."])
    with tab_quick:
        png_bytes = _render_activity_figure(
            label,
            points,
            curve,
            limits,
            bounds,
            st.session_state["s3_xerr"],
            st.session_state["s3_yerr"],
            unc,
            st.session_state["s3_band"],
            st.session_state["s3_combined"],
        )
        st.image(png_bytes, width="content")
        if points is not None and len(points[0]):
            with st.expander(f"Calculated activity coefficients table ({len(points[0])} rows)", expanded=False):
                st.dataframe(activity_frame(*points, bounds, unc), width="stretch", hide_index=True)

    with tab_pub:
        render_plot_studio(curve, points, bounds, unc, default_label=label)
