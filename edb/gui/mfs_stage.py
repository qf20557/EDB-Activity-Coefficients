"""Stage 2 Streamlit UI: MFS & gradients."""
import csv
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import streamlit as st

from ..activity import sort_mfs_gradient
from ..config import COMPOUNDS_PATH, CSV_COLUMNS, RADIUS_UNCERTAINTY_UM
from ..fit_io import clean_fit_arrays, read_edb_csv, read_fit_csv
from ..mfs import auto_bin_count, interval_results, make_mfs_function
from ..parsing import parse_bin_width, parse_limits, parse_number
from ..raster import build_stage2_figure
from ..trace import Trace
from .widgets import ask_open_files, ask_save_file, extract_selection_box, match_compound

S2_MODES = {
    "select": "Select interval (drag)",
    "zoom": "Zoom (drag)",
}


def init_stage2_state(compounds):
    names = list(compounds)
    def_a = match_compound(compounds, "Malonic acid") if names else ""
    def_b = match_compound(compounds, "Tween 20") if names else ""
    def_s = match_compound(compounds, "Water") if names else ""
    defaults = {
        "s2_t": None,
        "s2_r2": None,
        "s2_edb": None,
        "s2_edb_scale": 1.0,
        "s2_rows": [],
        "s2_last_dir": None,
        "s2_comp_a": def_a,
        "s2_comp_b": def_b,
        "s2_comp_s": def_s,
        "s2_mass_a": "1.04",
        "s2_mass_b": "1.36",
        "s2_mass_s": "12.0",
        "s2_r0": "",
        "s2_lo": "",
        "s2_hi": "",
        "s2_bin_width": "auto",
        "s2_sigma": f"{RADIUS_UNCERTAINTY_UM:g}",
        "s2_xmin": "",
        "s2_xmax": "",
        "s2_ymin": "",
        "s2_ymax": "",
        "s2_mode": "select",
        "s2_nonce": 0,
        "s2_source": "No data yet: press Next on stage 1, or load a fit CSV and a cleaned EDB file.",
        "s2_status": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def receive_stage2_data(data, compounds):
    init_stage2_state(compounds)
    t, r2 = clean_fit_arrays(data["fit_t"], data["fit_r2"])
    edb = data["edb"]
    scale = data["factor"]
    st.session_state["s2_t"] = t
    st.session_state["s2_r2"] = r2
    st.session_state["s2_edb"] = edb
    st.session_state["s2_edb_scale"] = scale
    st.session_state["s2_lo"] = f"{t[0]:.8g}"
    st.session_state["s2_hi"] = f"{t[-1]:.8g}"
    st.session_state["s2_r0"] = f"{edb.first()[1] * scale:.8g}"
    st.session_state["s2_sigma"] = f"{data['sigma']:.6g}"
    st.session_state["s2_rows"] = []
    _reset_s2_zoom()
    refit = " (refitted on the corrected radius, same window)" if data.get("refitted") else ""
    st.session_state["s2_source"] = (
        f"From stage 1: {edb.count:,} cleaned points and a {len(t):,}-point fit{refit}, "
        f"refractive-index corrected radius (factor {scale:.4f}), uncertainty ±{data['sigma']:.4g} um. "
        "Initial radius set from the first cleaned point (editable)."
    )
    st.session_state["s2_status"] = f"Fit: {len(t):,} points, {t[0]:.6g}-{t[-1]:.6g} s."


def _reset_s2_zoom():
    st.session_state["s2_xmin"] = ""
    st.session_state["s2_xmax"] = ""
    st.session_state["s2_ymin"] = ""
    st.session_state["s2_ymax"] = ""
    st.session_state["s2_nonce"] += 1


def _take_sigma(sigma):
    if sigma is None:
        return f" No uncertainty stored; keeping ±{st.session_state['s2_sigma']} um."
    st.session_state["s2_sigma"] = f"{sigma:.6g}"
    return f" Radius uncertainty ±{sigma:.4g} um taken from the file."


def _set_r0_from_edb():
    edb = st.session_state["s2_edb"]
    if edb is None:
        st.session_state["s2_status"] = "No cleaned EDB data loaded."
        return
    st.session_state["s2_r0"] = f"{edb.first()[1] * st.session_state['s2_edb_scale']:.8g}"


def _set_r0_from_fit():
    t, r2 = st.session_state["s2_t"], st.session_state["s2_r2"]
    if t is None:
        st.session_state["s2_status"] = "No fit loaded."
        return
    st.session_state["s2_r0"] = f"{math.sqrt(r2[0]):.8g}"


def _set_r0_from_dry(compounds):
    t, r2 = st.session_state["s2_t"], st.session_state["s2_r2"]
    if t is None:
        st.session_state["s2_status"] = "No fit loaded."
        return
    try:
        start = parse_number(st.session_state["s2_lo"], "Interval start")
        a = compounds[st.session_state["s2_comp_a"]]
        b = compounds[st.session_state["s2_comp_b"]]
        s = compounds[st.session_state["s2_comp_s"]]
        v_a = parse_number(st.session_state["s2_mass_a"], "Mass A") / a["density_g_ml"]
        v_b = parse_number(st.session_state["s2_mass_b"], "Mass B") / b["density_g_ml"]
        v_s = parse_number(st.session_state["s2_mass_s"], "Solvent mass") / s["density_g_ml"]
    except (ValueError, KeyError) as err:
        st.session_state["s2_status"] = f"Dry-radius anchor: {err}"
        return
    r_dry = math.sqrt(float(np.interp(start, t, r2)))
    r0 = r_dry * ((v_a + v_b + v_s) / (v_a + v_b)) ** (1 / 3)
    st.session_state["s2_r0"] = f"{r0:.8g}"
    st.session_state["s2_status"] = (
        f"r0 set to {r0:.6g} um so the fit radius at {start:g} s ({r_dry:.6g} um) is the dry A+B radius."
    )


def _evaluate_model(compounds):
    if not compounds:
        return None, None, None, 0.0, f"No compounds loaded: place a valid {COMPOUNDS_PATH.name} beside this script.", False
    try:
        args = (
            compounds,
            st.session_state["s2_comp_a"],
            st.session_state["s2_comp_b"],
            st.session_state["s2_comp_s"],
            parse_number(st.session_state["s2_mass_a"], "Mass A"),
            parse_number(st.session_state["s2_mass_b"], "Mass B"),
            parse_number(st.session_state["s2_mass_s"], "Solvent mass"),
        )
        r0 = parse_number(st.session_state["s2_r0"], "Initial radius")
        sigma = parse_number(st.session_state["s2_sigma"], "Radius uncertainty")
        if not 0 <= sigma < r0:
            raise ValueError("Radius uncertainty must be zero or positive and below the initial radius.")
        mfs, info = make_mfs_function(*args, r0)
        mfs_minus = make_mfs_function(*args, r0 - sigma)[0] if sigma else None
        mfs_plus = make_mfs_function(*args, r0 + sigma)[0] if sigma else None
        note = (
            f"Initial dry x(A) = {info['x_a0']:.6g}; calculated dry (A+B) radius = {info['r_dry']:.6g} um. "
            "Below this radius solvent is gone and A is lost. "
            + (f"Radius uncertainty ±{sigma:.4g} um propagated." if sigma else "No radius uncertainty propagated.")
        )
        return mfs, mfs_minus, mfs_plus, sigma, note, True
    except ValueError as err:
        return None, None, None, 0.0, f"Composition: {err}", False


def _scenarios(r2, mfs, mfs_minus, mfs_plus, sigma):
    if not sigma:
        return [(r2, mfs)] * 3
    r = np.sqrt(r2)
    lower = np.where(r > sigma, (r - sigma) ** 2, np.nan)
    return [(r2, mfs), (lower, mfs_minus), ((r + sigma) ** 2, mfs_plus)]


def calculate_stage2_intervals(compounds):
    t, r2 = st.session_state["s2_t"], st.session_state["s2_r2"]
    if t is None:
        return False, "Press Next on stage 1, or load a fit CSV."
    mfs, mfs_minus, mfs_plus, sigma, info_note, ok = _evaluate_model(compounds)
    if not ok:
        return False, info_note
    try:
        lo = parse_number(st.session_state["s2_lo"], "Interval start")
        hi = parse_number(st.session_state["s2_hi"], "Interval end")
        width = parse_bin_width(st.session_state["s2_bin_width"])
        lo, hi = max(lo, float(t[0])), min(hi, float(t[-1]))
        rows, skipped, resampled = interval_results(
            t, _scenarios(r2, mfs, mfs_minus, mfs_plus, sigma), lo, hi, width
        )
    except ValueError as err:
        return False, str(err)

    st.session_state["s2_rows"] = rows
    msg = f"{len(rows)} interval(s) calculated"
    if width is None:
        n_bins = auto_bin_count(lo, hi)
        msg += f" (automatic: {n_bins} equal bin(s) of {(hi - lo) / n_bins:.6g} s)"
    msg += f" with ±{sigma:.4g} um radius uncertainty." if sigma else " (no radius uncertainty)."
    if skipped:
        msg += (
            f" {skipped} skipped: A would be negative there (fitted radius below the B-only radius "
            "for this composition and initial radius)."
        )
    if resampled:
        msg += (
            f" {resampled} bin(s) are narrower than the spacing of the fit points, so the fit was "
            "interpolated inside them."
        )
    partial = sum(any(math.isnan(v) for v in row) for row in rows)
    if partial:
        msg += f" {partial} interval(s) have no value for a shifted radius (A would be negative there)."
    st.session_state["s2_status"] = msg
    return True, None


def _on_s2_plot_select():
    box = extract_selection_box(st.session_state.get("s2_plot"))
    st.session_state["s2_nonce"] += 1
    if box is None:
        return
    xmin, xmax, ymin, ymax = box
    mode = st.session_state["s2_mode"]
    t = st.session_state["s2_t"]
    if mode == "select":
        if t is not None and xmax > xmin:
            st.session_state["s2_lo"] = f"{max(xmin, float(t[0])):.8g}"
            st.session_state["s2_hi"] = f"{min(xmax, float(t[-1])):.8g}"
    elif mode == "zoom":
        if xmax > xmin and ymax > ymin:
            st.session_state["s2_xmin"] = f"{xmin:.8g}"
            st.session_state["s2_xmax"] = f"{xmax:.8g}"
            st.session_state["s2_ymin"] = f"{ymin:.8g}"
            st.session_state["s2_ymax"] = f"{ymax:.8g}"


def _write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)


def stage2_handoff(compounds):
    if not st.session_state["s2_rows"]:
        ok, err = calculate_stage2_intervals(compounds)
        if not ok or not st.session_state["s2_rows"]:
            return None, err or "No valid intervals calculated."
    arr = np.asarray(st.session_state["s2_rows"], dtype=float)
    try:
        return sort_mfs_gradient(arr[:, 0], arr[:, 1], arr[:, 2:]), None
    except ValueError as err:
        return None, str(err)


def render_mfs_stage(compounds, on_back_stage, on_next_stage):
    init_stage2_state(compounds)

    # Top action bar matching the original desktop layout
    b1, b2, b3, b4, b5, b6, b7 = st.columns([1.0, 1.25, 1.15, 0.9, 0.9, 0.75, 1.4])
    load_fit_clicked = b1.button("Load fit CSV", key="s2_load_fit_btn", width="stretch")
    load_edb_clicked = b2.button("Load cleaned EDB data", key="s2_load_edb_btn", width="stretch")
    calc_clicked = b3.button("Calculate intervals", type="primary", key="s2_calc_btn", width="stretch")
    export_clicked = b4.button("Export CSV", key="s2_export_btn", width="stretch")
    clear_clicked = b5.button("Clear results", key="s2_clear_btn", width="stretch")
    back_clicked = b6.button("← Back", key="s2_back_btn", width="stretch")
    next_clicked = b7.button("Next: activity coefficients →", type="primary", key="s2_next_btn", width="stretch")

    if load_fit_clicked:
        path = ask_open_files(
            title="Fit CSV",
            filetypes=[("CSV (*.csv)", "*.csv"), ("All files (*.*)", "*.*")],
            multiple=False,
            initial_dir=st.session_state["s2_last_dir"],
        )
        if path:
            st.session_state["s2_last_dir"] = str(Path(path).parent)
            try:
                t, r2, sigma = read_fit_csv(path)
                st.session_state["s2_t"] = t
                st.session_state["s2_r2"] = r2
                st.session_state["s2_lo"] = f"{t[0]:.8g}"
                st.session_state["s2_hi"] = f"{t[-1]:.8g}"
                st.session_state["s2_rows"] = []
                _reset_s2_zoom()
                st.session_state["s2_source"] = f"Corrected-radius fit loaded from {Path(path).name}." + _take_sigma(sigma)
                st.session_state["s2_status"] = f"Fit: {len(t):,} points, {t[0]:.6g}-{t[-1]:.6g} s."
            except Exception as err:
                st.error(f"Fit CSV error: {err}")

    if load_edb_clicked:
        path = ask_open_files(
            title="Cleaned EDB data",
            filetypes=[("Cleaned data (*.csv;*.npz)", "*.csv;*.npz"), ("All files (*.*)", "*.*")],
            multiple=False,
            initial_dir=st.session_state["s2_last_dir"],
        )
        if path:
            st.session_state["s2_last_dir"] = str(Path(path).parent)
            prog = st.progress(0.0, text="Loading 0%...")
            started = time.perf_counter()
            try:
                t, r, sigma = read_edb_csv(
                    path,
                    progress=lambda frac: prog.progress(frac, text=f"Loading {100 * frac:.0f}%..."),
                )
                edb = Trace(t, r)
                st.session_state["s2_edb"] = edb
                st.session_state["s2_edb_scale"] = 1.0
                t0, r0 = edb.first()
                st.session_state["s2_r0"] = f"{r0:.8g}"
                st.session_state["s2_rows"] = []
                _reset_s2_zoom()
                st.session_state["s2_status"] = (
                    f"EDB file: {len(edb):,} points read in {time.perf_counter() - started:.2f} s. "
                    f"First point t = {t0:.6g} s, corrected radius = {r0:.6g} um (set as initial radius; editable)."
                    + _take_sigma(sigma)
                )
            except Exception as err:
                st.error(f"EDB data error: {err}")
            finally:
                prog.empty()

    if calc_clicked:
        ok, err = calculate_stage2_intervals(compounds)
        if not ok:
            st.error(f"Calculation error: {err}")

    if clear_clicked:
        st.session_state["s2_rows"] = []

    if export_clicked:
        rows = st.session_state["s2_rows"]
        if not rows:
            st.warning("Calculate intervals first.")
        else:
            out_path = ask_save_file(
                title="Export experimental binned gradients CSV",
                filetypes=[("CSV (*.csv)", "*.csv")],
                initial_file="Experimental Data.csv",
                default_ext=".csv",
                initial_dir=st.session_state["s2_last_dir"],
            )
            if out_path:
                st.session_state["s2_last_dir"] = str(Path(out_path).parent)
                try:
                    _write_rows(out_path, rows)
                    st.session_state["s2_status"] = f"Exported {len(rows)} row(s) to {Path(out_path).name}."
                except OSError as err:
                    st.error(f"Export error: {err}")

    if back_clicked:
        on_back_stage()

    if next_clicked:
        exp_data, err = stage2_handoff(compounds)
        if err:
            st.error(err)
        else:
            on_next_stage(exp_data)

    st.caption(st.session_state["s2_source"])
    _render_stage2_interactive(compounds)


@st.fragment
def _render_stage2_interactive(compounds):
    names = list(compounds)

    with st.container(border=True):
        st.markdown("##### Initial (wet) droplet composition")
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            st.selectbox("A: evaporating", options=names, key="s2_comp_a")
            st.text_input("Mass A (g)", key="s2_mass_a")
        with c2:
            st.selectbox("B: conserved", options=names, key="s2_comp_b")
            st.text_input("Mass B (g)", key="s2_mass_b")
        with c3:
            st.selectbox("Solvent", options=names, key="s2_comp_s")
            st.text_input("Mass Solvent (g)", key="s2_mass_s")

        r1, r2_col, r3, r4, r5 = st.columns(5, gap="small")
        r1.text_input("Initial radius (um)", key="s2_r0")
        r2_col.text_input("Interval start (s)", key="s2_lo")
        r3.text_input("Interval end (s)", key="s2_hi")
        r4.text_input("Bin width (s; auto ≈ 1000, 0 = one)", key="s2_bin_width")
        r5.text_input("Radius uncertainty (um)", key="s2_sigma")

        rb1, rb2, rb3, rb_note = st.columns([1.05, 0.95, 1.05, 2.95])
        rb1.button("r₀ = first EDB point", key="s2_r0_edb", on_click=_set_r0_from_edb, width="stretch")
        rb2.button("r₀ = fit at start", key="s2_r0_fit", on_click=_set_r0_from_fit, width="stretch")
        rb3.button("r₀ from dry radius", key="s2_r0_dry", on_click=_set_r0_from_dry, args=(compounds,), width="stretch")
        rb_note.caption(
            "Uncertainty (corrected radius, 0 = off) shifts the whole fit and r₀ down / up together; "
            "the spread gives the error bars in stage 3."
        )

    _, _, _, sigma_val, model_note, model_ok = _evaluate_model(compounds)
    if st.session_state["s2_status"]:
        st.markdown(f"**{st.session_state['s2_status']}**")
    if model_ok:
        st.caption(model_note)
    else:
        st.warning(model_note)

    with st.container(border=True):
        mc1, mc2, ax_col = st.columns([1.4, 0.65, 2.2], vertical_alignment="center")
        with mc1:
            st.radio(
                "Plot mode",
                options=list(S2_MODES),
                format_func=lambda k: S2_MODES[k],
                horizontal=True,
                key="s2_mode",
            )
        with mc2:
            st.button("Reset view", key="s2_reset_view", on_click=_reset_s2_zoom, width="stretch")
        with ax_col:
            ax1, ax2, ax3, ax4 = st.columns(4)
            ax1.text_input("X min (s)", key="s2_xmin", placeholder="auto")
            ax2.text_input("X max (s)", key="s2_xmax", placeholder="auto")
            ax3.text_input("Y min", key="s2_ymin", placeholder="auto")
            ax4.text_input("Y max", key="s2_ymax", placeholder="auto")

    try:
        limits = parse_limits([st.session_state[k] for k in ("s2_xmin", "s2_xmax", "s2_ymin", "s2_ymax")])
    except ValueError as err:
        st.warning(f"Axis limits ignored: {err}")
        limits = (None, None, None, None)

    try:
        int_lo = parse_number(st.session_state["s2_lo"], "start")
        int_hi = parse_number(st.session_state["s2_hi"], "end")
        b_width = parse_bin_width(st.session_state["s2_bin_width"])
    except ValueError:
        int_lo = int_hi = b_width = None

    fig = build_stage2_figure(
        edb=st.session_state["s2_edb"],
        edb_scale=st.session_state["s2_edb_scale"],
        fit_t=st.session_state["s2_t"],
        fit_r2=st.session_state["s2_r2"],
        sigma=sigma_val if model_ok else 0.0,
        interval_lo=int_lo,
        interval_hi=int_hi,
        bin_width=b_width,
        mode=st.session_state["s2_mode"],
        limits=limits,
        nonce=st.session_state["s2_nonce"],
    )

    st.plotly_chart(
        fig,
        width="stretch",
        key="s2_plot",
        on_select=_on_s2_plot_select,
        selection_mode="box",
        config={"displaylogo": False},
    )

    rows = st.session_state["s2_rows"]
    if rows:
        st.dataframe(pd.DataFrame(rows, columns=CSV_COLUMNS), width="stretch", hide_index=True)
