"""Stage 1 Streamlit UI: Clean & fit."""
import csv
from pathlib import Path
import time

import numpy as np
import streamlit as st

from ..config import EDB_ASSUMED_RI, RADIUS_UNCERTAINTY_UM
from ..culling import PointStore
from ..fitting import export_points, fit_curve, fit_trace
from ..optics import correction_factor
from ..parsing import file_size, parse_limits, parse_number
from ..raster import build_stage1_figure
from ..raw_data import read_files, write_cleaned
from .widgets import ask_open_files, ask_save_file, extract_selection_box

MODE_LABELS = {
    "fit": "Fit region",
    "zoom": "Zoom",
    "cull": "Cull",
    "keep": "Keep box",
}

MODE_HELP = {
    "fit": "Fit region: drag horizontally to fit that time region.",
    "zoom": "Zoom: drag a rectangle to zoom immediately.",
    "cull": "Cull: drag a rectangle to remove all points inside it immediately.",
    "keep": "Keep box: drag a rectangle to remove all points outside it.",
}


def init_stage1_state():
    defaults = {
        "s1_files": [],  # ordered list of file path strings
        "s1_selected": set(),  # set of selected file path strings in the list
        "s1_last_dir": None,
        "s1_store": None,
        "s1_fit": None,
        "s1_squared": False,
        "s1_use_err": False,
        "s1_compound": "",
        "s1_allow_no_ri": False,
        "s1_show_raw": True,
        "s1_show_corr": False,
        "s1_mode": "fit",
        "s1_fit_lo": "",
        "s1_fit_hi": "",
        "s1_xmin": "",
        "s1_xmax": "",
        "s1_ymin": "",
        "s1_ymax": "",
        "s1_nonce": 0,
        "s1_status": (
            "Add raw EDB files or cleaned data (in acquisition order), then click Load / plot. "
            "Files of several GB are supported and all points are plotted."
        ),
        "s1_fit_text": "Fit: none yet.",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _factor_for(compounds, name):
    info = compounds.get(name, {})
    n = info.get("refractive_index")
    if n is not None and n > 0:
        return correction_factor(float(n))
    return 1.0


def _fit_scale(compounds):
    if st.session_state["s1_show_corr"]:
        return _factor_for(compounds, st.session_state["s1_compound"])
    return 1.0


def _cull_scale(compounds):
    if st.session_state["s1_show_corr"] and not st.session_state["s1_show_raw"]:
        return _factor_for(compounds, st.session_state["s1_compound"])
    return 1.0


def _run_fit(compounds):
    store = st.session_state["s1_store"]
    if store is None:
        st.session_state["s1_fit_text"] = "Load data first."
        return
    try:
        lo = parse_number(st.session_state["s1_fit_lo"], "Fit from", optional=True)
        hi = parse_number(st.session_state["s1_fit_hi"], "Fit to", optional=True)
        scale = _fit_scale(compounds)
        fit = fit_trace(store, lo, hi, scale)
        fit["scale"] = scale
        st.session_state["s1_fit"] = fit
        basis = "corrected radius" if scale != 1.0 else "radius"
        st.session_state["s1_fit_text"] = (
            f"{fit['label']} to {basis}, {fit['lo']:.6g}-{fit['hi']:.6g} s, {fit['n']:,} retained points. "
            f"RMS residual {fit['rms']:.4g} µm, R² {fit['r2']:.6f}. {fit['detail']}"
        )
    except ValueError as err:
        st.session_state["s1_fit"] = None
        st.session_state["s1_fit_text"] = f"Fit failed: {err}"


def _on_basis_or_compound_change(compounds):
    if not (st.session_state["s1_show_raw"] or st.session_state["s1_show_corr"]):
        st.session_state["s1_show_raw"] = True
    name = st.session_state["s1_compound"]
    if name and compounds.get(name, {}).get("refractive_index") is None:
        st.session_state["s1_status"] = f"No refractive_index for {name} in compounds.json; factor = 1."
    if st.session_state["s1_store"] is not None and st.session_state["s1_fit"] is not None:
        _run_fit(compounds)
    st.session_state["s1_ymin"] = ""
    st.session_state["s1_ymax"] = ""


def _on_squared_change():
    st.session_state["s1_ymin"] = ""
    st.session_state["s1_ymax"] = ""


def _on_clear_region(compounds):
    st.session_state["s1_fit_lo"] = ""
    st.session_state["s1_fit_hi"] = ""
    _run_fit(compounds)


def _after_edit(compounds):
    store = st.session_state["s1_store"]
    total, kept = len(store), store.count
    st.session_state["s1_status"] = (
        f"{kept:,} retained / {total:,} loaded points; {total - kept:,} culled. Source files unchanged."
    )
    if st.session_state["s1_fit"] is not None:
        _run_fit(compounds)


def _on_undo(compounds):
    store = st.session_state["s1_store"]
    if store is not None and store.undo():
        _after_edit(compounds)


def _on_restore(compounds):
    store = st.session_state["s1_store"]
    if store is not None:
        store.restore()
        _after_edit(compounds)


def _on_reset_zoom():
    st.session_state["s1_xmin"] = ""
    st.session_state["s1_xmax"] = ""
    st.session_state["s1_ymin"] = ""
    st.session_state["s1_ymax"] = ""
    st.session_state["s1_nonce"] += 1


def _on_mode_change():
    st.session_state["s1_status"] = MODE_HELP[st.session_state["s1_mode"]]


def _on_plot_select(compounds):
    box = extract_selection_box(st.session_state.get("s1_plot"))
    st.session_state["s1_nonce"] += 1
    store = st.session_state["s1_store"]
    if box is None or store is None:
        return
    xmin, xmax, ymin, ymax = box
    mode = st.session_state["s1_mode"]
    if mode == "fit":
        if xmax > xmin:
            st.session_state["s1_fit_lo"] = f"{xmin:.8g}"
            st.session_state["s1_fit_hi"] = f"{xmax:.8g}"
            _run_fit(compounds)
    elif mode == "zoom":
        if xmax > xmin and ymax > ymin:
            st.session_state["s1_xmin"] = f"{xmin:.8g}"
            st.session_state["s1_xmax"] = f"{xmax:.8g}"
            st.session_state["s1_ymin"] = f"{ymin:.8g}"
            st.session_state["s1_ymax"] = f"{ymax:.8g}"
    elif mode in ("cull", "keep"):
        try:
            removed = store.cull_box(
                (xmin, xmax, ymin, ymax),
                st.session_state["s1_squared"],
                keep_inside=(mode == "keep"),
                scale=_cull_scale(compounds),
            )
        except ValueError as err:
            st.session_state["s1_status"] = f"Cull: {err}"
            return
        if removed:
            _after_edit(compounds)


def _check_correction(compounds, action):
    name = st.session_state["s1_compound"]
    if name not in compounds:
        return None, (
            f"Choose the droplet compound (Radius correction) before {action}: "
            "stage 2 and the exported files always use the corrected radius."
        )
    if compounds[name].get("refractive_index") is None and not st.session_state["s1_allow_no_ri"]:
        return None, (
            f"'{name}' has no refractive_index in compounds.json (factor = 1.0). "
            "Tick 'Confirm factor = 1.0' under Radius correction to continue."
        )
    return _factor_for(compounds, name), None


def _corrected_fit(compounds):
    fit = st.session_state["s1_fit"]
    factor = _factor_for(compounds, st.session_state["s1_compound"])
    if fit["scale"] == factor:
        return fit
    refit = fit_trace(st.session_state["s1_store"], fit["lo"], fit["hi"], factor)
    refit["scale"] = factor
    return refit


def _write_fit_csv(path, fit, use_err):
    x, radius, radius2 = fit_curve(fit, export_points(fit))
    header = [
        "time_s",
        "fitted_radius_corrected_um",
        "fitted_radius_corrected_squared_um2",
        "radius_correction_factor",
    ]
    columns = [x, radius, radius2, np.full_like(x, fit["scale"])]
    if use_err:
        sigma = RADIUS_UNCERTAINTY_UM * fit["scale"]
        header += [
            "fitted_radius_corrected_err_um",
            "fitted_radius_corrected_lower_um",
            "fitted_radius_corrected_upper_um",
            "fitted_radius_corrected_squared_err_um2",
            "fitted_radius_corrected_squared_lower_um2",
            "fitted_radius_corrected_squared_upper_um2",
        ]
        columns += [
            np.full_like(radius, sigma),
            radius - sigma,
            radius + sigma,
            2 * radius * sigma,
            (radius - sigma) ** 2,
            (radius + sigma) ** 2,
        ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(zip(*columns))


def stage1_handoff(compounds):
    store = st.session_state["s1_store"]
    fit = st.session_state["s1_fit"]
    if store is None:
        return None, "Load EDB data first."
    if fit is None:
        return None, "Fit the data before moving on (click Fit or drag a fit region)."
    factor, err = _check_correction(compounds, "moving on")
    if err:
        return None, err
    try:
        c_fit = _corrected_fit(compounds)
    except ValueError as exc:
        return None, f"Could not fit the corrected radius: {exc}"
    x, _, radius2 = fit_curve(c_fit, export_points(c_fit))
    return dict(
        edb=store.snapshot(),
        fit_t=x,
        fit_r2=radius2,
        factor=factor,
        sigma=RADIUS_UNCERTAINTY_UM * factor,
        refitted=c_fit is not fit,
    ), None


def _toggle_file_select(path, key):
    if st.session_state.get(key):
        st.session_state["s1_selected"].add(path)
    else:
        st.session_state["s1_selected"].discard(path)


def _add_files_click():
    chosen = ask_open_files(
        title="Select EDB files or cleaned data",
        filetypes=[
            ("Data files (*.txt;*.dat;*.csv;*.npz)", "*.txt;*.dat;*.csv;*.npz"),
            ("All files (*.*)", "*.*"),
        ],
        multiple=True,
        initial_dir=st.session_state["s1_last_dir"],
    )
    if chosen:
        st.session_state["s1_last_dir"] = str(Path(chosen[0]).parent)
        files = st.session_state["s1_files"]
        for p in chosen:
            if p not in files:
                files.append(p)
        st.session_state["s1_status"] = (
            f"{len(files)} file(s) listed. Put them in acquisition order, then click Load / plot."
        )


def _remove_selected_click():
    selected = st.session_state["s1_selected"]
    st.session_state["s1_files"] = [p for p in st.session_state["s1_files"] if p not in selected]
    selected.clear()


def _clear_files_click():
    st.session_state["s1_files"].clear()
    st.session_state["s1_selected"].clear()


def _move_file_click(direction):
    files = st.session_state["s1_files"]
    selected = [i for i, p in enumerate(files) if p in st.session_state["s1_selected"]]
    if len(selected) != 1:
        return
    i = selected[0]
    j = i + direction
    if 0 <= j < len(files):
        files[i], files[j] = files[j], files[i]


def render_clean_fit_stage(compounds, on_next_stage):
    init_stage1_state()

    # Top row: 3 side-by-side panels matching the original desktop layout
    col_files, col_opts, col_corr = st.columns([1.25, 1.1, 1.05], gap="medium")

    with col_files:
        with st.container(border=True):
            st.markdown("##### Files (acquisition order)")
            list_col, btn_col = st.columns([1.85, 1.0], gap="small")
            files = st.session_state["s1_files"]

            with list_col:
                with st.container(height=195, border=True):
                    if not files:
                        st.caption("Click 'Add files...' to add EDB files.")
                    else:
                        for idx, path in enumerate(files, 1):
                            p_obj = Path(path)
                            try:
                                sz = f"  ({file_size(p_obj.stat().st_size)})"
                            except OSError:
                                sz = ""
                            chk_key = f"s1_fchk_{path}"
                            st.session_state[chk_key] = path in st.session_state["s1_selected"]
                            st.checkbox(
                                f"{idx:02d}. {p_obj.name}{sz}",
                                key=chk_key,
                                on_change=_toggle_file_select,
                                args=(path, chk_key),
                            )

            selected_indices = [i for i, p in enumerate(files) if p in st.session_state["s1_selected"]]
            can_move_up = len(selected_indices) == 1 and selected_indices[0] > 0
            can_move_down = len(selected_indices) == 1 and selected_indices[0] < len(files) - 1

            with btn_col:
                st.button("Add files...", key="s1_add_btn", on_click=_add_files_click, width="stretch")
                st.button(
                    "Remove selected",
                    key="s1_rem_btn",
                    disabled=not selected_indices,
                    on_click=_remove_selected_click,
                    width="stretch",
                )
                st.button(
                    "Move up",
                    key="s1_up_btn",
                    disabled=not can_move_up,
                    on_click=_move_file_click,
                    args=(-1,),
                    width="stretch",
                )
                st.button(
                    "Move down",
                    key="s1_down_btn",
                    disabled=not can_move_down,
                    on_click=_move_file_click,
                    args=(1,),
                    width="stretch",
                )
                st.button(
                    "Clear",
                    key="s1_clr_btn",
                    disabled=not files,
                    on_click=_clear_files_click,
                    width="stretch",
                )

    with col_opts:
        with st.container(border=True):
            st.markdown("##### Plot and fit")
            st.checkbox(
                "Plot radius² (otherwise radius)",
                key="s1_squared",
                on_change=_on_squared_change,
            )
            st.checkbox(
                "100nm uncertainty (error bars, fit band, CSV error columns)",
                key="s1_use_err",
            )
            f1, f2 = st.columns(2)
            f1.text_input("Fit from (s)", key="s1_fit_lo", placeholder="start")
            f2.text_input("to (s)", key="s1_fit_hi", placeholder="end")
            st.caption("(blank = whole trace; or drag in Fit region mode)")

    with col_corr:
        with st.container(border=True):
            st.markdown("##### Radius correction (`compounds.json`)")
            comp_options = [""] + sorted(compounds)
            st.selectbox(
                "Compound",
                options=comp_options,
                format_func=lambda c: "(select compound...)" if c == "" else c,
                key="s1_compound",
                on_change=_on_basis_or_compound_change,
                args=(compounds,),
            )
            chosen = st.session_state["s1_compound"]
            info = compounds.get(chosen, {})
            ri = info.get("refractive_index")
            rho = info.get("density_g_ml")
            factor = _factor_for(compounds, chosen)

            m1, m2 = st.columns(2)
            m1.caption(f"Refractive index: **{'—' if ri is None else f'{ri:g}'}**")
            m2.caption(f"Density (g/mL): **{'—' if rho is None else f'{rho:g}'}**")

            if chosen and ri is None:
                st.checkbox("Confirm factor = 1.0 (no RI)", key="s1_allow_no_ri")

            c_raw, c_corr = st.columns(2)
            c_raw.checkbox(
                "Plot raw radius",
                key="s1_show_raw",
                on_change=_on_basis_or_compound_change,
                args=(compounds,),
            )
            c_corr.checkbox(
                "Plot corrected radius",
                key="s1_show_corr",
                on_change=_on_basis_or_compound_change,
                args=(compounds,),
            )
            st.caption(
                f"EDB assumed RI {EDB_ASSUMED_RI}; factor **{factor:.4f}**. "
                "Stage 2 and exported fits always use corrected radius."
            )

    # Action bar matching the original desktop pipeline buttons
    a1, a2, a3, a4, a5, a6 = st.columns([1.0, 0.75, 1.0, 1.15, 1.05, 1.3])
    load_clicked = a1.button(
        "Load / plot",
        type="primary",
        key="s1_load_btn",
        disabled=not st.session_state["s1_files"],
        width="stretch",
    )
    a2.button("Fit", key="s1_fit_btn", on_click=_run_fit, args=(compounds,), width="stretch")
    a3.button("Clear fit region", key="s1_clear_fit_btn", on_click=_on_clear_region, args=(compounds,), width="stretch")
    save_clicked = a4.button("Save cleaned data", key="s1_save_btn", width="stretch")
    export_clicked = a5.button("Export fit CSV", key="s1_export_btn", width="stretch")
    next_clicked = a6.button("Next: MFS & gradients →", type="primary", key="s1_next_btn", width="stretch")

    if load_clicked:
        files = list(st.session_state["s1_files"])
        prog = st.progress(0.0, text="Loading 0%...")
        started = time.perf_counter()
        try:
            t, r, skipped = read_files(
                files,
                progress=lambda frac: prog.progress(frac, text=f"Loading {100 * frac:.0f}%..."),
            )
            st.session_state["s1_store"] = PointStore(t, r)
            st.session_state["s1_fit"] = None
            _on_reset_zoom()
            st.session_state["s1_status"] = (
                f"Loaded {len(t):,} points from {len(files)} file(s) in {time.perf_counter() - started:.2f} s; "
                f"skipped {skipped:,} non-data lines. Radius assumed in µm; time starts at 0 s."
            )
            st.session_state["s1_fit_text"] = "Fit: none yet. Drag in Fit region mode or click Fit for the whole trace."
        except Exception as err:
            st.error(f"Load error: {err}")
        finally:
            prog.empty()

    if save_clicked:
        store = st.session_state["s1_store"]
        if store is None:
            st.warning("Load data first.")
        else:
            fac, err = _check_correction(compounds, "saving")
            if err:
                st.warning(err)
            else:
                out_path = ask_save_file(
                    title="Save cleaned data",
                    filetypes=[
                        ("NumPy binary: saves and reloads large data in seconds (*.npz)", "*.npz"),
                        ("CSV (*.csv)", "*.csv"),
                    ],
                    initial_file="Cleaned EDB Data.npz",
                    default_ext=".npz",
                    initial_dir=st.session_state["s1_last_dir"],
                )
                if out_path:
                    target = Path(out_path)
                    st.session_state["s1_last_dir"] = str(target.parent)
                    if any(target.exists() and Path(p).exists() and target.samefile(p) for p in st.session_state["s1_files"]):
                        st.error("Choose an output file different from the inputs.")
                    else:
                        prog = st.progress(0.0, text="Saving 0%...")
                        started = time.perf_counter()
                        snap = store.snapshot()
                        sig = RADIUS_UNCERTAINTY_UM * fac if st.session_state["s1_use_err"] else None
                        try:
                            cnt = write_cleaned(
                                str(target),
                                snap,
                                fac,
                                sig,
                                progress=lambda frac: prog.progress(frac, text=f"Saving {100 * frac:.0f}%..."),
                            )
                            st.session_state["s1_status"] = (
                                f"Saved {cnt:,} retained rows to {target.name} in "
                                f"{time.perf_counter() - started:.2f} s (correction factor {fac:.4f})."
                            )
                        except Exception as exc:
                            st.error(f"Save failed: {exc}")
                        finally:
                            prog.empty()

    if export_clicked:
        if st.session_state["s1_fit"] is None:
            st.warning("Run a fit first.")
        else:
            fac, err = _check_correction(compounds, "exporting the fit")
            if err:
                st.warning(err)
            else:
                try:
                    c_fit = _corrected_fit(compounds)
                    out_path = ask_save_file(
                        title="Export fit CSV",
                        filetypes=[("CSV (*.csv)", "*.csv")],
                        initial_file="Fitted EDB Data.csv",
                        default_ext=".csv",
                        initial_dir=st.session_state["s1_last_dir"],
                    )
                    if out_path:
                        st.session_state["s1_last_dir"] = str(Path(out_path).parent)
                        _write_fit_csv(out_path, c_fit, st.session_state["s1_use_err"])
                        st.session_state["s1_status"] = (
                            f"Corrected-radius fit exported ({export_points(c_fit):,} points, "
                            f"factor {c_fit['scale']:.4f})."
                        )
                except Exception as exc:
                    st.error(f"Export error: {exc}")

    if next_clicked:
        data, err = stage1_handoff(compounds)
        if err:
            st.warning(err)
        else:
            on_next_stage(data)

    _render_stage1_plot_fragment(compounds)


@st.fragment
def _render_stage1_plot_fragment(compounds):
    with st.container(border=True):
        m_col, u_col, ax_col = st.columns([1.6, 1.15, 1.85], vertical_alignment="center")
        with m_col:
            st.radio(
                "Plot mode (drag on the plot)",
                options=list(MODE_LABELS),
                format_func=lambda k: MODE_LABELS[k],
                horizontal=True,
                key="s1_mode",
                on_change=_on_mode_change,
            )
        with u_col:
            u1, u2, u3 = st.columns(3)
            store = st.session_state["s1_store"]
            u1.button(
                "Undo cull",
                key="s1_undo_btn",
                disabled=not (store is not None and store.history),
                on_click=_on_undo,
                args=(compounds,),
                width="stretch",
            )
            u2.button(
                "Restore all",
                key="s1_restore_btn",
                disabled=store is None or store.count == len(store),
                on_click=_on_restore,
                args=(compounds,),
                width="stretch",
            )
            u3.button("Reset zoom", key="s1_reset_zoom_btn", on_click=_on_reset_zoom, width="stretch")

        with ax_col:
            x1, x2, y1, y2 = st.columns(4)
            x1.text_input("X min", key="s1_xmin", placeholder="auto")
            x2.text_input("X max", key="s1_xmax", placeholder="auto")
            y1.text_input("Y min", key="s1_ymin", placeholder="auto")
            y2.text_input("Y max", key="s1_ymax", placeholder="auto")

    st.caption(st.session_state["s1_status"])
    st.markdown(f"**{st.session_state['s1_fit_text']}**")

    try:
        limits = parse_limits([st.session_state[k] for k in ("s1_xmin", "s1_xmax", "s1_ymin", "s1_ymax")])
    except ValueError as err:
        st.warning(f"Axis limits ignored: {err}")
        limits = (None, None, None, None)

    factor = _factor_for(compounds, st.session_state["s1_compound"])
    fig = build_stage1_figure(
        store=st.session_state["s1_store"],
        fit=st.session_state["s1_fit"],
        squared=st.session_state["s1_squared"],
        use_err=st.session_state["s1_use_err"],
        show_raw=st.session_state["s1_show_raw"],
        show_corr=st.session_state["s1_show_corr"],
        factor=factor,
        fit_scale=_fit_scale(compounds),
        mode=st.session_state["s1_mode"],
        limits=limits,
        nonce=st.session_state["s1_nonce"],
    )

    st.plotly_chart(
        fig,
        width="stretch",
        key="s1_plot",
        on_select=lambda: _on_plot_select(compounds),
        selection_mode="box",
        config={"displaylogo": False},
    )
