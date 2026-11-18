"""Main Streamlit application assembling the EDB Pipeline stages."""
from pathlib import Path
import sys

import streamlit as st

from ..compounds import load_compounds
from ..config import COMPOUNDS_PATH, ROOT
from .activity_stage import receive_stage3_data, render_activity_stage
from .clean_fit import render_clean_fit_stage
from .mfs_stage import receive_stage2_data, render_mfs_stage
from .solution_designer import render_solution_designer

STAGE_1 = "1 · Clean & fit"
STAGE_2 = "2 · MFS & gradients"
STAGE_3 = "3 · Activity coefficients"
STAGE_4 = "4 · Solution designer"
STAGES = (STAGE_1, STAGE_2, STAGE_3, STAGE_4)


def in_streamlit_runtime():
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
    except ImportError:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    return get_script_run_ctx() is not None


@st.cache_data(show_spinner=False)
def _cached_compounds(mtime):
    return load_compounds()


def get_compounds():
    try:
        mtime = COMPOUNDS_PATH.stat().st_mtime if COMPOUNDS_PATH.exists() else 0.0
        return _cached_compounds(mtime), None
    except Exception as err:
        return {}, f"Place a valid {COMPOUNDS_PATH.name} beside EDB Pipeline.py. ({err})"


def _switch_stage(target):
    st.session_state["_pending_stage"] = target


def _on_stage_nav_change():
    chosen = st.session_state.get("stage_nav")
    if chosen:
        st.session_state["active_stage"] = chosen


def run_app():
    st.set_page_config(
        page_title="EDB Pipeline: evaporation data → activity coefficients",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px; }
        div[data-testid="stMetric"] { padding: 0.25rem 0.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "active_stage" not in st.session_state:
        st.session_state["active_stage"] = STAGE_1
    pending = st.session_state.pop("_pending_stage", None)
    if pending is not None:
        st.session_state["active_stage"] = pending
        st.session_state["stage_nav"] = pending
    elif "stage_nav" not in st.session_state or not st.session_state["stage_nav"]:
        st.session_state["stage_nav"] = st.session_state["active_stage"]

    compounds, comp_err = get_compounds()
    if comp_err:
        st.error(comp_err)

    h_col, nav_col = st.columns([1.15, 1.85], vertical_alignment="center")
    with h_col:
        st.markdown("### EDB Pipeline · Evaporation → Activity Coefficients")
    with nav_col:
        st.segmented_control(
            "Pipeline stage",
            options=STAGES,
            selection_mode="single",
            key="stage_nav",
            on_change=_on_stage_nav_change,
            label_visibility="collapsed",
        )

    stage = st.session_state["active_stage"]

    def go_stage2(data):
        try:
            receive_stage2_data(data, compounds)
        except ValueError as err:
            st.error(f"Hand-over failed: {err}")
            return
        _switch_stage(STAGE_2)
        st.rerun(scope="app")

    def go_stage3(experiment):
        receive_stage3_data(experiment)
        _switch_stage(STAGE_3)
        st.rerun(scope="app")

    def back_to_stage1():
        _switch_stage(STAGE_1)
        st.rerun(scope="app")

    def back_to_stage2():
        _switch_stage(STAGE_2)
        st.rerun(scope="app")

    if stage == STAGE_1:
        render_clean_fit_stage(compounds, on_next_stage=go_stage2)
    elif stage == STAGE_2:
        render_mfs_stage(compounds, on_back_stage=back_to_stage1, on_next_stage=go_stage3)
    elif stage == STAGE_3:
        render_activity_stage(on_back_stage=back_to_stage2)
    else:
        render_solution_designer(compounds)


def main(script_path=None):
    """Run the Streamlit app directly when inside `streamlit run`, or launch the Streamlit server automatically
    when invoked via `python "EDB Pipeline.py"`."""
    if in_streamlit_runtime():
        run_app()
    else:
        from streamlit.web import cli as stcli

        target = Path(script_path) if script_path else (ROOT / "EDB Pipeline.py")
        sys.argv = ["streamlit", "run", str(target.resolve()), *sys.argv[1:]]
        raise SystemExit(stcli.main())
