"""Launch the EDB Solution Designer (Streamlit).

Run with either:
    streamlit run "EDB Solutions.py"
or directly:
    python "EDB Solutions.py"
"""
from pathlib import Path
import sys

import streamlit as st

from edb.gui.app import in_streamlit_runtime
from edb.gui.solution_designer import render_solution_designer


def run_solutions_app():
    st.set_page_config(page_title="EDB Solution Designer", layout="centered")
    st.title("EDB Solution Designer")
    render_solution_designer()


if in_streamlit_runtime():
    run_solutions_app()
elif __name__ == "__main__":
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]
    raise SystemExit(stcli.main())
