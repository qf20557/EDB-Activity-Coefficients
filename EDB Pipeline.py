"""Launch the EDB pipeline (Streamlit).

Run with either:
    streamlit run "EDB Pipeline.py"
or directly:
    python "EDB Pipeline.py"
"""
from pathlib import Path

from edb.gui.app import in_streamlit_runtime, main, run_app

if in_streamlit_runtime():
    run_app()
elif __name__ == "__main__":
    main(Path(__file__))
