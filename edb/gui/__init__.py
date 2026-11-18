"""Streamlit interface: one module per stage, assembled by edb.gui.app."""
import matplotlib

from ..config import SAVE_DPI

matplotlib.use("Agg")
matplotlib.rcParams["savefig.dpi"] = SAVE_DPI
