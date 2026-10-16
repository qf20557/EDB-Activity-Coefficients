"""Tk interface: one module per stage, assembled by app.PipelineApp."""
import matplotlib

from ..config import SAVE_DPI

matplotlib.rcParams["savefig.dpi"] = SAVE_DPI  # every save (including toolbar save buttons) is 1200 dpi
