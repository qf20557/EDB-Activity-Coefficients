"""Constants shared by the pipeline stages."""
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMPOUNDS_PATH = ROOT / "compounds.json"

# Stage 1 (EDB Plotter)
RADIUS_UNCERTAINTY_UM = 0.1
MAX_ERROR_BARS = 20000  # error bars are thinned to this many; at that density they overlap into a band anyway
MAX_FIT_BINS = 3000
FIT_EXPORT_POINTS = 1000  # points in the exported fit CSV and in the fit handed to stage 2 (kept identical)
DICT_TAUS = np.geomspace(0.003, 30.0, 70)
EDB_ASSUMED_RI = 1.335
MEDIAN_OBS_ANGLE = math.pi * (58.725 + 31.275) / (2 * 180)

# Stage 2 (EDB Fit Processing)
MIN_POINTS = 3
MAX_BINS = 20000
# columns 3-6: the same quantities with the whole radius trace (fit and initial radius) shifted down / up by the
# radius uncertainty; stage 3 turns these into error bars
CSV_COLUMNS = ["Average MFS", "Gradient (um2/s)", "Average MFS (radius - unc)", "Gradient (radius - unc) (um2/s)",
               "Average MFS (radius + unc)", "Gradient (radius + unc) (um2/s)"]
LIMIT_KEYS = (("X min (s)", "x_min"), ("X max (s)", "x_max"), ("Y min", "y_min"), ("Y max", "y_max"))

# Stage 3 (EDB Activity Coefficients)
AIOMFAC_COLUMN = "a_x"
AIOMFAC_DIR = ROOT / "AIOMFAC simulations"
SAVE_DPI = 1200
# theoretical (DropletLab) CSV columns, matched by name (see column_key); older 2-column files are read by position
THEORY_X, THEORY_G = "Average MFS", "Gradient (µm²/s)"
THEORY_SIGMA = "Gradient uncertainty (± µm²/s)"
THEORY_P = ("Acid p_vap at T_ref (Pa)", "Acid p_vap lower (Pa)", "Acid p_vap upper (Pa)")
THEORY_CHECK = ("Gradient lower p_vap at nominal MFS (µm²/s)", "Gradient upper p_vap at nominal MFS (µm²/s)")
PVAP_COLUMNS = ("Activity coefficient p_vap lower", "Activity coefficient p_vap upper",
                "Activity coefficient min (radius, at p_vap upper)", "Activity coefficient max (radius, at p_vap lower)",
                "Activity coefficient combined lower", "Activity coefficient combined upper") + THEORY_P
COLOURS = ("black", "red", "blue", "green", "orange", "purple", "grey", "brown", "teal", "magenta")
MARKERS = {"square": "s", "circle": "o", "triangle": "^", "diamond": "D", "cross (x)": "x", "plus": "+"}
LINE_STYLES = {"solid": "-", "dashed": "--", "dash-dot": "-.", "dotted": ":"}
