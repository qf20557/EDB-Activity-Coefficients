"""Constants shared by the pipeline stages."""
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMPOUNDS_PATH = ROOT / "compounds.json"

# Big data: traces are read, fitted, culled, drawn and saved block by block, never copied whole
READ_BLOCK_BYTES = 16 << 20  # raw files are parsed in blocks of this many bytes
READ_THREADS = max(1, min(4, (os.cpu_count() or 2) // 2))  # blocks parsed at once
CHUNK_ROWS = 1 << 20  # rows per block for in-memory work
WRITE_ROWS = 100_000  # rows per block when saving (sets how often progress and Cancel are checked)
MAX_PLOT_POINTS = 40_000  # points drawn when 'Plot every Nth point' is auto

# Stage 1 (EDB Plotter)
RADIUS_UNCERTAINTY_UM = 0.1
MAX_ERROR_BARS = 20000  # error bars are thinned to this many; at that density they overlap into a band anyway
MAX_FIT_BINS = 3000
# The exported fit CSV and the fit handed to stage 2 are the same points: at least FIT_EXPORT_POINTS, and on a long
# trace one every FIT_EXPORT_STEP seconds (up to the maximum), so that every stage 2 bin holds plenty of them
FIT_EXPORT_POINTS = 1000
FIT_EXPORT_STEP = 10.0
MAX_FIT_EXPORT_POINTS = 500_000
DICT_TAUS = np.geomspace(0.003, 30.0, 70)
EDB_ASSUMED_RI = 1.335
MEDIAN_OBS_ANGLE = math.pi * (58.725 + 31.275) / (2 * 180)

# Stage 2 (EDB Fit Processing)
MIN_POINTS = 3
MAX_BINS = 20000
AUTO_BIN_SECONDS = 1000.0  # automatic bins are as close to this width as the interval allows
BIN_RESAMPLE_POINTS = 21  # a bin holding fewer than MIN_POINTS fit points is evaluated on this many interpolated ones
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
