"""Stage 3: AIOMFAC-web output files."""
import re

import numpy as np
import pandas as pd

from .config import AIOMFAC_COLUMN, AIOMFAC_DIR


def parse_aiomfac(path):
    with open(path, encoding="utf-8-sig", errors="replace") as handle:
        lines = handle.read().splitlines()
    tables, i = {}, 0
    while i < len(lines):
        match = re.match(r"\s*Mixture's component # :\s*(\d+)", lines[i])
        if not match:
            i += 1
            continue
        number = match.group(1).zfill(2)
        i += 1
        while i < len(lines) and not re.match(r"\s*no\.\s", lines[i]):
            i += 1
        if i >= len(lines):
            break
        names = lines[i].split()
        i += 2
        rows = []
        while i < len(lines) and re.match(r"\s*\d+\s", lines[i]):
            parts = lines[i].split()
            if len(parts) == len(names):
                rows.append(parts)
            i += 1
        if rows:
            tables[number] = pd.DataFrame(rows, columns=names).apply(pd.to_numeric, errors="coerce")
    if not tables:
        raise ValueError("No component tables found. Is this an AIOMFAC-web output file?")
    return tables


def aiomfac_simulations(folder=AIOMFAC_DIR):
    """Relative paths of every AIOMFAC output file in the repo's simulations folder (subfolders included)."""
    if not folder.is_dir():
        return []
    return sorted((p.relative_to(folder).as_posix() for p in folder.rglob("*.txt")), key=str.lower)


def aiomfac_curve(tables, component, quantity=AIOMFAC_COLUMN):
    if component not in tables:
        raise ValueError(f"Component {component} not in file (found: {', '.join(tables)}).")
    df = tables[component]
    x_col, y_col = f"x_i({component})", f"{quantity}({component})"
    if x_col not in df or y_col not in df:
        raise ValueError(f"Columns {x_col} / {y_col} not found.")
    keep = np.isfinite(df[x_col]) & np.isfinite(df[y_col])
    flag = [c for c in df.columns if c.startswith("flag")]
    flagged = 0
    if flag:
        bad = df[flag[0]].fillna(0) != 0
        flagged = int((bad & keep).sum())
        keep &= ~bad
    x, y = df.loc[keep, x_col].to_numpy(float), df.loc[keep, y_col].to_numpy(float)
    order = np.argsort(x)
    return x[order], y[order], flagged
