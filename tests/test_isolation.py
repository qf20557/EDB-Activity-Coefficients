"""The maths modules import without Tk, and stage 1 maths without pandas, so each can be tested on its own."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ["edb.raw_data", "edb.culling", "edb.optics", "edb.fitting", "edb.parsing", "edb.compounds"]
OTHERS = ["edb.fit_io", "edb.mfs", "edb.activity", "edb.aiomfac", "edb.plotting"]


def loaded_after(module):
    code = f"import sys, {module}; print(' '.join(sorted(m.split('.')[0] for m in sys.modules)))"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True, text=True)
    return set(out.stdout.split())


@pytest.mark.parametrize("module", STAGE1 + OTHERS)
def test_no_tk(module):
    assert not {"tkinter", "_tkinter"} & loaded_after(module)


@pytest.mark.parametrize("module", STAGE1)
def test_stage1_maths_without_pandas_or_matplotlib(module):
    assert not {"pandas", "matplotlib"} & loaded_after(module)
