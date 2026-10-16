"""AIOMFAC-web parsing in isolation, on every simulation shipped in the repo."""
import numpy as np
import pytest

from edb.aiomfac import aiomfac_curve, aiomfac_simulations, parse_aiomfac
from edb.config import AIOMFAC_DIR

SIMULATIONS = aiomfac_simulations()


def test_simulations_found():
    assert SIMULATIONS and all(name.endswith(".txt") for name in SIMULATIONS)


@pytest.mark.parametrize("name", SIMULATIONS)
def test_parse(name):
    tables = parse_aiomfac(AIOMFAC_DIR / name)
    x, y, flagged = aiomfac_curve(tables, "01")
    assert len(x) and np.all(np.diff(x) >= 0) and np.all(np.isfinite(y))
    with pytest.raises(ValueError):
        aiomfac_curve(tables, "99")
