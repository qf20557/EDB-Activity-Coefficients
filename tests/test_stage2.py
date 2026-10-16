"""Stage 2 maths in isolation: reading stage 1 exports, the MFS model and binned gradients."""
import numpy as np
import pandas as pd
import pytest

from edb.config import RADIUS_UNCERTAINTY_UM
from edb.fit_io import read_edb_csv, read_fit_csv
from edb.mfs import bin_edges, interval_results, make_mfs_function

RECORDS = {"A": dict(density_g_ml=1.6, molar_mass_g_mol=104.0), "B": dict(density_g_ml=1.1, molar_mass_g_mol=1228.0),
           "W": dict(density_g_ml=1.0, molar_mass_g_mol=18.0)}


def test_read_fit_csv(tmp_path):
    t = np.linspace(0, 10, 20)
    path = tmp_path / "fit.csv"
    pd.DataFrame({"time_s": t, "fitted_radius_corrected_um": 5 - 0.1 * t,
                  "radius_correction_factor": 1.02}).to_csv(path, index=False)
    t2, r2, sigma = read_fit_csv(path)
    np.testing.assert_allclose(r2, (5 - 0.1 * t) ** 2)
    assert sigma == pytest.approx(RADIUS_UNCERTAINTY_UM * 1.02)
    pd.DataFrame({"time_s": t, "fitted_radius_um": 5 - 0.1 * t}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no corrected radius column"):
        read_fit_csv(path)


def test_read_edb_csv_sigma_from_ratio(tmp_path):
    path = tmp_path / "clean.csv"
    pd.DataFrame({"time_s": [0, 1, 2], "radius_raw_um": [5.0, 4.9, 4.8],
                  "radius_corrected_um": [5.1, 4.998, 4.896]}).to_csv(path, index=False)
    t, r, sigma = read_edb_csv(path)
    assert len(t) == 3 and sigma == pytest.approx(RADIUS_UNCERTAINTY_UM * 1.02)


def test_mfs_model():
    mfs, info = make_mfs_function(RECORDS, "A", "B", "W", 1.0, 1.0, 10.0, 20.0)
    assert mfs(20.0 ** 2) == pytest.approx(info["x_a0"])
    assert mfs(info["r_dry"] ** 2) == pytest.approx(info["x_a0"])
    assert mfs((0.95 * info["r_dry"]) ** 2) < info["x_a0"]  # below the dry radius A evaporates
    with pytest.raises(ValueError):
        make_mfs_function(RECORDS, "A", "A", "W", 1.0, 1.0, 10.0, 20.0)


def test_interval_gradients():
    t = np.linspace(0, 100, 101)
    r2 = 400 - 0.5 * t
    mfs = lambda values: np.full(len(values), 0.3)
    rows, skipped = interval_results(t, [(r2, mfs)] * 3, 0, 100, 25)
    assert len(rows) == 4 and skipped == 0
    np.testing.assert_allclose(np.asarray(rows)[:, [1, 3, 5]], -0.5)
    np.testing.assert_allclose(np.asarray(rows)[:, [0, 2, 4]], 0.3)
    np.testing.assert_array_equal(bin_edges(0, 10, 4), [0, 4, 8, 10])
