"""Stage 1 maths in isolation: reading EDB files, culling, refractive-index correction and the monotone fit."""
import numpy as np
import pytest

from edb.config import EDB_ASSUMED_RI
from edb.culling import PointStore
from edb.fitting import fit_curve, fit_model
from edb.optics import correction_factor
from edb.raw_data import read_files


def test_read_files_continues_time_across_files(tmp_path):
    first, second = tmp_path / "a.txt", tmp_path / "b.txt"
    first.write_text("time radius\n10 5.0\n11 4.9\n12 4.8\n")
    second.write_text("0, 4.7\n1, 4.6\nbad line\n")
    data, skipped = read_files([first, second])
    np.testing.assert_array_equal(data[:, 0], [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(data[:, 1], [5.0, 4.9, 4.8, 4.7, 4.6])
    assert skipped == 2
    with pytest.raises(ValueError):
        read_files([])


def test_cull_undo_restore():
    store = PointStore(np.column_stack([np.arange(10.0), np.full(10, 2.0)]))
    assert store.cull_box((2, 4, 0, 10), squared=False) == 3
    assert store.cull_box((5, 9, 3.5, 4.5), squared=True) == 5  # radius 2 -> radius² 4
    with pytest.raises(ValueError):
        store.cull_box((0, 9, 0, 10), squared=False)
    assert store.undo() and len(store.ids()) == 7
    store.restore()
    assert store.keep.all() and not store.history


def test_correction_factor():
    assert correction_factor(EDB_ASSUMED_RI) == pytest.approx(1.0)
    assert correction_factor(1.45) != pytest.approx(1.0)


def test_fit_recovers_decay():
    t = np.linspace(0, 500, 400)
    r = 2 + 3 * np.exp(-t / 50)
    fit = fit_model(t, r)
    assert fit["r2"] > 0.99999 and fit["rms"] < 1e-3
    x, radius, radius2 = fit_curve(fit, 50)
    np.testing.assert_allclose(radius, 2 + 3 * np.exp(-x / 50), atol=5e-3)
    np.testing.assert_allclose(radius2, radius ** 2)
    with pytest.raises(ValueError):
        fit_model(t, r, 100, 50)
