"""Vapour-pressure uncertainty in stage 3 (activity coefficients), using the DropletLab file MMA T20.csv."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from edb import activity, config, plotting

ROOT = Path(__file__).resolve().parents[1]
THEORY = Path(__file__).resolve().with_name("MMA T20.csv")
BASELINE_COMMIT = "3d1539d"  # pipeline before the p_vap uncertainty was added


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module




@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    """The pipeline as it was at BASELINE_COMMIT, for the identical-output check."""
    try:
        source = subprocess.run(["git", "show", f"{BASELINE_COMMIT}:EDB Pipeline.py"], cwd=ROOT, check=True,
                                capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git or commit {BASELINE_COMMIT} not available")
    path = tmp_path_factory.mktemp("baseline") / "edb_baseline.py"
    path.write_bytes(source)
    module = load_module("edb_baseline", path)
    module.COMPOUNDS_PATH = ROOT / "compounds.json"
    return module


def experiment(x_theory, g_theory):
    """Synthetic stage 2 output inside the theoretical range: gamma varies from 0.5 to 2, with radius-shifted
    (MFS, gradient) columns as stage 2 exports them."""
    x = np.linspace(0.05, 0.85, 40)
    g_ideal = np.interp(x, x_theory, g_theory)
    g = g_ideal * np.linspace(0.5, 2.0, len(x))
    extra = np.column_stack([x - 0.004, g * 0.93, x + 0.004, g * 1.06])
    return x, g, extra


def gamma_with_pvap(path):
    x_t, g_t, pvap = activity.read_theory_csv(path)
    x_e, g_e, extra = experiment(x_t, g_t)
    x, gamma, outside, zero, bounds, truncated = activity.activity_coefficients(x_e, g_e, x_t, g_t, extra)
    factors = activity.pvap_factors(x, x_t, g_t, pvap)
    return x_t, g_t, pvap, x, gamma, bounds, factors


def without_p_columns(tmp_path):
    df = pd.read_csv(THEORY, encoding="utf-8-sig")
    path = tmp_path / "MMA T20 no p.csv"
    df.drop(columns=list(config.THEORY_P)).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def test_columns_matched_by_name(tmp_path):
    x_t, g_t, pvap = activity.read_theory_csv(THEORY)
    df = pd.read_csv(THEORY, encoding="utf-8-sig")
    assert np.all(np.diff(x_t) >= 0)
    np.testing.assert_array_equal(x_t, np.sort(df["Average MFS"].to_numpy()))
    assert pvap["p"] == pytest.approx(1.1e-3) and pvap["p_lower"] == pytest.approx(0.8e-3)
    assert pvap["p_upper"] == pytest.approx(1.4e-3)
    # shuffled column order gives the same theory curve and uncertainty
    shuffled = tmp_path / "shuffled.csv"
    df[df.columns[::-1]].to_csv(shuffled, index=False, encoding="utf-8-sig")
    x2, g2, pvap2 = activity.read_theory_csv(shuffled)
    for new, old in ((x2, x_t), (g2, g_t), (pvap2["sigma"], pvap["sigma"])):  # rewritten CSV: last-digit rounding
        np.testing.assert_allclose(new, old, rtol=1e-14, atol=0)


def test_source_a_factors():
    x_t, g_t, pvap, x, gamma, bounds, factors = gamma_with_pvap(THEORY)
    f_lo, f_hi, source = factors
    assert source == "a"
    assert np.round(f_lo, 3) == pytest.approx(0.786) and np.all(np.round(f_lo, 3) == 0.786)
    assert np.all(np.round(f_hi, 3) == 1.375)
    unc = activity.pvap_bounds(gamma, bounds, factors, pvap)
    np.testing.assert_allclose(unc["lower"], gamma * 1.1e-3 / 1.4e-3, rtol=1e-12)
    np.testing.assert_allclose(unc["upper"], gamma * 1.1e-3 / 0.8e-3, rtol=1e-12)
    note = activity.pvap_note(unc)
    assert "p_vap 1.1e-3 Pa (0.8-1.4e-3)" in note and "gamma x0.786-x1.375, systematic" in note
    assert "source a" in note and "Bound-run check" in note


def test_source_b_matches_source_a(tmp_path):
    *_, factors_a = gamma_with_pvap(THEORY)
    x_t, g_t, pvap, x, gamma, bounds, factors_b = gamma_with_pvap(without_p_columns(tmp_path))
    assert pvap["p"] is None and factors_b[2] == "b"
    np.testing.assert_allclose(factors_b[0], factors_a[0], atol=1e-6, rtol=0)
    np.testing.assert_allclose(factors_b[1], factors_a[1], atol=1e-6, rtol=0)
    note = activity.pvap_note(activity.pvap_bounds(gamma, bounds, factors_b, pvap))
    assert "source b" in note and "x0.786-x1.375" in note


@pytest.mark.parametrize("drop_p", [False, True])
def test_bounds_never_symmetric(tmp_path, drop_p):
    path = without_p_columns(tmp_path) if drop_p else THEORY
    x_t, g_t, pvap, x, gamma, bounds, factors = gamma_with_pvap(path)
    unc = activity.pvap_bounds(gamma, bounds, factors, pvap)
    rel = np.interp(x, x_t, pvap["sigma"]) / np.abs(np.interp(x, x_t, g_t))
    assert np.all(unc["upper"] - gamma > gamma - unc["lower"])  # asymmetric, wider upwards
    assert not np.allclose(unc["lower"], gamma * (1 - rel), rtol=1e-3)
    assert not np.allclose(unc["upper"], gamma * (1 + rel), rtol=1e-3)
    np.testing.assert_allclose(unc["lower"], gamma / (1 + rel), rtol=1e-6)
    np.testing.assert_allclose(unc["upper"], gamma / (1 - rel), rtol=1e-6)


def test_radius_bounds_and_combined():
    x_t, g_t, pvap, x, gamma, bounds, factors = gamma_with_pvap(THEORY)
    unc = activity.pvap_bounds(gamma, bounds, factors, pvap)
    np.testing.assert_allclose(unc["radius_lower"], bounds[2] * 1.1 / 1.4, rtol=1e-12)
    np.testing.assert_allclose(unc["radius_upper"], bounds[3] * 1.1 / 0.8, rtol=1e-12)
    d_lo = np.hypot(np.log(gamma / bounds[2]), np.log(1.4 / 1.1))
    d_hi = np.hypot(np.log(bounds[3] / gamma), np.log(1.1 / 0.8))
    np.testing.assert_allclose(unc["combined_lower"], gamma * np.exp(-d_lo), rtol=1e-12)
    np.testing.assert_allclose(unc["combined_upper"], gamma * np.exp(d_hi), rtol=1e-12)
    # quadrature is wider than either part but narrower than the linear (worst-case) envelope
    assert np.all(unc["combined_lower"] < np.minimum(unc["lower"], bounds[2]))
    assert np.all(unc["combined_lower"] > unc["radius_lower"])
    assert np.all(unc["combined_upper"] > np.maximum(unc["upper"], bounds[3]))
    assert np.all(unc["combined_upper"] < unc["radius_upper"])


def test_open_upper_bound_when_rel_at_least_one(tmp_path):
    x_t = np.linspace(0.1, 0.9, 9)
    g_t = -np.ones_like(x_t)
    path = tmp_path / "wide.csv"
    pd.DataFrame({"Average MFS": x_t, "Gradient (µm²/s)": g_t,
                  "Gradient uncertainty (± µm²/s)": np.linspace(0.2, 1.4, 9)}).to_csv(path, index=False,
                                                                                      encoding="utf-8-sig")
    x_t, g_t, pvap = activity.read_theory_csv(path)
    x = np.array([0.2, 0.5, 0.8])
    gamma = np.array([1.0, 1.0, 1.0])
    factors = activity.pvap_factors(x, x_t, g_t, pvap)
    assert np.isfinite(factors[1][0]) and np.isinf(factors[1][2])
    unc = activity.pvap_bounds(gamma, None, factors, pvap)
    assert np.isinf(unc["upper"][2]) and np.isinf(unc["combined_upper"][2])
    assert "Upper bound open (infinite) for 1 point(s)" in activity.pvap_note(unc)
    plotting.draw_activity(Figure().add_subplot(111), "A", (x, gamma), unc=unc, show_band=True, show_combined=True)


def test_plots_draw_with_pvap():
    x_t, g_t, pvap, x, gamma, bounds, factors = gamma_with_pvap(THEORY)
    unc = activity.pvap_bounds(gamma, bounds, factors, pvap)
    ax = Figure().add_subplot(111)
    plotting.draw_activity(ax, "A", (x, gamma), None, None, bounds, True, True, unc, True, True)
    assert len(ax.collections) >= 3  # band, radius bars, combined bars
    assert ax.get_ylim()[0] <= np.min(unc["combined_lower"]) and ax.get_ylim()[1] >= np.max(unc["combined_upper"])
    series = [dict(name="Experiment", kind="points", x=x, y=gamma, bounds=bounds, pvap=unc)]
    style = dict(label="Experiment", color="black", marker="s", size=5, style="-", width=1.2, bar_color="blue",
                 bar_alpha=0.5, band_label="p_vap range", band_color="#cccccc", band_alpha=0.3, comb_color="red",
                 comb_alpha=0.7)
    o = dict(series={"Experiment": style}, xmin=None, xmax=None, ymin=None, ymax=None, xscale="linear",
             yscale="log", font_size=12, xlabel="x", ylabel="y", title="", grid=False, legend=True, xerr=True,
             yerr=True, band=True, combined=True)
    plotting.draw_publication(ax, series, o)
    assert "p_vap range" in ax.get_legend_handles_labels()[1]


def two_column_files(tmp_path):
    df = pd.read_csv(THEORY, encoding="utf-8-sig")
    files = []
    for name, header in (("plain.csv", ["MFS", "Gradient"]), ("named.csv", ["Average MFS", "Gradient (µm²/s)"])):
        path = tmp_path / name
        df.iloc[:, :2].set_axis(header, axis=1).to_csv(path, index=False, encoding="utf-8-sig")
        files.append(path)
    return files


def test_two_column_file_identical_to_baseline(tmp_path, baseline):
    x, g, extra = experiment(*activity.read_theory_csv(THEORY)[:2])
    stage2 = pd.DataFrame(np.column_stack([x, g, extra]), columns=config.CSV_COLUMNS)
    exp_path = tmp_path / "stage2.csv"
    stage2.to_csv(exp_path, index=False)
    for path in two_column_files(tmp_path):
        old_theory = baseline.read_mfs_gradient(path)[:2]
        x_t, g_t, pvap = activity.read_theory_csv(path)
        assert pvap is None
        np.testing.assert_array_equal(x_t, old_theory[0])
        np.testing.assert_array_equal(g_t, old_theory[1])
        exp_new, exp_old = activity.read_mfs_gradient(exp_path), baseline.read_mfs_gradient(exp_path)
        new = activity.activity_coefficients(*exp_new[:2], x_t, g_t, exp_new[2])
        old = baseline.activity_coefficients(*exp_old[:2], *old_theory, exp_old[2])
        for a, b in zip(new, old):
            if isinstance(a, tuple):
                for u, v in zip(a, b):
                    np.testing.assert_array_equal(u, v)
            else:
                np.testing.assert_array_equal(a, b)
        assert activity.pvap_factors(new[0], x_t, g_t, pvap) is None
        # export: the baseline built exactly these columns in ActivityStage.export
        columns = {"MFS": old[0], "Activity coefficient": old[1]}
        columns.update(zip(("MFS min", "MFS max", "Activity coefficient min", "Activity coefficient max"), old[4]))
        expected = pd.DataFrame(columns).to_csv(index=False, float_format="%.10g")
        assert activity.activity_frame(new[0], new[1], new[4], None).to_csv(index=False, float_format="%.10g") == expected


def test_open_upper_bound_with_point_above_axes():
    """Combined bars with open upper bounds still draw when Y max is below a point (it used to raise 'yerr must not
    contain negative values', so the publication plot stopped updating while combined bars were on)."""
    x_t, g_t, pvap, x, gamma, bounds, factors = gamma_with_pvap(THEORY)
    open_upper = np.where(np.arange(len(x)) % 5 == 0, np.inf, factors[1])
    unc = activity.pvap_bounds(gamma, bounds, (factors[0], open_upper, "b"), dict(pvap, p=None))
    ymax = float(np.median(gamma))
    ax = Figure().add_subplot(111)
    plotting.draw_activity(ax, "A", (x, gamma), None, (None, None, None, ymax), bounds, True, True, unc, True, True)
    series = [dict(name="Experiment", kind="points", x=x, y=gamma, bounds=bounds, pvap=unc)]
    style = dict(label="Experiment", color="black", marker="s", size=5, style="-", width=1.2, bar_color="blue",
                 bar_alpha=0.5, band_label="p_vap range", band_color="#cccccc", band_alpha=0.3, comb_color="red",
                 comb_alpha=0.7)
    for yscale in ("log", "linear"):
        o = dict(series={"Experiment": style}, xmin=None, xmax=None, ymin=None, ymax=ymax, xscale="linear",
                 yscale=yscale, font_size=12, xlabel="x", ylabel="y", title="", grid=False, legend=True, xerr=True,
                 yerr=True, band=True, combined=True)
        plotting.draw_publication(ax, series, o)
