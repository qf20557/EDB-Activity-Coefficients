"""Large-file paths: the block reader, the thinned display, culling, the block-wise fit and the cleaned-data files
give the same answers as the simple whole-array versions they replaced."""
import csv
import math

import numpy as np
import pytest

from edb import config, raw_data
from edb.config import RADIUS_UNCERTAINTY_UM
from edb.culling import PointStore
from edb.fit_io import read_edb_csv
from edb.fitting import fit_model, fit_trace
from edb.raw_data import read_files, read_pair, write_cleaned
from edb.trace import Cancelled, Trace


def reference_read(paths):
    """The original line-by-line reader, kept here as the definition of what read_files must return."""
    rows, skipped = [], 0
    last_raw = last_time = None
    last_step = 0.0
    for path in paths:
        offset, first = 0.0, True
        with open(path, encoding="utf-8-sig", errors="replace") as handle:
            for line in handle:
                pair = read_pair(line)
                if pair is None:
                    skipped += 1
                    continue
                raw, radius = pair
                if first:
                    if last_raw is not None and raw <= last_raw:
                        offset = last_time + last_step - raw
                    first = False
                time = raw + offset
                if last_raw is not None and raw > last_raw:
                    last_step = raw - last_raw
                last_raw, last_time = raw, time
                rows.append((time, radius))
    data = np.asarray(rows, dtype=float)
    data[:, 0] -= data[0, 0]
    return data[:, 0], data[:, 1], skipped


@pytest.fixture
def small_blocks(monkeypatch):
    """Blocks of a few kB / a few hundred rows, so small test data crosses many block boundaries."""
    monkeypatch.setattr(config, "READ_BLOCK_BYTES", 3000)
    monkeypatch.setattr(config, "CHUNK_ROWS", 257)
    monkeypatch.setattr(config, "WRITE_ROWS", 100)
    monkeypatch.setattr(raw_data, "SLOW_BYTES", 200)


def messy_file(path, rng, start, rows, newline="\n", sep="\t"):
    """Numeric rows with the awkward lines real files contain scattered through them."""
    junk = ["", "   ", "time radius", "# comment", "nan 5", "3 inf", "7 -1", "8 0", "9", "1_0 2.5", "a,b,c",
            "5.5,4.25,extra,columns", "  6.5   4.125  ", "1e3 2e0 9", '"3" 4', "NA 3", "0x10 2", "12:30 4"]
    t = start + np.cumsum(rng.uniform(0.01, 0.02, rows))
    r = 20 + rng.normal(0, 0.05, rows)
    lines = ["Time (s)" + sep + "Radius (um)"]
    for i in range(rows):
        lines.append(f"{t[i]!r}{sep}{r[i]!r}" + (f"{sep}{i}" if i % 7 == 0 else ""))
        if rng.random() < 0.01:
            lines.append(junk[int(rng.integers(len(junk)))])
    path.write_bytes(newline.join(lines).encode("utf-8") + (newline.encode() if rows % 2 else b""))


@pytest.mark.parametrize("newline,sep", [("\n", "\t"), ("\r\n", ","), ("\r\n", " "), ("\r", "\t"), ("\n", ", ")])
def test_read_files_matches_line_by_line_reader(tmp_path, small_blocks, newline, sep):
    rng = np.random.default_rng(1)
    paths = [tmp_path / f"{i}.txt" for i in range(3)]
    for path, start, rows in zip(paths, (100.0, 0.0, 500.0), (4001, 1500, 2000)):  # second file's clock restarts
        messy_file(path, rng, start, rows, newline, sep)
    paths[0].write_bytes(b"\xef\xbb\xbf" + paths[0].read_bytes())
    t, r, skipped = read_files(paths)
    t_ref, r_ref, skipped_ref = reference_read(paths)
    np.testing.assert_array_equal(t, t_ref)  # bit for bit
    np.testing.assert_array_equal(r, r_ref)
    assert skipped == skipped_ref and t.flags.c_contiguous and r.flags.c_contiguous


def test_read_files_progress_and_cancel(tmp_path, small_blocks):
    path = tmp_path / "a.txt"
    messy_file(path, np.random.default_rng(2), 0.0, 3000)
    seen = []
    read_files([path], progress=seen.append)
    assert seen == sorted(seen) and seen[-1] == 1.0 and len(seen) > 5
    with pytest.raises(Cancelled):
        read_files([path], cancel=lambda: len(seen) > 0)
    with pytest.raises(ValueError, match="No valid rows"):
        empty = tmp_path / "empty.txt"
        empty.write_text("just a header\n")
        read_files([empty])


def noisy_store(n=20000, seed=3):
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.uniform(0.5, 1.5, n))
    return PointStore(t, 5 + 20 * np.exp(-t / 4000) + rng.normal(0, 0.05, n))


def test_display_thins_but_keeps_spikes(small_blocks):
    store = noisy_store()
    store.r[[123, 15000]] = 60.0, 0.5  # one spike each way
    t, r = store.display(budget=400)
    assert len(t) <= 400 and np.all(np.diff(t) > 0)
    assert r.max() == 60.0 and r.min() == 0.5
    t, r = store.display(lo=store.t[100], hi=store.t[199], budget=400)  # zoomed in: every point
    np.testing.assert_array_equal(r, store.r[100:200])
    store.cull_box((0, store.t[5000], 0, 100), squared=False)  # culled points are never drawn
    t, r = store.display(budget=400)
    assert t.min() > store.t[5000] and r.max() < 60.0
    t, r = store.display(stride=50)
    np.testing.assert_array_equal(t, store.t[::50][store.keep[::50]])


def test_cull_matches_whole_array_mask_and_undo(small_blocks):
    store = noisy_store()
    t, r = store.t, store.r
    for box, squared, keep_inside, scale in (((2000, 9000, 9.5, 12), False, False, 1.02),
                                             ((0, 15000, 30, 600), True, True, 1.0),
                                             ((4000, 4200, 0, 1e9), True, False, 1.0)):
        y = (r * scale) ** 2 if squared else r * scale
        inside = (t >= box[0]) & (t <= box[1]) & (y >= box[2]) & (y <= box[3])
        expected = store.keep & (inside if keep_inside else ~inside)
        before = store.keep.copy()
        assert store.cull_box(box, squared, keep_inside, scale) == before.sum() - expected.sum()
        np.testing.assert_array_equal(store.keep, expected)
        assert store.count == expected.sum()
    kinds = {entry.dtype for entry in store.history}
    assert kinds == {np.dtype(np.uint8), np.dtype(np.intp)}  # large culls store a packed mask, small ones row numbers
    snapshot = store.snapshot()
    assert store.undo() and store.undo() and store.undo() and not store.undo()
    assert store.keep.all() and store.count == len(store)
    assert snapshot.count < len(store)  # unaffected by the undo


def test_block_fit_matches_whole_array_fit(monkeypatch):
    store = noisy_store(50000)
    store.cull_box((10000, 12000, 0, 100), squared=False)
    t, r = store.active()
    whole = fit_model(t, r * 1.02, 500.0, 40000.0)
    monkeypatch.setattr(config, "CHUNK_ROWS", 999)
    blocks = fit_trace(store, 500.0, 40000.0, scale=1.02)
    assert (blocks["n"], blocks["lo"], blocks["hi"]) == (whole["n"], whole["lo"], whole["hi"])
    assert blocks["rms"] == pytest.approx(whole["rms"], rel=1e-9) and blocks["r2"] == pytest.approx(whole["r2"])
    x = np.linspace(500, 40000, 50)
    np.testing.assert_allclose(blocks["func"](x), whole["func"](x), rtol=1e-9)
    few = fit_trace(Trace(t[:200], r[:200]))  # below the bin count: fitted point by point
    assert few["n"] == 200
    with pytest.raises(ValueError, match="at least 10"):
        fit_trace(Trace(np.repeat(1.0, 50) + np.arange(50) % 2, np.ones(50)))


def test_cleaned_csv_is_unchanged_and_reads_back(tmp_path, small_blocks):
    store = noisy_store(3000)
    store.cull_box((500, 900, 0, 100), squared=False)
    t, r = store.active()
    factor = 1.0213
    sigma = RADIUS_UNCERTAINTY_UM * factor
    rc = r * factor
    for name, sig, header, columns in (
            ("plain.csv", None, ["time_s", "radius_raw_um", "radius_corrected_um"], (t, r, rc)),
            ("errors.csv", sigma, ["time_s", "radius_raw_um", "radius_corrected_um", "radius_corrected_err_um",
                                   "radius_corrected_squared_um2", "radius_corrected_squared_err_um2"],
             (t, r, rc, np.full_like(rc, sigma), rc * rc, 2 * rc * sigma))):
        with open(tmp_path / "expected.csv", "w", newline="", encoding="utf-8") as handle:  # the original writer
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(zip(*columns))
        seen = []
        assert write_cleaned(tmp_path / name, store, factor, sig, progress=seen.append) == len(t)
        assert (tmp_path / name).read_bytes() == (tmp_path / "expected.csv").read_bytes()
        assert seen[-1] == 1.0 and not (tmp_path / f"{name}.part").exists()
        t2, r2, sigma2 = read_edb_csv(tmp_path / name)
        np.testing.assert_array_equal(t2, t)
        np.testing.assert_array_equal(r2, rc)
        assert sigma2 == pytest.approx(sigma)
    with pytest.raises(Cancelled):
        write_cleaned(tmp_path / "cancelled.csv", store, factor, cancel=lambda: True)
    assert not list(tmp_path.glob("cancelled*"))
    with pytest.raises(ValueError, match="radius_corrected_um"):
        (tmp_path / "raw.txt").write_text("1 2\n3 4\n")
        read_edb_csv(tmp_path / "raw.txt")


def test_cleaned_binary_round_trip(tmp_path):
    store = noisy_store(3000)
    store.cull_box((500, 900, 0, 100), squared=False)
    t, r = store.active()
    for sig in (None, 0.25):
        write_cleaned(tmp_path / "clean.npz", store, 1.0213, sig)
        t2, r2, sigma = read_edb_csv(tmp_path / "clean.npz")  # stage 2: corrected radius
        np.testing.assert_array_equal(t2, t)
        np.testing.assert_array_equal(r2, r * 1.0213)
        assert sigma == pytest.approx(RADIUS_UNCERTAINTY_UM * 1.0213 if sig is None else sig)
        t1, r1, skipped = read_files([tmp_path / "clean.npz"])  # stage 1: raw radius, time from zero
        np.testing.assert_array_equal(t1, t - t[0])
        np.testing.assert_array_equal(r1, r)
        assert skipped == 0 and math.isfinite(t1[-1])
