import threading

import pytest
from pytest import approx
import numpy as np

import wmm2025 as wmm
import wmm2025.base as base

# Reference point from NOAA's official WMM2025 test values
# (WMM2025COF.zip -> WMM2025_TestValues.txt): year 2025.0, alt 65 km,
# lat 43 N, lon 93 E -> X=24299.852822 Y=210.517066 Z=50037.923998
# F=55626.621348 D=0.50 I=64.10. Expected values below are the package's
# own output for those inputs, which agrees with NOAA to the precision
# NOAA publishes (mm-nT / 0.01 deg).


def test_wmm2025():
    mag = wmm.wmm(43, 93, alt_km=65, yeardec=2025.0)

    assert isinstance(mag, dict)
    assert isinstance(mag["north"], np.ndarray)

    assert mag["north"].item() == approx(24299.85307628114)
    assert mag["east"].item() == approx(210.51706617299905)
    assert mag["down"].item() == approx(50037.923996804515)
    assert mag["total"].item() == approx(55626.621458380745)

    assert mag["incl"].item() == approx(64.09661390135923)
    assert mag["decl"].item() == approx(0.4963584605572848)
    assert mag["time"] == 2025.0


def test_wmm2025_point():
    mag = wmm.wmm_point(43, 93, alt_km=65, yeardec=2025.0)
    assert isinstance(mag, dict)

    assert mag["north"] == approx(24299.85307628114)
    assert mag["east"] == approx(210.51706617299905)
    assert mag["down"] == approx(50037.923996804515)
    assert mag["total"] == approx(55626.621458380745)

    assert mag["incl"] == approx(64.09661390135923)
    assert mag["decl"] == approx(0.4963584605572848)


def test_oversized_counts_are_rejected():
    """ctypes truncates silently, so counts crossing into C must be checked."""
    assert base._c_int_size(0, "n") == 0
    assert base._c_int_size(2**31 - 1, "n") == 2**31 - 1

    for bad in (2**31, 2**33 + 3, -1):
        with pytest.raises(ValueError):
            base._c_int_size(bad, "n")


def test_concurrent_calls_do_not_interfere():
    """The C layer shares scratch state, so calls must be serialized."""
    points = [(lat, lon) for lat in range(-80, 81, 40) for lon in range(-180, 181, 90)]
    expected = {p: wmm.wmm_point(p[0], p[1], 5.0, 2026.5)["decl"] for p in points}
    mismatches = []

    def worker(epoch: float) -> None:
        for _ in range(100):
            for lat, lon in points:
                # A differing epoch rewrites the cached time-adjusted coefficients.
                wmm.wmm_point(lat, lon, 5.0, epoch)
                got = wmm.wmm_point(lat, lon, 5.0, 2026.5)["decl"]
                if got != expected[(lat, lon)]:
                    mismatches.append((lat, lon, got))

    threads = [threading.Thread(target=worker, args=(2020.0 + i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not mismatches


def test_wmm_grid_matches_point():
    lon, lat = np.meshgrid(np.arange(-180, 181, 60), np.arange(-90, 91, 45))
    mag = wmm.wmm(lat, lon, 1.5, 2025.5)

    assert mag["north"].shape == lat.shape
    assert mag["glat"].shape == (lat.shape[0],)
    assert mag["glon"].shape == (lon.shape[1],)
    for i in range(lat.shape[0]):
        for j in range(lat.shape[1]):
            pt = wmm.wmm_point(lat[i, j], lon[i, j], 1.5, 2025.5)
            for key in ("north", "east", "down", "total", "incl", "decl"):
                assert mag[key][i, j] == approx(pt[key])


def test_wmm_grid_rejects_bad_layout():
    lon, lat = np.meshgrid(np.arange(0, 30, 10), np.arange(0, 30, 10))
    with pytest.raises(ValueError):
        wmm.wmm(lat.T, lon, 0.0, 2025.0)
    with pytest.raises(ValueError):
        wmm.wmm(lat, lon[:2], 0.0, 2025.0)


def test_transect_scalar_matches_point():
    tr = wmm.transect(43, 93, 65, 2025.0)
    pt = wmm.wmm_point(43, 93, 65, 2025.0)

    assert isinstance(tr["north"], float)
    for key in ("north", "east", "down", "total", "incl", "decl"):
        assert tr[key] == approx(pt[key])


def test_transect_broadcasts_constants():
    lats = np.linspace(-60, 60, 7)
    tr = wmm.transect(lats, 10.0, 0.0, np.linspace(2025.0, 2029.0, 7))

    assert tr["decl"].shape == lats.shape
    for i, lat in enumerate(lats):
        pt = wmm.wmm_point(lat, 10.0, 0.0, 2025.0 + i * 4.0 / 6)
        assert tr["decl"][i] == approx(pt["decl"])
        assert tr["total"][i] == approx(pt["total"])


def test_transect_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        wmm.transect(np.zeros(3), np.zeros(4), 0.0, 2025.0)


def test_transect_empty():
    tr = wmm.transect(np.array([]), 0.0, 0.0, 2025.0)
    assert tr["north"].shape == (0,)
