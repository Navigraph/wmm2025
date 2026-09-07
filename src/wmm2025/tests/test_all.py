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
