from pytest import approx
import xarray

import wmm2025 as wmm

# Reference point from NOAA's official WMM2025 test values
# (WMM2025COF.zip -> WMM2025_TestValues.txt): year 2025.0, alt 65 km,
# lat 43 N, lon 93 E -> X=24299.852822 Y=210.517066 Z=50037.923998
# F=55626.621348 D=0.50 I=64.10. Expected values below are the package's
# own output for those inputs, which agrees with NOAA to the precision
# NOAA publishes (mm-nT / 0.01 deg).


def test_wmm2025():
    mag = wmm.wmm(43, 93, alt_km=65, yeardec=2025.0)

    assert isinstance(mag, xarray.Dataset)

    assert mag.north.item() == approx(24299.85307628114)
    assert mag.east.item() == approx(210.51706617299905)
    assert mag.down.item() == approx(50037.923996804515)
    assert mag.total.item() == approx(55626.621458380745)

    assert mag.incl.item() == approx(64.09661390135923)
    assert mag.decl.item() == approx(0.4963584605572848)


def test_wmm2025_point():
    mag = wmm.wmm_point(43, 93, alt_km=65, yeardec=2025.0)
    assert isinstance(mag, dict)

    assert mag["north"] == approx(24299.85307628114)
    assert mag["east"] == approx(210.51706617299905)
    assert mag["down"] == approx(50037.923996804515)
    assert mag["total"] == approx(55626.621458380745)

    assert mag["incl"] == approx(64.09661390135923)
    assert mag["decl"] == approx(0.4963584605572848)
