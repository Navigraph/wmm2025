from __future__ import annotations

import atexit
import ctypes as ct
from pathlib import Path

import numpy as np

from .build import build, get_libpath, needs_rebuild

SDIR = Path(__file__).parent
BDIR = SDIR / "build"
COF_PATH = SDIR / "WMM.COF"

dllfn = get_libpath(BDIR, "wmm20")
if needs_rebuild(dllfn):
    build()
    dllfn = get_libpath(BDIR, "wmm20")
    if not dllfn.is_file():
        raise ModuleNotFoundError(f"could not find {dllfn}")

libwmm = ct.CDLL(str(dllfn))

_c_double_p = ct.POINTER(ct.c_double)

libwmm.wmm_init.argtypes = [ct.c_char_p]
libwmm.wmm_init.restype = ct.c_int

libwmm.wmm_free.argtypes = []
libwmm.wmm_free.restype = None

libwmm.wmm_eval.argtypes = [
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    _c_double_p,
    _c_double_p,
    _c_double_p,
    _c_double_p,
    _c_double_p,
    _c_double_p,
]
libwmm.wmm_eval.restype = ct.c_int

libwmm.wmm_eval_grid.argtypes = [
    _c_double_p,
    _c_double_p,
    ct.c_int,
    ct.c_double,
    ct.c_double,
    _c_double_p,
    _c_double_p,
    _c_double_p,
    _c_double_p,
    _c_double_p,
    _c_double_p,
]
libwmm.wmm_eval_grid.restype = ct.c_int

libwmm.wmm_eval_latlon_grid.argtypes = [
    _c_double_p,
    ct.c_int,
    _c_double_p,
    ct.c_int,
    ct.c_double,
    ct.c_double,
    _c_double_p,
    _c_double_p,
    _c_double_p,
    _c_double_p,
    _c_double_p,
    _c_double_p,
]
libwmm.wmm_eval_latlon_grid.restype = ct.c_int

libwmm.wmm_eval_many.argtypes = [
    _c_double_p,
    _c_double_p,
    _c_double_p,
    _c_double_p,
    ct.c_int,
    _c_double_p,
    _c_double_p,
    _c_double_p,
    _c_double_p,
    _c_double_p,
    _c_double_p,
]
libwmm.wmm_eval_many.restype = ct.c_int


def _require_ok(ret: int, what: str) -> None:
    if ret != 0:
        raise RuntimeError(f"{what} failed (code {ret})")


def _init() -> None:
    if not COF_PATH.is_file():
        raise FileNotFoundError(f"WMM coefficient file not found: {COF_PATH}")
    _require_ok(libwmm.wmm_init(str(COF_PATH).encode("utf-8")), "wmm_init")


_init()
atexit.register(libwmm.wmm_free)


def _as_f64_1d(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64).reshape(-1)


def _out_ptrs(n: int):
    north = np.empty(n, dtype=np.float64)
    east = np.empty(n, dtype=np.float64)
    down = np.empty(n, dtype=np.float64)
    total = np.empty(n, dtype=np.float64)
    decl = np.empty(n, dtype=np.float64)
    incl = np.empty(n, dtype=np.float64)
    return north, east, down, total, decl, incl


def _cptr(a: np.ndarray):
    return a.ctypes.data_as(_c_double_p)


def wmm(glats: np.ndarray, glons: np.ndarray, alt_km: float, yeardec: float) -> dict:
    """
    wmm computes the value of the world magnetic model at grid points specified by glats and
    glons, for a single altitude value. glats and glons should be in degrees.

    glats and glons should be generated from something like np.meshgrid, so they should be
    2-D arrays.

    Returns a dict of numpy arrays: glat/glon (1-D), field components (2-D), and time.
    """

    glats = np.atleast_2d(np.ascontiguousarray(glats, dtype=np.float64))
    glons = np.atleast_2d(np.ascontiguousarray(glons, dtype=np.float64))

    # expect lon[:,i] constant and lat[i,:] constant (meshgrid layout)
    if glats.shape[0] > 1 and not np.allclose(np.diff(glons, axis=0), 0):
        raise ValueError("glons must be constant along axis 0 (meshgrid layout)")
    if glats.shape[1] > 1 and not np.allclose(np.diff(glats, axis=1), 0):
        raise ValueError("glats must be constant along axis 1 (meshgrid layout)")
    if glats.shape != glons.shape:
        raise ValueError(f"glats shape {glats.shape} != glons shape {glons.shape}")

    # Only the row/column vectors matter: the C side evaluates the outer product
    # and reuses the Legendre functions across each latitude row.
    lat_axis = np.ascontiguousarray(glats[:, 0], dtype=np.float64)
    lon_axis = np.ascontiguousarray(glons[0, :], dtype=np.float64)
    nlat = lat_axis.size
    nlon = lon_axis.size
    north, east, down, total, decl, incl = _out_ptrs(nlat * nlon)

    _require_ok(
        libwmm.wmm_eval_latlon_grid(
            _cptr(lat_axis),
            nlat,
            _cptr(lon_axis),
            nlon,
            float(alt_km),
            float(yeardec),
            _cptr(north),
            _cptr(east),
            _cptr(down),
            _cptr(total),
            _cptr(decl),
            _cptr(incl),
        ),
        "wmm_eval_latlon_grid",
    )

    shape = glats.shape
    return {
        "glat": lat_axis,
        "glon": lon_axis,
        "north": north.reshape(shape),
        "east": east.reshape(shape),
        "down": down.reshape(shape),
        "total": total.reshape(shape),
        "incl": incl.reshape(shape),
        "decl": decl.reshape(shape),
        "time": float(yeardec),
    }


def transect(glats: np.ndarray, glons: np.ndarray, alt_km: np.ndarray, yeardec: np.ndarray) -> dict:
    """
    compute a transect through the WMM

    All inputs should be either single values, indicating held constant or numpy.ndarray
    of the same size.
    """

    inputs = {k: np.asarray(v) for k, v in vars().items()}
    szs = {k: v.shape for k, v in inputs.items() if v.size > 1}

    if len(szs) > 1:
        shapes = list(szs.values())
        if any(s != shapes[0] for s in shapes[1:]):
            raise ValueError(f"incompatible input shapes: {szs}")
        sz = shapes[0]
    elif len(szs) == 1:
        sz = list(szs.values())[0]
    else:
        sz = ()

    ref_input = {k: v if v.size > 1 else np.full(sz, v.item() if v.shape == () else v.flat[0]) for k, v in inputs.items()}

    flat = {k: _as_f64_1d(v) for k, v in ref_input.items()}
    n = flat["glats"].size
    north, east, down, total, decl, incl = _out_ptrs(n)

    _require_ok(
        libwmm.wmm_eval_many(
            _cptr(flat["glats"]),
            _cptr(flat["glons"]),
            _cptr(flat["alt_km"]),
            _cptr(flat["yeardec"]),
            n,
            _cptr(north),
            _cptr(east),
            _cptr(down),
            _cptr(total),
            _cptr(decl),
            _cptr(incl),
        ),
        "wmm_eval_many",
    )

    if sz == ():
        return {
            "north": north.item(),
            "east": east.item(),
            "down": down.item(),
            "total": total.item(),
            "decl": decl.item(),
            "incl": incl.item(),
        }

    return {
        "north": north.reshape(sz),
        "east": east.reshape(sz),
        "down": down.reshape(sz),
        "total": total.reshape(sz),
        "decl": decl.reshape(sz),
        "incl": incl.reshape(sz),
    }


def wmm_point(glat: float, glon: float, alt_km: float, yeardec: float) -> dict[str, float]:
    """
    wmm_point computes the value of the world magnetic model at a specific point.
    """

    glat = float(glat)
    glon = float(glon)
    alt_km = float(alt_km)
    yeardec = float(yeardec)

    x = ct.c_double()
    y = ct.c_double()
    z = ct.c_double()
    T = ct.c_double()
    D = ct.c_double()
    mI = ct.c_double()

    _require_ok(
        libwmm.wmm_eval(
            glat,
            glon,
            alt_km,
            yeardec,
            ct.byref(x),
            ct.byref(y),
            ct.byref(z),
            ct.byref(T),
            ct.byref(D),
            ct.byref(mI),
        ),
        "wmm_eval",
    )

    return {
        "glat": glat,
        "glon": glon,
        "north": x.value,
        "east": y.value,
        "down": z.value,
        "total": T.value,
        "incl": mI.value,
        "decl": D.value,
        "time": yeardec,
    }
