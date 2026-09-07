from __future__ import annotations

import atexit
import ctypes as ct
import os
import threading
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

# wmm_eval_grid() exists in the C library for scattered points but has no Python
# binding; anything added here must route its count through _c_int_size().

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


# The C layer holds the model coefficients and the Legendre / spherical-harmonic
# scratch buffers in one process-global struct, and ctypes drops the GIL for the
# duration of each call. Serialize every entry point so that concurrent calls
# from Python threads cannot interleave on that shared state. Callers who want
# real parallelism should use separate processes, which get their own copy.
_lock = threading.Lock()

if hasattr(os, "register_at_fork"):
    # A child forked while another thread held the lock would inherit it locked.
    os.register_at_fork(after_in_child=lambda: globals().__setitem__("_lock", threading.Lock()))


def _require_ok(ret: int, what: str) -> None:
    if ret != 0:
        raise RuntimeError(f"{what} failed (code {ret})")


def _init() -> None:
    if not COF_PATH.is_file():
        raise FileNotFoundError(f"WMM coefficient file not found: {COF_PATH}")
    with _lock:
        ret = libwmm.wmm_init(str(COF_PATH).encode("utf-8"))
    _require_ok(ret, "wmm_init")


def _shutdown() -> None:
    with _lock:
        libwmm.wmm_free()


_init()
atexit.register(_shutdown)


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


_C_INT_MAX = 2**31 - 1


def _c_int_size(value: int, name: str) -> int:
    """
    Validate a count that crosses into the C API as ``int``.

    ctypes truncates silently rather than raising, so an oversized count would
    reach C as a smaller (or negative) number and quietly produce partial
    output. The C side indexes the outer product with ``size_t``, so only the
    individual axis lengths are bounded, not their product.
    """
    value = int(value)
    if not 0 <= value <= _C_INT_MAX:
        raise ValueError(f"{name} must be between 0 and {_C_INT_MAX}, got {value}")
    return value


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
    nlat = _c_int_size(lat_axis.size, "number of latitudes")
    nlon = _c_int_size(lon_axis.size, "number of longitudes")
    north, east, down, total, decl, incl = _out_ptrs(nlat * nlon)

    with _lock:
        ret = libwmm.wmm_eval_latlon_grid(
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
        )
    _require_ok(ret, "wmm_eval_latlon_grid")

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

    inputs = {
        "glats": np.asarray(glats),
        "glons": np.asarray(glons),
        "alt_km": np.asarray(alt_km),
        "yeardec": np.asarray(yeardec),
    }
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

    # Broadcast the held-constant inputs to the common shape.
    flat = {
        k: _as_f64_1d(v if v.size > 1 else np.full(sz, v.flat[0] if v.size else v))
        for k, v in inputs.items()
    }
    n = _c_int_size(flat["glats"].size, "number of points")
    north, east, down, total, decl, incl = _out_ptrs(n)

    with _lock:
        ret = libwmm.wmm_eval_many(
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
        )
    _require_ok(ret, "wmm_eval_many")

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

    with _lock:
        ret = libwmm.wmm_eval(
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
        )
    _require_ok(ret, "wmm_eval")

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
