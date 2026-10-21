# WMM2025

WMM2025 World Magnetic Model...in simple Python.
The NOAA C reference implementation is wrapped with `ctypes` and exposed through a small NumPy-based API.
Tested on Linux, Mac and Windows.

At this time Visual Studio (MSVC) is not supported since MSVC doesn't export function symbols without additional headers,
which is typically done with something like SWIG.
On Windows use MinGW (the build selects the "MinGW Makefiles" generator unless `CMAKE_GENERATOR` is set).

## Install

Requirements: Python, NumPy, [CMake](https://cmake.org/download/) (>= 3.14) and a C compiler.

```sh
git clone https://github.com/Navigraph/wmm2025

python -m pip install -e wmm2025
```

This Python wrapper of WMM2025 uses a build-on-run technique.
The first time you import `wmm2025`, you will see messages from CMake and your C compiler.
The shared library is built into `src/wmm2025/build/` and is rebuilt automatically
if `CMakeLists.txt` or any C source/header is newer than the built library.

## Usage

All functions take geodetic latitude and longitude in degrees, altitude in km above the ellipsoid and time as a decimal year.
Field components (`north`, `east`, `down`, `total`) are in nT, `decl` and `incl` in degrees.

### Single point

```python
import wmm2025

mag = wmm2025.wmm_point(glat, glon, alt_km, yeardec)
# dict: glat, glon, north, east, down, total, incl, decl, time
```

### Latitude/longitude grid

`glats` and `glons` are 2-D arrays in `numpy.meshgrid` layout, evaluated at a single altitude:

```python
import numpy as np
import wmm2025

lon, lat = np.meshgrid(np.arange(-180, 181, 10), np.arange(-90, 91, 10))
mag = wmm2025.wmm(lat, lon, alt_km, yeardec)
# dict: glat, glon (1-D axes), north, east, down, total, incl, decl (2-D), time
```

### Transect

Evaluate many points at once. Each argument is either a scalar (held constant) or an array; all arrays must have the same shape:

```python
mag = wmm2025.transect(glats, glons, alt_km, yeardec)
# dict: north, east, down, total, decl, incl (arrays, or floats if all inputs are scalars)
```

The C layer keeps global state, so calls are serialized with a lock and are thread-safe but not parallel.
For parallelism use separate processes.

### Example script: declination / inclination maps

`RunWMM2025.py` plots declination and inclination contours (requires `matplotlib`, available via the `plot` extra):

```sh
python -m pip install -e wmm2025[plot]
python RunWMM2025.py 2025.62 0
```

where the arguments are the decimal year and the altitude in km.

## Development

```sh
python -m pip install -e wmm2025[tests,lint]

flake8
mypy
pytest
```

## Reference

* WMM2025 [inclination map](https://www.ngdc.noaa.gov/geomag/WMM/data/WMM2025/WMM2025_I_MERC.pdf)
* WMM2025 [declination map](https://www.ngdc.noaa.gov/geomag/WMM/data/WMM2025/WMM2025_D_MERC.pdf)
* WMM2025 [test values](https://www.ngdc.noaa.gov/geomag/WMM/data/WMM2025/WMM2025testvalues.pdf)
