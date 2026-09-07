"""
A generic, clean way to build C/C++/Fortran code "build on run"

Michael Hirsch, Ph.D.
https://www.scivision.dev
"""

import shutil
from pathlib import Path
import importlib.resources
import subprocess
import sys
import os


def package_src_dir() -> Path:
    with importlib.resources.path(__package__, "CMakeLists.txt") as f:
        return f.parent


def build():
    """
    attempt to build using CMake
    """

    exe = shutil.which("cmake")
    if not exe:
        raise FileNotFoundError("CMake not available")

    s = package_src_dir()
    b = s / "build"
    g = []
    if sys.platform == "win32" and not os.environ.get("CMAKE_GENERATOR"):
        g = ["-G", "MinGW Makefiles"]
    subprocess.check_call([exe, f"-S{s}", f"-B{b}"] + g)
    subprocess.check_call([exe, "--build", str(b), "--parallel"])


def compile_inputs() -> list[Path]:
    """
    Every file that can change the built library.

    Discovered rather than listed, so that adding a source or header cannot
    silently leave users on stale native code.
    """
    s = package_src_dir()
    return [s / "CMakeLists.txt", *sorted((s / "src").rglob("*.c")),
            *sorted((s / "src").rglob("*.h"))]


def needs_rebuild(dllfn: Path) -> bool:
    if not dllfn.is_file():
        return True
    dll_mtime = dllfn.stat().st_mtime
    return any(f.stat().st_mtime > dll_mtime for f in compile_inputs() if f.is_file())


def get_libpath(bin_dir: Path, stem: str) -> Path:
    if sys.platform in ("win32", "cygwin"):
        dllfn = bin_dir / ("lib" + stem + ".dll")
    elif sys.platform == "linux":
        dllfn = bin_dir / ("lib" + stem + ".so")
    elif sys.platform == "darwin":
        dllfn = bin_dir / ("lib" + stem + ".dylib")
    else:
        raise ValueError(f"Unsupported platform: {sys.platform}")

    return dllfn
