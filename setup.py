# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Group iSP and contributors

from setuptools import setup, find_packages
from Cython.Build import cythonize
import glob

print("DISCOVERED PACKAGES:", find_packages(include=["himap", "himap.*"]))
# Setting up
setup(
    # name="himap",
    # packages=find_packages(include=["himap", "himap.*,himap.cython_build"]),
    ext_modules=cythonize(glob.glob("himap/cython_build/fwd_bwd.pyx"), compiler_directives={'boundscheck': False})
)
