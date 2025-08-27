# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Group iSP and contributors

from setuptools import setup
from Cython.Build import cythonize
import glob

setup(
    ext_modules=cythonize(glob.glob("himap/cython_build/fwd_bwd.pyx"),compiler_directives={'boundscheck': False})
)
