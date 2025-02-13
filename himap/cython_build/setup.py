from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("fwd_bwd.pyx",compiler_directives={'boundscheck': False})
)