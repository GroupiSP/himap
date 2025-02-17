from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    name='fwd_bwd',
    setup_requires=['setuptools>=18.0', 'cython'],
    ext_modules=cythonize([
        Extension(
            'himap.cython_build.fwd_bwd',
            sources=['himap/cython_build/fwd_bwd.pyx'],
        ),
    ], compiler_directives={'boundscheck': False}),
)
