import os
import sys
from setuptools import setup, Extension
import pybind11

functions_module = Extension(
    'black_hole_core',
    sources=['src/bindings.cpp'],
    include_dirs=[
        'include',
        pybind11.get_include(),
    ],
    language='c++',
    extra_compile_args=['-std=c++17', '-O3', '-march=native', '-ffast-math'],
)

setup(
    name='black_hole_core',
    version='0.1.0',
    description='Barnes-Hut Octree N-body simulation with relativistic/pseudo-Newtonian Black Hole',
    ext_modules=[functions_module],
    zip_safe=False,
)
