#=========================
# setup_picalc_pyx_omp.py
#=========================
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "picalc_pyx_omp",
        ["picalc_pyx_omp.pyx"],
        extra_compile_args=['/openmp'],
        extra_link_args=['/openmp'],
    )
]

setup(name="picalc_pyx_omp",
      ext_modules=cythonize(ext_modules))
