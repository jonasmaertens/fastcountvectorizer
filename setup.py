from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extension_fastcv = Extension(
    "fastcountvectorizer._count_vocab_cy",
    ["fastcountvectorizer/_count_vocab_cy.pyx"],
    include_dirs=[numpy.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

setup(
    ext_modules=cythonize([extension_fastcv])
)
