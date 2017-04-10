from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import os
os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("cutils", ["cutils.pyx"])]
)
