from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='Hello world app',
    ext_modules=cythonize("learn.pyx"),
)

setup(
    name='Hello world app',
    ext_modules=cythonize("goutil.pyx"),
)

setup(
    name='Hello world app',
    ext_modules=cythonize("goinput.pyx"),
)

setup(
    name='Hello world app',
    ext_modules=cythonize("reinforce.pyx"),
)
