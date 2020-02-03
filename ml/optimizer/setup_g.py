from setuptools import Extension
from setuptools import setup

gradient_descent = Extension('gradient_descent', sources=['gradient_descent.c'])


setup(
    name="gradient_descent",
    version=1.0,
    ext_modules=[gradient_descent]
)