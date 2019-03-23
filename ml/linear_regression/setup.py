from setuptools import Extension
from setuptools import setup

gradient_descent = Extension('linear_regression', sources=['linear_regression.c'])


setup(
    name="linear_regression",
    version=1.0,
    ext_modules=[gradient_descent]
)