from setuptools import Extension
from setuptools import setup

linear_regression = Extension('lin_reg', sources=['linear_regression.cpp'])


setup(
    name="lin_reg",
    version=1.0,
    ext_modules=[linear_regression]
)
