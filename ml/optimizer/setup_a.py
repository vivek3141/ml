from setuptools import Extension
from setuptools import setup

adam = Extension('gradient_descent', sources=['adam.c'])


setup(
    name="adam",
    version=1.0,
    ext_modules=[adam]
)
