from setuptools import setup
from setuptools import find_packages
from setuptools import Extension

gradient_descent = Extension('gradient_descent', sources=[
    'ml/optimizer/gradient_descent.cpp'])
linear_regression = Extension('lin_reg', sources=[
    'ml/linear_regression/linear_regression.cpp'])
adam = Extension('adam', sources=[
    'ml/optimizer/adam.cpp'])


setup(
        name='ml-python',
        version='2.3.1',
        author="Vivek Verma",
        packages=find_packages(),
        author_email="vivekverma3141@gmail.com",
        url="https://github.com/vivek3141/ml",
        license='MIT',
        description="The easiest way to do machine learning",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        ext_modules=[gradient_descent, linear_regression, adam],
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 2",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            ]
        )
