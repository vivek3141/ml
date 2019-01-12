from setuptools import setup
from setuptools import find_packages

setup(
    name='ml-python',
    version='2.0',
    author="Vivek Verma",
    packages=find_packages(),
    author_email="vivekverma3141@gmail.com",
    url="https://github.com/vivek3141/ml",
    license='MIT',
    description="The easiest way to do machine learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
