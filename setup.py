from setuptools import setup

setup(
    name='ml-python',
    version='0.2',
    packages=['ml',],
    author="Vivek Verma",
    author_email="vivekverma3141@gmail.com",
    url="https://github.com/vivek3141/ml",
    license='MIT',
    description="The easiest way to do machine learning",
    long_description=open("DESC.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)