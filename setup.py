from setuptools import setup

setup(
    name='ml-python',
    version='0.9',
    packages=['ml', 'ml.nn', 'ml.linear_regression', 'ml.activation', 'ml.k_means', 'ml.logistic_regression',
              'ml.error', 'ml.cnn', 'ml.random', 'ml.graph', 'ml.graph.lr'],
    author="Vivek Verma",
    author_email="vivekverma3141@gmail.com",
    url="https://github.com/vivek3141/ml",
    license='MIT',
    description="The easiest way to do machine learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
