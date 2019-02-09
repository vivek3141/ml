[![Build Test](https://travis-ci.com/vivek3141/ml.svg?branch=master)](https://travis-ci.com/vivek3141/ml)
[![Downloads](https://pepy.tech/badge/ml-python)](https://pepy.tech/project/ml-python)
[![PyPi Version](https://img.shields.io/pypi/v/ml-python.svg)](https://pypi.python.org/pypi/ml-python)
[![Python Compatibility](https://img.shields.io/pypi/pyversions/ml-python.svg)](https://pypi.python.org/pypi/fastai)
[![License](https://img.shields.io/pypi/l/ml-python.svg)](https://pypi.python.org/pypi/ml-python)
# ML

This module provides for the easiest way to implement Machine Learning algorithms without the need to know about them.

Learn the module here:
* [YouTube](https://www.youtube.com/watch?v=ReMIzozsx8Y)
* [Blog Post](https://vivek3141.github.io/blog/posts/ml.html)

Use this module if
- You are a complete beginner to Machine Learning.
- You find other modules too complicated.

This module is not meant for high level tasks, but only for simple use and learning.

I would not recommend using this module for big projects.

This module uses a tensorflow backend.

### Pip installation
```bash
pip install ml-python
```
### Python installation
```bash
git clone https://github.com/vivek3141/ml
cd ml
python setup.py install
```
### Bash Installation
```bash
git clone https://github.com/vivek3141/ml
cd ml
sudo make install
```
This module has support for ANNs, CNNs, linear regression, logistic regression, k-means.

## Examples
Examples for all implemented structures can be found in `/examples`. <br>
In this example, linear regression is used.
<br><br>
First, import the required modules.
```python
import numpy as np
from ml.linear_regression import LinearRegression
```
Then make the required object
```python
l = LinearRegression()
```
This code below randomly generates 50 data points from 0 to 10 for us to run linear regression on.
```python
# Randomly generating the data and converting the list to int
x = np.array(list(map(int, 10*np.random.random(50))))
y = np.array(list(map(int, 10*np.random.random(50))))
```
Lastly, train it. Set `graph=True` to visualize the dataset and the model.

```python
l.fit(data=x, labels=y, graph=True)
```
![Linear Regression](https://raw.githubusercontent.com/vivek3141/ml/master/images/linear_regression.png)<br><br>
The full code can be found in `/examples/linear_regression.py`
## Makefile
A Makefile is included for easy installation.<br>
To install using make run
```bash
sudo make
```
Note: Superuser privileges are only required if python is installed at `/usr/local/lib`
## License
All code is available under the [MIT License](https://github.com/vivek3141/ml/blob/master/LICENSE.md)

## Contributing
Pull requests are always welcome, so feel free to create one. Please follow the pull request template, so
your intention and additions are clear.
## Contact
Feel free to contact me by:
* Email: vivnps.verma@gmail.com
* GitHub Issue: [create issue](https://github.com/vivek3141/ml/issues/new)
