[![Build Test](https://travis-ci.com/vivek3141/ml.svg?branch=master)](https://travis-ci.com/vivek3141/ml)
[![Downloads](https://pepy.tech/badge/ml-python)](https://pepy.tech/project/ml-python)
[![PyPi Version](https://img.shields.io/pypi/v/ml-python.svg)](https://pypi.python.org/pypi/ml-python)
[![License](https://img.shields.io/pypi/l/ml-python.svg)](https://pypi.python.org/pypi/ml-python)
# ML

This module provides for the easiest way to implement Machine Learning algorithms. It also has in-built support for graphing and optimizers based in C.

Learn the module here:
* [YouTube](https://www.youtube.com/watch?v=ReMIzozsx8Y)
* [Blog Post](https://vivek3141.github.io/blog/posts/ml.html)
* [Examples](https://github.com/vivek3141/ml/tree/master/examples)

This module uses a tensorflow backend.

## Implemented Algorithms
* 2D CNN `ml.cnn`
* Basic MLP `ml.nn`
* K-Means `ml.k_means`
* Linear Regression `ml.linear_regression`
    * optimized with C
* Logistic Regression `ml.logistic_regression`
* Graph Modules `ml.graph`
    * Graph any function with or without data points - `from ml.graph import graph_function, graph_function_and_data`
* Nonlinear Regression `ml.regression`
* Optimizers - `ml.optimizer` optimized with C
    * GradientDescentOptimizer - `from ml.optimizer import GradientDescentOptimizer`
    * NewtonMethodOptimizer - `from ml.optimizer import NewtonMethodOptimizer`
    * AdamOptimizer - `from ml.optimizer import AdamOptimizer`	
* <b>UNSTABLE</b> - Character generating RNN - `ml.rnn`

#### You can find examples for all of these in `/examples`

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
