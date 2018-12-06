#!/usr/bin/env bash
cd tests > /dev/null 2>&1
python3 k_means.py > /dev/null 2>&1
python3 linear_regression.py > /dev/null 2>&1
python3 neuralnetwork.py > /dev/null 2>&1
python3 cnn.py > /dev/null 2>&1
python3 logistic_regression.py > /dev/null 2>&1
python3 regression.py > /dev/null 2>&1
echo "Passed"
