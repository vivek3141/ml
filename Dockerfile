FROM ubuntu:18.04
RUN pip install -r requirements.txt
RUN python setup.py install