FROM python:3.8-slim

WORKDIR /src

# both files are explicitly required!
COPY ./1_Train_Models/Pipfile  ./
COPY ./1_Train_Models/Pipfile.lock ./

RUN pip install pipenv
RUN apt-get update && \
    apt-get -y install gcc g++
RUN pipenv install


RUN mkdir 0_Data
RUN mkdir 1_Train_Models

COPY ./1_Train_Models/train.py /src/1_Train_Models/train.py
COPY ./1_Train_Models/preprocessing.py /src/1_Train_Models/preprocessing.py
COPY ./1_Train_Models/modeltraining.py /src/1_Train_Models/modeltraining.py
COPY ./1_Train_Models/predictions.py /src/1_Train_Models/predictions.py
COPY ./1_Train_Models/implicitwrapper.py /src/1_Train_Models/implicitwrapper.py
