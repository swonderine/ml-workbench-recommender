# Train & Serve a Machine Learning Model as a Microservice using Docker and Track the Model-Performance using MLflow

### Let's build a recommendation engine with a collaborative filtering model, track the model results with MLflow and use Flask to serve batch-predictions of products other users may like as well

### What we will do

We will create of a local machine learning workbench aka containerized docker setup for production software-like development workflows. Having that in mind, we will use the following tools to do so:

+ [MLFlow](https://www.mlflow.org/) for experiment tracking and model management
+ [MinIO](https://min.io/) to mimic AWS S3 and act as an artifact and data storage system
+ PostgreSQL for a SQL engine and to serve as a backend store for MLFlow

We will end up with an easily and quickly configurable docker setup that gets all these tools up and running. Our goal is not to make this fully runnable in the cloud (yet). We want a workig setup that we can start with one command to be able to experiment working with all kinds of different data and models.

In the second part, we will train and track a collaborative filtering model and use the following tools for training & serving predictions of similar products other users may like as well with a RESTful API:

+ [implicit](https://github.com/benfred/implicit) to train our model
+ [Flask REST API](https://flask.palletsprojects.com/en/2.0.x/) to serve our predictions
