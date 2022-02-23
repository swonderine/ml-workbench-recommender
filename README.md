# Train & Serve a Machine Learning Model as a Microservice using Docker and Track the Model-Performance using MLflow

### Let's build a recommendation engine with a collaborative filtering model, track the model results with MLflow and use Flask to serve batch-predictions of products other users may like as well

### What we will do

We will create of a local machine learning workbench aka containerized docker setup for production software-like development workflows. Having that in mind, we will use the following tools to do so:

+ [MLFlow](https://www.mlflow.org/) for experiment tracking and model management
+ [MinIO](https://min.io/) to mimic AWS S3 and act as an artifact and data storage system
+ PostgreSQL for a SQL engine and to serve as a backend store for MLFlow

Furthermore, we will train and track a collaborative filtering model and use the following tools for training & serving predictions of similar products other users may like as well with a RESTful API:

+ [implicit](https://github.com/benfred/implicit) to train our model
+ [Flask REST API](https://flask.palletsprojects.com/en/2.0.x/) to serve our predictions

We will end up with an easily and quickly configurable docker setup that gets all the above tools up and running.

### Let's set things up

For local testing purposes we can schedule our `train.py` script in the `train` container with a cronjob defined in `config/cron/config.ini` for now:

```python
[job-exec "train"]
schedule = 0 05 16 * * * 
container = recommender-local-train
command = /bin/bash -c 'pipenv run python3 /1_Train_Models/train.py'
```
To build all the individual images run

```
$ docker-compose build
```

To then start all the services run

```
$ docker-compose up -d
```

All the services should now be up and running and the cronjob should run the `train.py` script at whatever time selected.

```
$ docker container ps
```

For testing purposes of the API let's run:

```
$ python test_api.py`
```
