# Train & Serve a Machine Learning Model as a Microservice using Docker and Track the Model-Performance using MLflow

### Let's build a recommendation engine with a collaborative filtering model, track the model results with MLflow and use Flask to serve batch-predictions of related products other users may like as well

### What we will do

We will create of a local machine learning workbench aka containerized docker setup for production software-like development workflows. Having that in mind, we will use the following tools to do so:

+ [MLFlow](https://www.mlflow.org/) for experiment tracking and model management
+ [MinIO](https://min.io/) to mimic AWS S3 and act as an artifact and data storage system
+ PostgreSQL for a SQL engine and to serve as a backend store for MLFlow

Furthermore, we will train and track a collaborative filtering model and use the following tools for training & serving predictions of similar products other users may like as well with a RESTful API:

+ [implicit](https://github.com/benfred/implicit) to train our model
+ [Flask REST API](https://flask.palletsprojects.com/en/2.0.x/) to serve our predictions

We will end up with an easily and quickly configurable docker setup that gets all the above tools up and running.

### Data
Synthetic data created from real website data consisting of a product catalog `product_catalog.csv` and raw user interaction data `journey.csv`, needs to be copied into `0_Data`. It can be accessed [here](https://drive.google.com/drive/folders/1ntpYRe5bsLWiMlnCj-AtXNCJdEoXoKSs?usp=sharing) .

### Setting things up

For local testing purposes we can schedule our `train.py` script in the `train` container with a cronjob defined in `config/cron/config.ini`

```python
[job-exec "train"]
schedule = 0 05 16 * * * 
container = recommender-local-train
command = /bin/bash -c 'pipenv run python3 /1_Train_Models/train.py'
```
Before building the images rename `.env.example` to `.env`. To then create all the individual images run

```
$ docker-compose build
```

To then start all the services in detached mode run

```
$ docker-compose up -d
```

All the services are now be up and running and the cronjob runs the script `train.py` at whatever time specified. To check running containers run

```
$ docker container ps
```

### MLflow and Minio

We can access the MLflow GUI on http://localhost:5000 and the Minio Console on http://localhost:9000/ .


### Testing the API

The API now runs on http://localhost:5001/related_similar_items . To test it let's run the script `test_api.py` (make sure you have the relevant packages installed locally)

```
$ python3 test_api.py`
```

### Learn More

A more detailed explanation of the individual steps and services can be found [here](http://stefanbrunhuber.com/output/articles/using-docker-and-mlflow-to-deploy-and-track-machine-learning-models-with-a-local-ml-workbench.html#using-docker-and-mlflow-to-deploy-and-track-machine-learning-models-with-a-local-ml-workbench)
