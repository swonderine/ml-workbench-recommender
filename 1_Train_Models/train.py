"""
train.py
~~~~~~
Train implicit model
Major steps:
    #1 Load Data
    #2 Preprocess Data
    #3 Training
    #4 Batch Predicitions
    #5 MLflow logging metrics & parameters
    #6 MLflow saving and logging of model
"""

import os
# import sqlalchemy - needed if data is fetched from Database
import pandas as pd
import implicit
import joblib
import json
from datetime import date
import mlflow
import logging
from urllib.parse import urlparse
from sys import version_info
import cloudpickle
import warnings
warnings.filterwarnings('ignore')

# Python version
PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)

# Logger
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Print environment variables
for k, v in sorted(os.environ.items()):
    print(k+':', v)
print('\n')

# Set environment variables for MLflow
MIN_SAMPLE_OUTPUT = 35

# MLflow settings
mlflow_settings = dict(
    username="mlflow",
    password="mlflow",
    host="mlflow-server",
    port=5000,
)

# Set tracking uri for MLflow
mlflow.set_tracking_uri(
    "http://{username}:{password}@{host}:{port}".format(**mlflow_settings)
    )

# Set experiment id for MLflow 
current_date = date.today()
experiment_id = mlflow.set_experiment("recommender_similar_items")

# Create a Conda environment for the new MLflow Model that contains all necessary dependencies
conda_env = {
     'channels': ['defaults'],
     'dependencies': [
       'python={}'.format(PYTHON_VERSION),
       'pip',
       {
         'pip': [
           'mlflow=={}'.format(implicit.__version__),
           'implicit=={}'.format(implicit.__version__),
           'cloudpickle=={}'.format(cloudpickle.__version__),
           'joblib=={}'.format(joblib.__version__),
         ],
       },
     ],
     'name': 'implicit_env'
}


def train():
    
    with mlflow.start_run(run_name=f"recommender_{current_date}"):
        
            
        #############################################################################
        # ----------------------------------- # 1 --------------------------------- #
        # ------------- Load Product Catalog & Raw User Journey Data -------------- #
        #############################################################################
        
        product_catalog = pd.read_csv("./0_Data/product_catalog.csv")
        raw_data =  pd.read_csv("./0_Data/journey.csv")
    
        
        #############################################################################
        # ---------------------------------- # 2 ---------------------------------- #
        # ---------------------------  Data preprocessing ------------------------- #
        #############################################################################
    
        # Import Class PreProcess
        from preprocessing import PreProcess
        
        # Instantiate Object
        pre = PreProcess(product_catalog,raw_data)
        
        # Create product dataframe
        df_products = pre.create_catalog()
  
        # Create sparse item user matrix
        sparse_item_user = pre.transform()
        
        
        # Log df_products as MLflow artifact
        df_products.to_csv("0_Data/df_products.csv")
        mlflow.log_artifact("0_Data/df_products.csv","data/")
        
                
        #############################################################################
        # ---------------------------------- # 3 ---------------------------------- #
        # ----------------- Find best hyper-parameters & train model -------------- #
        #############################################################################
        
        # Import Class TrainImplicit
        from modeltraining import TrainImplicit
        
        # Instantiate Object
        training = TrainImplicit(sparse_item_user)
        
        # Find best model and get hyperparameters
        best_hyperparams = training.random_search_implicit(num_samples=20)
        
        # Fit Model with best hyperparameters
        trained_model = training.train_best(best_hyperparams)
        
        #############################################################################
        # ---------------------------------- # 4 ---------------------------------- #
        # ----------------- Batch Predictions for similar products  --------------- #
        #############################################################################
        
        from predictions import BatchPredictions
        
        # Instantiate Object
        pred = BatchPredictions(sparse_item_user,df_products,trained_model)
        
        # Batch Predictions
        similar_items = pred.product_batch_predictions_implicit()
        
        # Store similar_items.json in 0_Data
        with open("0_Data/similar_items.json", 'w') as outfile:
            json.dump(similar_items, outfile, indent = 4, sort_keys = False)
            print("similar_implicit.json stored in 0_Data")
        # Log similar_items.json as artifact
        mlflow.log_dict(similar_items, "data/similar_items.json")
        
        #############################################################################
        # --------------------------------- # 5 ----------------------------------- #
        # ------------------- MLflow - Logging Metrics &Paramters------------------ #
        #############################################################################
        
        # Log Hyperparameters & MAP@5
        mlflow.log_metric("MAPat5", best_hyperparams["map5"])
        mlflow.log_param("alpha", best_hyperparams["alpha"])
        mlflow.log_param("factors", best_hyperparams["factors"])
        mlflow.log_param("regularization", best_hyperparams["regularization"])
        mlflow.log_param("iterations", best_hyperparams["iterations"])
        mlflow.log_param("Date", current_date)
        
        #############################################################################
        # ---------------------------------- # 6 ---------------------------------- #
        # ---------------------------- MLflow save & log model -------------------- #
        #############################################################################
        
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        print(tracking_url_type_store)
        
        from implicitwrapper import ImplicitWrapper
        
        # Model name & path 
        model_name = "implicit_model"
        implicit_model_path = model_name + ".joblib"
                    
        # Store implicit model as joblib file
        joblib.dump(trained_model, implicit_model_path, compress=True)
        print('Implicit Model saved')

        # Create an 'artifacts' dictionary that assigns a unique name to the saved implicit model file.
        # This dictionary will be passed to 'mlflow.pyfunc.save_model', which will copy the model file
        # into the new MLflow Model's directory.

        artifacts = {
            "implicit_model": implicit_model_path
            }
        
        mlflow_pyfunc_model_path = model_name
        
        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            mlflow.pyfunc.log_model("model",
                                     registered_model_name="implicit_model",
                                     python_model=ImplicitWrapper(),
                                     artifacts=artifacts)
        else:
            mlflow.pyfunc.log_model("model",
                                     path=mlflow_pyfunc_model_path,
                                     python_model=ImplicitWrapper(),
                                     artifacts=artifacts)

if __name__ == '__main__':
    train()


# Sources
# https://stackoverflow.com/questions/65887231/use-mlflow-to-serve-a-custom-python-model-for-scoring
# https://towardsdatascience.com/mlflow-part-3-logging-models-to-a-tracking-server-54b6aa3cd00f
# https://www.alexanderjunge.net/blog/mlflow-sagemaker-deploy/
# https://stackoverflow.com/questions/56627446/docker-compose-how-to-use-minio-in-and-outside-of-the-docker-network
# https://qooba.net/2021/05/22/feast-with-ai-feed-your-mlflow-models-with-feature-store/
# https://github.com/darshikaf/mlflow-tracking-server-with-minio/blob/master/tests/mlflow_client_test.py
# https://medium.com/@pennyqxr/how-save-and-load-fasttext-model-in-mlflow-format-37e4d6017bf0




