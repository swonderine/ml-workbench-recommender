"""
modeltraining.py
~~~~~~
Train the implicit-model
"""

import numpy as np
import implicit
from implicit.evaluation import train_test_split as train_test_split_implicit
import itertools
# from sklearn.model_selection import ParameterGrid
# from scipy.sparse import coo_matrix, csr_matrix

class TrainImplicit:
    """
    Train implicit model
    """
    
    def __init__(self,sparse_item_user):
        self.sparse_item_user = sparse_item_user
        print("ModelTrain object created")

    def _train_test_split(self,sparse):
        """
        Create training and test data
            
        Parameters
        ----------
        sparse: Dataframe
            Sparse Interaction Matrix

        Returns
        -------
        train_item_user:
            Interaction Matrix for training
        test_item_user:
            Interaction Matrix for testing
        """  
        
        
        data = train_test_split_implicit(sparse,.7)

        train_item_user, test_item_user = data[0],data[1]
        
        return train_item_user, test_item_user
    
                
    def random_search_implicit(self, num_samples = 5):
        """
        Sample random hyperparameters, fit an implicit-model, and evaluate it
        on the test set.
    
        Parameters
        ----------
    
        train: sparse csr_matrix [n_items, n_users]
            Training data.
        test: sparse csr_matrix [n_users, n_items]
            Test data.
        num_samples: int, optional
            Number of hyperparameter samples to evaluate.
    
    
        Returns
        -------
    
        generator of (map5, hyperparameter dict)
    
        """
        
        # Train & Test Data #
        train, test = self._train_test_split(self.sparse_item_user)
        
        # internal fitting to finmodel
        def fitting(train, test, num_samples):
            
            i = 0
            
            def sample_hyperparameters_implicit():
                """
                Yield possible hyperparameter choices.
                """
                
                while True:
                    yield {
                        "factors": np.random.randint(10, 300),
                        "iterations": np.random.randint(10, 100),
                        "regularization":np.random.randint(0.01,40)                    
                    }
            
            for hyperparams in itertools.islice(sample_hyperparameters_implicit(), num_samples):
                
                i = i + 1            
                
                model_implicit = implicit.als.AlternatingLeastSquares(**hyperparams)
                
                # random value between 0 & 100 - Implicit Paper suggests 40
                alpha = np.random.randint(1, 80)
                
                # create data
                data_conf = (train * alpha).astype('double')
                
                # Fit Model & Evaluate at MAP@K = 5
                model_implicit.fit((data_conf),show_progress=True)
                map5 = implicit.evaluation.mean_average_precision_at_k(model_implicit, train.tocsr(), test.tocsr(), K = 5)
                print(map5)
                
                print(f"  --- Implicit Model Fitting - Iteration {i} of {num_samples}")
                
                # Add Alpha        
                hyperparams["alpha"] = alpha
        
                yield (map5, hyperparams)
            
        
            
        # Return max MAP5 & according hyperparams from random search
        (map5, hyperparams_implicit) = max(fitting(train, test, \
                                                   num_samples), key=lambda x: x[0])
        
        # Add Key-Value with name of model & map5 to dict
        hyperparams_implicit['map5'] = float(map5)    
        
        return hyperparams_implicit
    
    
    
    def train_best(self, hyperparams):
        
        """
        Fit implicit model with specific hyperparams

        Parameters
        ----------
        hyperparams: dict
            Hyperparameters from best model fit

        Returns
        -------            
        Model: model_implicit        
        """

        
        # Initialize a model with best hyperparameters        
        model_implicit = implicit.als.AlternatingLeastSquares(factors=hyperparams["factors"],
                                                              regularization=hyperparams["regularization"],
                                                              iterations=hyperparams["iterations"]
                                                              )
        
        alpha=hyperparams["alpha"]
        data = (self.sparse_item_user * alpha).astype('double')

        
        # train the model on a sparse matrix of item/user/confidence weights
        model_implicit.fit(data)

        
        return model_implicit