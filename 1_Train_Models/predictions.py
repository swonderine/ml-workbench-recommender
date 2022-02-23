"""
predictions.py
~~~~~~
Batch predictions for implicit model
"""

import pandas as pd
import implicit
from collections import defaultdict
from itertools import islice

MLFLOW_ARTIFACT_ROOT = "/tmp/mlruns"

class BatchPredictions:
    """
    Train implicit model
    """
    
    def __init__(self,sparse_item_user,df_products,model_implicit):
        self.sparse_item_user = sparse_item_user
        self.df_products = df_products
        self.model_implicit = model_implicit
        
        print("Batch Predictions Object Created")

    def _similar_products_implicit(self, item_id, model_implicit, df_products, N = 11):
        """
        Predict N similar products
    
        Parameters
        ----------
    
        item_id: integer
            product integer id
        model_implicit: implicit model
            model
        N: int
            Number of similar products
    
    
        Returns
        -------
    
        sim: Dataframe
            dataframe with similar products
    
        """
                    
        # Similar products
        sim = model_implicit.similar_items(item_id,N)
                
        # Find simlar in df_products        
        first,snd = zip(*sim)
        # Access first element of tuple 
        first = pd.DataFrame(first)
        # Transform to dataframe
        first.columns = ['product_int_id']

        # Merge columns, we need for interactions with API when similar products requested
        sim = first.merge(df_products[['product_int_id','sku','name']], how='left', on='product_int_id')
        
        return sim
    
        
    def product_batch_predictions_implicit(self):
        """
        Loop through products & predict similar products according to implicit model


        Returns
        -------
        d_impl: dictionary
            dictionary of all similar products
    
        """
    
        # Initiate dictionary
        d_impl = defaultdict(dict)
        
        iters = self.sparse_item_user.shape[0] # length of rows 
        for x in range(iters):
            
            pred_similar_implicit = self._similar_products_implicit(x,
                                                                    self.model_implicit,
                                                                    self.df_products)
                
            sku = pred_similar_implicit.iloc[0]['sku']
            
            # Create dictionary for similar products
            dd_impl = defaultdict(dict)
            
            for i, row in islice(pred_similar_implicit.iterrows(), 1, None):
                dd_impl[i] = row.drop(['product_int_id']).to_dict()
                
            # Store similar products in dictionary
            d_impl[sku] = dd_impl

        
        return d_impl  