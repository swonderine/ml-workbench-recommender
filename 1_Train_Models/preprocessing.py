"""
preprocessing.py
~~~~~~
Preprocess raw user data & catalog to serve as model input
"""

import pandas as pd
import numpy as np
import scipy.sparse as sparse


class PreProcess:
    """
    Clean and wrangle raw user journey data & product catalog
    """
    
    def __init__(self,catalog,raw_data):
        self.catalog = catalog
        self.raw_data = raw_data
        print("PreProcess object created")
    

    def _assign_unique_id(self,data, id_col_name):
        """
        Generate unique integer id for data (in our case users or products)

        Parameters
        ----------
        data: Dataframe
            Pandas Dataframe for df_clients or df_products
        id_col_name : String 
            new integer id's column name

        Returns
        -------
        Dataframe
            Updated dataframe containing new integer id column
        """
        
        new_df=data.assign(
            int_id_col_name=np.arange(len(data)) # start=1, stop=len(data)+1, step=1)#len(data))
            ).reset_index(drop=True)
        
        return new_df.rename(columns={'int_id_col_name': id_col_name})
    
    def _merge_data(self, left_data, left_key, right_data, right_key, how):
        """
        Merging two dataframe.
        
        Parameters
        -----------
        left_data: Dataframe
            Left side dataframe for merge
        left_key: String
            Left Dataframe merge key
        right_data: Dataframe
            Right side dataframe for merge
        right_key: String
            Right Dataframe merge key
        how: String
            Method of merge (inner, left, right, outer)
                 
        Returns
        --------
        Dataframe
            A new dataframe merging left and right dataframe
        """
        return left_data.merge(
            right_data,
            how=how,
            left_on=left_key,
            right_on=right_key)
    
    def _merge_transform_raw_data(self, data, products):
        """
        Transforms raw data extracted from the database from long to wide and connect with product catalog
        
        Parameters
        -----------
        data: Dataframe
            Raw input from user journey
        products: Dataframe
            Product Catalog
        Returns
        -------
        data: Dataframe
            Wrangled raw dataframes to be in suitable format for Modeling
            Steps:
                1 - 'EventActions' need to be in aggregated format for each client
        """
        
        # Drop rows with duplicate index values - just in case
        # data = data.loc[~data.index.duplicated(), :]
        
        # Merge with product catalog stepwise taking eventType into account
        
        # pageviews
        data_pageviews = pd.merge(data.loc[data['eventType'] == 'pageview'], products, left_on='eventData', right_on='product_url', how='inner') # inner join with "product_url"
        # purchase
        data_purchase = pd.merge(data.loc[data['eventType'] == 'purchase'], products, left_on='eventData', right_on='sku', how='inner') # inner join with "sku"
        # addToCart
        data_addToCart = pd.merge(data.loc[data['eventType'] == 'addToCart'], products, left_on='eventData', right_on='sku', how='inner') # inner join with "sku
        # removedFromCart
        data_removedFromCart = pd.merge(data.loc[data['eventType'] == 'removedFromCart'], products, left_on='eventData', right_on='sku', how='inner') # inner join with "sku"

        # Concatenate to full dataframe
        data = pd.concat([data_pageviews,data_purchase,data_addToCart,data_removedFromCart])      
        
        # Transform and Aggregate Data for each client and each product in form of pageview/purchase/addToCart/removedFromCart
        
        # Create new dataframe
        data = data.groupby(['clientId','eventType','sku'])['dateHourMinute'].count().reset_index(name="count")

        # Transform from long to wide
        data = data.pivot_table(index =['clientId','sku'], columns='eventType', values='count')
        data = data.fillna(0)
        
        # Rating of User Interactions
        #
        # Can to be explored - we use 1* - 5* ratings for now
        # 5* Purchase = Highest
        # 4* addToCart or removedFromCart but no Purchase = 2nd highest
        # 3* addToCart and removedFromCart but no Purchase = Medium
        # 2* More Pageviews = 2nd Lowest
        # 1* One Pagevew = Lowest
        
        # Predefine initial rating 
        data['rating'] = 0
        
        # purchase 
        data.loc[data['purchase'] > 0, 'rating'] = 5
        
        # One pageview only
        data.loc[(data['pageview'] == 1) & (data['removedFromCart'] == 0) & (data['addToCart'] == 0) & (data['purchase'] == 0), 'rating'] = 1
        
        # More than one pageview
        data.loc[(data['pageview'] > 1) & (data['removedFromCart'] == 0) & (data['addToCart'] == 0) & (data['purchase'] == 0), 'rating'] = 2
    
        # AddToCart and Removed from Cart but no Purchase
        data.loc[(data['addToCart'] > 0) & (data['removedFromCart'] > 0) & (data['purchase'] == 2), 'rating'] = 3
        
        # AddToCart or Removed from Cart but no Purchase
        data.loc[(data['addToCart'] > 0) | (data['removedFromCart'] > 0) & (data['purchase'] == 2), 'rating'] = 4

        return data
    
    
    def create_catalog(self, id_column = 'product_int_id'):
        """
        Creates product catalog with unique integer ids
        
        Parameters
        -----------
        id_column
            
        Returns
        -------
        products: Dataframe
            A dataframe of product with unique integer ids
        """
        
        products = self._assign_unique_id(
            self.catalog, id_column)
        
        return products
    
    def create_clients(self, id_column = 'client_int_id'):
        """
        Creates client dataframe with unique integer ids
        
        Parameters
        -----------
        self
        Returns
        -------
        clients: Dataframe
            A dataframe of clients with unique integer ids
        """
        
        ## Create Dataframe with unique client ids
        clients = self.raw_data.drop_duplicates(['clientId'], keep = 'last')[['clientId']]
        
        clients = self._assign_unique_id(
            clients, id_column)
        
        return clients
    
    def create_actions(self):
        """
        Creates dataframe of all actions by transforming, aggregating & merging of raw data with products & client data

        Returns
        -------
        actions: Dataframe
            A dataframe of all actions with unique integer ids of clients and products
        """
        
        # df_products & df_clients
        products = self.create_catalog()
        clients = self.create_clients()
        
        # Merge & transform raw Data
        user_actions = self._merge_transform_raw_data(self.raw_data, products).reset_index()
                
        # Merge transformed user_actions with product catalog
        user_actions_merged = pd.merge(user_actions, products, on = 'sku')
        
        # Merge user_actions_with clients
        user_actions_full_merge = self._merge_data(user_actions_merged, "clientId", clients, "clientId", how = "left")
        
        return user_actions_full_merge
    
    
    def transform(self):
        """
        Creates a sparse csr matrix of user-item interactions
        
        
        Returns
        -------
        sparse_item_user: Dataframe
            A dataframe of clients with unique integer ids
        """
        
        df_actions = self.create_actions()
        
        sparse_item_user = sparse.csr_matrix((df_actions['rating'].astype(float), (df_actions['product_int_id'], df_actions['client_int_id'])))
        
        return sparse_item_user

    
    
