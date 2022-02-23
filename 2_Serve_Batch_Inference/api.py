"""
api.py
~~~~~~
Defines a simple REST API with Flask for Batch-Inference
"""

from flask import Flask, request # abort, jsonify, make_response
# from pandas import DataFrame
import os
import warnings
import json
# Ignore warnings
warnings.filterwarnings('ignore')

# Environment
service_name = os.environ['SERVICE_NAME']
version = os.environ['API_VERSION']

# Flask-API
app = Flask(__name__)

# Load similar items
with open("./0_Data/similar_items.json", 'r') as file:
    implicit_similar = json.load(file)
    print("similar_items.json loaded")

@app.route('/similar_others_liked', methods=['GET','POST'])
def similar_others_liked():
    """Other users liked aswell - similar products from implicit"""
    try:
        
        # Receive data - https://www.digitalocean.com/community/tutorials/processing-incoming-request-data-in-flask #
        data = request.get_json(force=True)
        print("Request received")
        
        # Extract Product-Sku
        sku = data['sku']

        # Get 10 similar items
        sim_imp = json.dumps(implicit_similar[sku], indent=4) #, sort_keys=True)
        
        print(sim_imp)
        
        # Return sim Items from Implicit
        return(sim_imp)            
        
    
    except ValueError:
        raise RuntimeError('Unfortunately something went wrong')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
