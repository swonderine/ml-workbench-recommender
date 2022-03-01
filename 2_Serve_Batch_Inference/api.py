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
with open("./0_Data/related_items.json", 'r') as file:
    implicit_related = json.load(file)
    print("related_items.json loaded")

@app.route('/related_others_liked', methods=['GET','POST'])
def related_others_liked():
    """Other users liked aswell - related products from implicit"""
    try:
        
        # Receive data
        data = request.get_json(force=True)
        print("Request received")
        
        # Extract Product-Sku
        sku = data['sku']

        # Get 10 similar items
        rel_imp = json.dumps(implicit_related[sku], indent=4)
        
        print(rel_imp)
        
        # Return sim Items from Implicit
        return(rel_imp)            
        
    
    except ValueError:
        raise RuntimeError('Unfortunately something went wrong')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
