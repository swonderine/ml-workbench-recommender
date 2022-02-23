import json
import requests

# Teekanne
data = {
    "sku": "4009300010241"
}

# New Stuff
data = {
    "sku": "KCGN83228538367005"
}


ip_address = 'http://0.0.0.0:5001/similar_others_liked'

r = requests.post(ip_address, json=data)

print(r.text)
print(r.status_code)
