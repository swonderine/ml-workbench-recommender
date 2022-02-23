import json
import requests

# Dictionary with sku
data = {
    "sku": "KCGN83228538367005"
}

# Address
ip_address = 'http://0.0.0.0:5001/similar_others_liked'

r = requests.post(ip_address, json=data)

print(r.text)
print(r.status_code)
