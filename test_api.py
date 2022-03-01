import requests

# Dictionary with sku
data = {
    "sku": "SLFI54432219010837"
}

# Address
ip_address = 'http://0.0.0.0:5001/related_others_liked'

r = requests.post(ip_address, json=data)

print(r.text)
