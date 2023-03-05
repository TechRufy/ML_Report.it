import requests

r = requests.post("http://127.0.0.1:5000", data="Woman are all stupid")

print(r.text)

