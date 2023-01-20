import requests

r = requests.post("https://mlreport.azurewebsites.net/", json={"ciaot": "ao"})

print(r.json())

