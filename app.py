from flask import Flask
from flask import request
import pickle

from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def hello_world():
    loaded_model = pickle.load(open("RandomForest.sav", 'rb'))
    clf = CountVectorizer()
    testo = request.get_data()
    result = loaded_model.predict(testo)
    print(result)

    return result
