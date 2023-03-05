from flask import Flask
from flask import request
import pickle

from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def hello_world():
    loaded_model = pickle.load(open("RandomForest.sav", 'rb'))
    clf = pickle.load(open("CountVectorizer.sav", 'rb'))
    testo = request.get_data()
    testo_cv = clf.transform([testo])
    result = loaded_model.predict(testo_cv)
    print(result)

    conversionDict = {0 : 'religion' , 1 : 'age', 2 : 'ethnicity', 3 : 'gender', 4 : 'not_cyberbullying'}

    conversionDict[result.tolist()[0]]

    return conversionDict[result.tolist()[0]]
