from flask import Flask
from flask import request
import pickle

import os

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer


app = Flask(__name__)

my_dir = os.path.dirname(__file__)
randomPath = os.path.join(my_dir, "RandomForest.pkl")
tfPath = os.path.join(my_dir, "../../../../Downloads/TF-IDF.pkl")


@app.route("/", methods=['GET', 'POST'])
def hello_world():


    loaded_model = pickle.load(open(randomPath, 'rb'))
    tf_idf = pickle.load(open(tfPath, 'rb'))
    testo = request.get_data()
    testo_cv = tf_idf.transform([testo])
    result = loaded_model.predict(testo_cv)

    conversionDict = {0 : 'religion' , 1 : 'age', 2 : 'ethnicity', 3 : 'gender', 4 : 'not_cyberbullying'}

    conversionDict[result.tolist()[0]]

    return conversionDict[result.tolist()[0]]
