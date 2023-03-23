import os
import pickle

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy import mean
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


def conf_matrix(y, y_pred, title, labels):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax = sns.heatmap(confusion_matrix(y, y_pred), annot=True, cmap='Purples', fmt='g', cbar=False,
                     annot_kws={"size": 30})
    plt.title(title, fontsize=25)
    ax.xaxis.set_ticklabels(labels, fontsize=16)
    ax.yaxis.set_ticklabels(labels, fontsize=14.5)
    ax.set_ylabel('Test', fontsize=25)
    ax.set_xlabel('Predicted', fontsize=25)
    plt.show()


def TrainingTestModel(model, FeatureTrainingSet, FeatureTestSet, LabelTrainingSet, LabelTestSet, NomeModello,
                      NomeFeature):
    # training e test del modello
    model.fit(FeatureTrainingSet, LabelTrainingSet)
    nb_pred = model.predict(FeatureTestSet)

    # stampa del report
    print('Classification Report for ' + NomeModello + ' with ' + NomeFeature + ':\n',
          classification_report(LabelTestSet, nb_pred, target_names=discriminazioni))
    # stampa della matrice di confusione
    conf_matrix(LabelTestSet, nb_pred, NomeModello + ' Sentiment Analysis ' + NomeFeature + '\nConfusion Matrix',
                discriminazioni)


# leggiamo i dati dopo il preProcessing
df = pd.read_csv("dati.csv")

# sostiuiamo i nomi delle categorie con dei numeri per semplificare la classificazione
df['discriminazione'] = df['discriminazione'].replace(
    {'religion': 0, 'age': 1, 'ethnicity': 2, 'gender': 3, 'not_cyberbullying': 4})

# dividiamo il dataset in feature e label
X = df['testi puliti']
y = df['discriminazione']

# creiamo il dataset di training e quello di test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

discriminazioni = ["religion", "age", "ethnicity", "gender", "not bullying"]

# eseguiamo il TF-IDF sul training set e poi trasformiamo entrambi i set
vectorizer = TfidfVectorizer(min_df=.0005, max_df=.8)
X_train_tf_idf = vectorizer.fit_transform(X_train)
X_test_tf_idf = vectorizer.transform(X_test)

# eseguiamo il countVectorizer allo stesso modo del TF-IDF
clf = CountVectorizer()
X_train_cv = clf.fit_transform(X_train)
X_test_cv = clf.transform(X_test)

# eseguiamo il countvectorizer anche per il k-fold cross validation
# clf = CountVectorizer()
# X_cv = clf.fit_transform(X)

# training e test del randomForest con CountVectorizer
rfCV = RandomForestClassifier(random_state=555)
TrainingTestModel(rfCV, X_train_cv, X_test_cv, y_train, y_test, "Random Forest", "CountVectorizer")

# training e test del RandomForest con TF-IDF
rfTF = RandomForestClassifier(random_state=555)
TrainingTestModel(rfTF, X_train_tf_idf, X_test_tf_idf, y_train, y_test, "Random Forest", "TF-IDF")

# # training e test del Naive Bayes con TF-IDF
nbTF = MultinomialNB(alpha=0.5)
TrainingTestModel(nbTF, X_train_tf_idf, X_test_tf_idf, y_train, y_test, "Naives Bayes", "TF-IDF")

# # training e test del Naive Bayes con il CountVectorizer
nbCV = MultinomialNB(alpha=0.5)
TrainingTestModel(nbCV, X_train_cv, X_test_cv, y_train, y_test, "Naives Bayes", "CountVectorizer")

# # training e test del DecisionTree con countVectorizer
dtreeCV = DecisionTreeClassifier()
TrainingTestModel(dtreeCV, X_train_cv, X_test_cv, y_train, y_test, "DecisionTree", "CountVectorizer")

# # training e test del DecisionTree con TF-IDF
dtreeTF = DecisionTreeClassifier()
TrainingTestModel(dtreeTF, X_train_tf_idf, X_test_tf_idf, y_train, y_test, "DecisionTree", "TF-IDF")

# esportazione del countvectorizer
script_dir = os.path.dirname(__file__)[:-9]
rel_path = "CountVectorizer.sav"
abs_file_path = os.path.join(script_dir, rel_path)
pickle.dump(clf, open(abs_file_path, 'wb'))

# esportazoine del TF-IDF
script_dir = os.path.dirname(__file__)[:-9]
rel_path = "TF-IDF.pkl"
abs_file_path = os.path.join(script_dir, rel_path)
pickle.dump(vectorizer, open(abs_file_path, 'wb'))

# esportazione del modello addestrato
script_dir = os.path.dirname(__file__)[:-9]
rel_path = "RandomForest.pkl"
abs_file_path = os.path.join(script_dir, rel_path)
pickle.dump(rfTF, open(abs_file_path, 'wb'))
