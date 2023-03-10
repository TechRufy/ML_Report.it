import os
import pickle

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


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


df = pd.read_csv("dati.csv")

df = df[df['discriminazione'] != 'other_cyberbullying']

df['discriminazione'] = df['discriminazione'].replace(
    {'religion': 0, 'age': 1, 'ethnicity': 2, 'gender': 3, 'not_cyberbullying': 4})

X = df['testi puliti']
y = df['discriminazione']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

discriminazioni = ["religion", "age", "ethnicity", "gender", "not bullying"]

vectorizer = TfidfVectorizer(min_df=.0005, max_df=.8)
X_train_tf_idf = vectorizer.fit_transform(X_train)
X_test_tf_idf = vectorizer.transform(X_test)

clf = CountVectorizer()
X_train_cv = clf.fit_transform(X_train)
X_test_cv = clf.transform(X_test)

script_dir = os.path.dirname(__file__)[:-9]
rel_path = "CountVectorizer.sav"
abs_file_path = os.path.join(script_dir, rel_path)

pickle.dump(clf, open(abs_file_path, 'wb'))

print("iniziata la random forest")
rf = RandomForestClassifier(random_state=555)
rf.fit(X_train_cv, y_train)
print("finisce la random forest")
nb_pred = rf.predict(X_test_cv)
print('Classification Report for Random Forest with countVectorizer:\n',
      classification_report(y_test, nb_pred, target_names=discriminazioni))

conf_matrix(y_test, nb_pred, 'Random Forest Sentiment Analysis con countVectorize\nConfusion Matrix', discriminazioni)

rf = RandomForestClassifier(random_state=555)
rf.fit(X_train_tf_idf, y_train)
nb_pred = rf.predict(X_test_tf_idf)

print('Classification Report for Random Forest with tf-idf:\n',
      classification_report(y_test, nb_pred, target_names=discriminazioni))

conf_matrix(y_test, nb_pred, 'Random Forest Sentiment Analysis con tf-idf\nConfusion Matrix', discriminazioni)

script_dir = os.path.dirname(__file__)[:-9]
rel_path = "RandomForest.pkl"
abs_file_path = os.path.join(script_dir, rel_path)
pickle.dump(rf, open(abs_file_path, 'wb'))

# tf_transformer = TfidfTransformer(use_idf=True)
# tf_transformer.fit(X_train_cv)
# X_train_tf = tf_transformer.transform(X_train_cv)
# X_test_tf = tf_transformer.transform(X_test_cv)

nb_clf = MultinomialNB(alpha=0.5)
nb_clf.fit(X_train_tf_idf, y_train)
nb_pred = nb_clf.predict(X_test_tf_idf)

print('Classification Report for Naive Bayes with tf-idf:\n', classification_report(y_test, nb_pred, target_names=discriminazioni))

conf_matrix(y_test, nb_pred, 'Naive Bayes Sentiment Analysis\nConfusion Matrix', discriminazioni)

nb_clf = MultinomialNB(alpha=0.5)
nb_clf.fit(X_train_cv, y_train)
nb_pred = nb_clf.predict(X_test_cv)

print('Classification Report for Naive Bayes with CuntVectorizer:\n', classification_report(y_test, nb_pred, target_names=discriminazioni))

conf_matrix(y_test, nb_pred, 'Naive Bayes Sentiment Analysis\nConfusion Matrix', discriminazioni)

dtree = DecisionTreeClassifier()

dtree.fit(X_train_cv, y_train)
nb_pred = dtree.predict(X_test_cv)

print('Classification Report for Classification Tree with CountVectorizer:\n',
      classification_report(y_test, nb_pred, target_names=discriminazioni))

conf_matrix(y_test, nb_pred, 'Classification Tree Sentiment Analysis\nConfusion Matrix', discriminazioni)

dtree = DecisionTreeClassifier()

dtree.fit(X_train_tf_idf, y_train)
nb_pred = dtree.predict(X_test_tf_idf)

print('Classification Report for Classification Tree with tf-idf:\n',
      classification_report(y_test, nb_pred, target_names=discriminazioni))

conf_matrix(y_test, nb_pred, 'Classification Tree Sentiment Analysis\nConfusion Matrix', discriminazioni)
