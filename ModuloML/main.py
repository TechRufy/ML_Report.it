import re
import nltk

from nltk.corpus import stopwords

# importiamo le librerie necessarie da sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn import decomposition

# importiamo le altre librerie necessarie
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dati.csv", encoding="ISO-8859-1")
indexNames = df[df['Category'] == 'other_cyberbullying'].index
df.drop(indexNames, inplace=True)
indexNames = df[df['Category'] == 'religion'].index
df.drop(indexNames, inplace=True)
# indexNames = df[df['Category'] == 'age'].index
# df.drop(indexNames, inplace=True)
# indexNames = df[df['Category'] == 'ethnicity'].index
# df.drop(indexNames, inplace=True)
# indexNames = df[df['Category'] == 'gender'].index
# df.drop(indexNames, inplace=True)
df.dropna(inplace=True)

vectorizer = TfidfVectorizer()
# fit_transform applica il TF-IDF ai testi puliti - salviamo la matrice di vettori in X
X = vectorizer.fit_transform(df['clean'].values.astype('U'))


# inizializziamo il KMeans con 4 cluster
kmeans = KMeans(n_clusters=3, random_state=20, n_init=10)
kmeans.fit(X)
clusters = kmeans.labels_


def get_top_keywords(n_terms):
    """Questa funzione restituisce le keyword per ogni centroide del KMeans"""
    df = pd.DataFrame(X.todense()).groupby(clusters).mean()  # raggruppa il vettore TF-IDF per gruppo
    terms = vectorizer.get_feature_names_out()  # accedi ai termini del tf idf
    for i, r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([terms[t] for t in np.argsort(r)[
                                          -n_terms:]]))  # per ogni riga del dataframe, trova gli n termini che hanno
        # il punteggio più alto


# inizializziamo la PCA con 2 componenti
# pca = PCA(n_components=2, random_state=42)
ipca = IncrementalPCA(n_components=2, batch_size=10)
# passiamo alla pca il nostro array X
# pca_vecs = pca.fit_transform(X.toarray())

pca_vecs = ipca.fit_transform(X)
# salviamo le nostre due dimensioni in x0 e x1
x0 = pca_vecs[:, 0]
x1 = pca_vecs[:, 1]

df['cluster'] = clusters
df['x0'] = x0
df['x1'] = x1

plt.figure(figsize=(9, 9), dpi=200)
# settiamo titolo
plt.title("Raggruppamento TF-IDF + KMeans", fontdict={"fontsize": 18})
# settiamo nome assi
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})
# creiamo diagramma a dispersione con seaborn, dove hue è la classe usata per raggruppare i dati
sns.scatterplot(data=df, x='x0', y='x1', hue='cluster', palette="bright")
plt.show()

plt.figure(figsize=(9, 9), dpi=200)
# settiamo titolo
plt.title("Raggruppamento TF-IDF + KMeans", fontdict={"fontsize": 18})
# settiamo nome assi
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})
sns.scatterplot(data=df, x='x0', y='x1', hue='Category', palette="bright")
plt.show()

f = open("dati.txt", "w")

df.to_string(encoding="utf-8", buf="dati.txt", columns=["clean", "cluster", "Category", "x0", "x1"])
quit()
