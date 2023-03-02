import re
import nltk

from nltk.corpus import stopwords

# importiamo le librerie necessarie da sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn import decomposition

# importiamo le altre librerie necessarie
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dati.csv")
# indexNames = df[df['Category'] == 'other_cyberbullying'].index
# df.drop(indexNames, inplace=True)
# indexNames = df[df['Category'] == 'religion'].index
# df.drop(indexNames, inplace=True)
# indexNames = df[df['Category'] == 'age'].index
# df.drop(indexNames, inplace=True)
# indexNames = df[df['Category'] == 'ethnicity'].index
# df.drop(indexNames, inplace=True)
# indexNames = df[df['Category'] == 'gender'].index
# df.drop(indexNames, inplace=True)

vectorizer = TfidfVectorizer()  # stavi a modificare i valori qui, prova
# fit_transform applica il TF-IDF ai testi puliti - salviamo la matrice di vettori in X
X = vectorizer.fit_transform(df['testi puliti'].values.astype('U'))
print("ciao")
# inizializziamo il KMeans con 4 cluster
kmeans = KMeans(n_clusters=6, random_state=20, n_init=10)
kmeans.fit(X)
clusters = kmeans.labels_
print("ciao")
dbmeans = DBSCAN(n_jobs=-1,eps=1)
dbmeans.fit(X)
clustersDB = dbmeans.labels_

print("ciao")
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
ipca = IncrementalPCA(n_components=2, batch_size=50)
print("ciao")
# passiamo alla pca il nostro array X
pca_vecs = ipca.fit_transform(X)

# salviamo le nostre due dimensioni in x0 e x1
x0 = pca_vecs[:, 0]
x1 = pca_vecs[:, 1]

df['clusterKmeans'] = clusters
df["clusterDBmeans"] = clustersDB
df['x0'] = x0
df['x1'] = x1

plt.figure(figsize=(9, 9), dpi=200)
# settiamo titolo
plt.title("Raggruppamento TF-IDF + KMeans", fontdict={"fontsize": 18})
# settiamo nome assi
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})
# creiamo diagramma a dispersione con seaborn, dove hue è la classe usata per raggruppare i dati
sns.scatterplot(data=df, x='x0', y='x1', hue='clusterKmeans', palette="bright")
plt.show()

plt.figure(figsize=(9, 9), dpi=200)
# settiamo titolo
plt.title("Raggruppamento TF-IDF + KMeans", fontdict={"fontsize": 18})
# settiamo nome assi
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})
sns.scatterplot(data=df, x='x0', y='x1', hue='discriminazione', palette="bright")
plt.show()

plt.figure(figsize=(9, 9), dpi=200)
# settiamo titolo
plt.title("Raggruppamento TF-IDF + KMeans", fontdict={"fontsize": 18})
# settiamo nome assi
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})
sns.scatterplot(data=df, x='x0', y='x1', hue='clusterDBmeans', palette="bright")
plt.show()

f = open("postDati.csv", "w", newline="")

df.to_csv(encoding="ISO-8859-1", path_or_buf=f, index=False)
quit()
