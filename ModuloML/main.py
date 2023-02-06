import re
import string
import nltk

from nltk.corpus import stopwords

# importiamo le librerie necessarie da sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# importiamo le altre librerie necessarie
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_text(text: str, remove_stopwords: bool) -> str:
    """Funzione che pulisce il testo in input andando a
    - rimuovere i link
    - rimuovere i caratteri speciali
    - rimuovere i numeri
    - rimuovere le stopword
    - trasformare in minuscolo
    - rimuovere spazi bianchi eccessivi
    Argomenti:
        text (str): testo da pulire
        remove_stopwords (bool): rimuovere o meno le stopword
    Restituisce:
        str: testo pulito
    """
    # rimuovi link
    text = re.sub(r"http\S+", "", text)
    # rimuovi numeri e caratteri speciali
    text = re.sub("[^A-Za-z0-9àèìòùé]+", " ", text)
    # rimuovere le stopword
    if remove_stopwords:
        # 1. crea token
        tokens = nltk.word_tokenize(text)
        # 2. controlla se è una stopword
        tokens = [w for w in tokens if not w.lower() in stopwords.words("italian")]
        # 3. unisci tutti i token
        text = " ".join(tokens)
    # restituisci il testo pulito, senza spazi eccessivi, in minuscolo
    text = text.lower().strip()
    return text


df = pd.read_csv("Progetto Fondamenti Intelligenza artificiale.csv")
del df["Informazioni cronologiche"]
df["raw"] = df["Testo"]
del df["Testo"]
df["clean"] = df["raw"].apply(lambda x: preprocess_text(x, remove_stopwords=True))

vectorizer = TfidfVectorizer()
# fit_transform applica il TF-IDF ai testi puliti - salviamo la matrice di vettori in X
X = vectorizer.fit_transform(df['clean'])

for i, c in enumerate(df["Categoria"]):
    df["Categoria"][i] = (df["Categoria"][i])[19:]

# inizializziamo il KMeans con 6 cluster
kmeans = KMeans(n_clusters=6, random_state=20, n_init=10)
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
pca = PCA(n_components=2, random_state=42)
# passiamo alla pca il nostro array X
pca_vecs = pca.fit_transform(X.toarray())
# salviamo le nostre due dimensioni in x0 e x1
x0 = pca_vecs[:, 0]
x1 = pca_vecs[:, 1]

df['cluster'] = clusters
df['x0'] = x0
df['x1'] = x1

plt.figure(figsize=(9, 9))
# settiamo titolo
plt.title("Raggruppamento TF-IDF + KMeans", fontdict={"fontsize": 18})
# settiamo nome assi
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})
# creiamo diagramma a dispersione con seaborn, dove hue è la classe usata per raggruppare i dati
sns.scatterplot(data=df, x='x0', y='x1', hue='cluster', palette=sns.color_palette("tab10")[:6])
plt.show()

plt.figure(figsize=(9, 9))
# settiamo titolo
plt.title("Raggruppamento TF-IDF + KMeans", fontdict={"fontsize": 18})
# settiamo nome assi
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})
sns.scatterplot(data=df, x='x0', y='x1', hue='Categoria', palette=sns.color_palette("tab10")[:6])
plt.show()

f = open("dati.txt", "w")

df.to_string(encoding="utf-8", buf="dati.txt", columns=["clean", "cluster", "Categoria"])
