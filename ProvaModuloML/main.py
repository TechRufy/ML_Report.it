# importiamo le librerie necessarie da sklearn
import matplotlib.pyplot as plt
import numpy as np
# importiamo le altre librerie necessarie
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

try:
    clean = pd.read_csv("dati.csv", encoding="ISO-8859-1")
except FileNotFoundError:
    df = None
except pd.errors.EmptyDataError:
    df = None

clean = clean.dropna()

vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)
# fit_transform applica il TF-IDF ai testi puliti - salviamo la matrice di vettori in X
X = vectorizer.fit_transform(clean["clean"].values.astype('U'))

# inizializziamo il KMeans con 3 cluster
kmeans = KMeans(n_clusters=3, random_state=20)
kmeans.fit(X)
clusters = kmeans.labels_


def get_top_keywords(n_terms):
    """Questa funzione restituisce le keyword per ogni centroide del KMeans"""
    df = pd.DataFrame(X.todense()).groupby(clusters).mean()  # raggruppa il vettore TF-IDF per gruppo
    terms = vectorizer.get_feature_names_out()  # accedi ai termini del tf idf
    for i, r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([terms[t] for t in np.argsort(r)[
                                          -n_terms:]]))  # per ogni riga del dataframe, trova gli n termini che hanno il punteggio più alto


# inizializziamo la PCA con 2 componenti
pca = PCA(n_components=2, random_state=42)
# passiamo alla pca il nostro array X
pca_vecs = pca.fit_transform(X.toarray())
# salviamo le nostre due dimensioni in x0 e x1
x0 = pca_vecs[:, 0]
x1 = pca_vecs[:, 1]

clean['cluster'] = clusters
clean['x0'] = x0
clean['x1'] = x1

plt.figure(figsize=(9, 9))
# settiamo titolo
plt.title("Raggruppamento TF-IDF + KMeans", fontdict={"fontsize": 18})
# settiamo nome assi
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})
# creiamo diagramma a dispersione con seaborn, dove hue è la classe usata per raggruppare i

sns.scatterplot(data=clean, x='x0', y='x1', hue='cluster', palette=sns.color_palette("tab10")[:6])

plt.show()
print(clean["cluster"])
f = open("dati.txt", "w")

clean.to_string(encoding="utf-8", buf="dati.txt", columns=["clean", "cluster"])
quit()
# f = open("dati.csv", "w")
# clean["clean"].to_csv(path_or_buf=f, encoding="ISO-8859-1")
