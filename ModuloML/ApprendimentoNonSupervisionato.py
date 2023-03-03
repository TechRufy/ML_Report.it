# importiamo le librerie necessarie da sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import IncrementalPCA

# importiamo le altre librerie necessarie
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dati.csv")
df = df[df['discriminazione'] != 'other_cyberbullying']
df = df[df['discriminazione'] != 'religion']
df = df[df['discriminazione'] != 'not_cyberbullying']

vectorizer = TfidfVectorizer(min_df=.0005, max_df=.8)
# fit_transform applica il TF-IDF ai testi puliti - salviamo la matrice di vettori in X
X = vectorizer.fit_transform(df['testi puliti'].values.astype('U'))

clf = CountVectorizer()
X_cv = clf.fit_transform(df["testi puliti"])


print("ciao")
# inizializziamo il KMeans con 4 cluster
kmeans = KMeans(n_clusters=3, random_state=20, n_init=10)
kmeans.fit(X_cv)
clusters = kmeans.labels_


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
pca_vecs = ipca.fit_transform(X_cv)

# salviamo le nostre due dimensioni in x0 e x1
x0 = pca_vecs[:, 0]
x1 = pca_vecs[:, 1]

df['clusterKmeans'] = clusters
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

f = open("postDati.csv", "w", newline="")

df.to_csv(encoding="ISO-8859-1", path_or_buf=f, index=False)
quit()
