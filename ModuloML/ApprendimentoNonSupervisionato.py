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

# x_pos = np.arange(len(df["discriminazione"].unique()))

# plt.figure(figsize=(9, 5))
# plt.bar(x_pos, df["discriminazione"].value_counts(), align='center')
# plt.xticks(x_pos, df["discriminazione"].unique())
# plt.ylabel('Numero di tweet')
# plt.xlabel('Categoria')
# plt.title('tweet per categoria')
# plt.show()
# print(df["discriminazione"].value_counts())


# df = df[df['discriminazione'] != 'other_cyberbullying']
# df = df[df['discriminazione'] != 'religion']
# df = df[df['discriminazione'] != 'not_cyberbullying']

vectorizer = TfidfVectorizer(min_df=.0005, max_df=.8)
# fit_transform applica il TF-IDF ai testi puliti - salviamo la matrice di vettori in X
X_tf = vectorizer.fit_transform(df['testi puliti'].values.astype('U'))

clf = CountVectorizer()
X_cv = clf.fit_transform(df["testi puliti"])

# inizializziamo il KMeans con 4 cluster
kmeansTF = KMeans(n_clusters=5, random_state=20, n_init=10)
kmeansTF.fit(X_cv)
clustersTF = kmeansTF.labels_

kmeansVC = KMeans(n_clusters=5, random_state=20, n_init=10)
kmeansVC.fit(X_tf)
clustersVC = kmeansVC.labels_

# inizializziamo la PCA con 2 componenti
ipca = IncrementalPCA(n_components=2, batch_size=50)
# passiamo alla pca il nostro array X
pca_vecs = ipca.fit_transform(X_cv)

# salviamo le nostre due dimensioni in x0 e x1
x0 = pca_vecs[:, 0]
x1 = pca_vecs[:, 1]

df['clusterKmeansTF'] = clustersTF
df['clusterKmeansVC'] = clustersVC
df['x0'] = x0
df['x1'] = x1

plt.figure(figsize=(9, 9), dpi=200)
# settiamo titolo
plt.title("Raggruppamento TF-IDF + KMeans", fontdict={"fontsize": 18})
# settiamo nome assi
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})
# creiamo diagramma a dispersione con seaborn, dove hue è la classe usata per raggruppare i dati
sns.scatterplot(data=df, x='x0', y='x1', hue='clusterKmeansTF', palette="bright")
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
plt.title("Raggruppamento countVectorizer + KMeans", fontdict={"fontsize": 18})
# settiamo nome assi
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})
sns.scatterplot(data=df, x='x0', y='x1', hue='clusterKmeansVC', palette="bright")
plt.show()

#f = open("postDati.csv", "w", newline="")

#df.to_csv(encoding="ISO-8859-1", path_or_buf=f, index=False)

