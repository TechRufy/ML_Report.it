# importiamo le librerie necessarie da sklearn
import matplotlib.pyplot as plt
# importiamo le altre librerie necessarie
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def plotFunction(titolo, dati, hue):
    plt.figure(figsize=(9, 9), dpi=200)
    # settiamo titolo
    plt.title(titolo, fontdict={"fontsize": 18})
    # settiamo nome assi
    plt.xlabel("X0", fontdict={"fontsize": 16})
    plt.ylabel("X1", fontdict={"fontsize": 16})
    # creiamo diagramma a dispersione con seaborn, dove hue è la classe usata per raggruppare i dati
    sns.scatterplot(data=dati, x='x0', y='x1', hue=hue, palette="bright")
    plt.show()


df = pd.read_csv("dati.csv")

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

plotFunction("Raggruppamento TF-IDF + KMeans",df,'clusterKmeansTF')
plotFunction("Raggruppamento TF-IDF + KMeans",df,'discriminazione')
plotFunction("Raggruppamento countVectorizer + KMeans",df,'clusterKmeansVC')

# f = open("postDati.csv", "w", newline="")

# df.to_csv(encoding="ISO-8859-1", path_or_buf=f, index=False)
