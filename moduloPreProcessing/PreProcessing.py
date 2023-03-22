import os
import re

import nltk
import numpy
import pandas as pd
from nltk.corpus import stopwords


def preprocess_text(text: str, remove_stopwords=True) -> str:
    """Funzione che pulisce il testo
    Argomenti:
        text (str): testo da pulire
        remove_stopwords (bool): rimuovere o meno le stopword
    Restituisce:
        str: testo pulito
    """

    """regex per la pulizia dei tweet rimuove i link, i tag e rimane solo numeri e lettere"""
    TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

    text = text.lower()

    text = re.sub(TEXT_CLEANING_RE, " ", text)

    """prendiamo le stopWord"""
    listStopWord = stopwords.words("english")

    """e ci aggiungiamo delle parole tipiche di twetter"""
    listStopWord.extend(['rt', 'mkr', 'didn', 'bc', 'n', 'm', 'im', 'll', 'y', 've', 'u', 'ur', 'don', 't', 's'])

    if remove_stopwords:
        # 1. crea token
        tokens = nltk.word_tokenize(text)
        # 2. controlla se è una stopword
        tokens = [w for w in tokens if not w.lower() in listStopWord]
        # 3. unisci tutti i token
        text = " ".join(tokens)
        # restituisci il testo pulito, senza spazi eccessivi, in minuscolo

    return text


# leggiamo il dataSet dal file
data = pd.read_csv("cyberbullying_tweets.csv")


# rinominiamo le colonne per una maggiore chiarezza
data = data.rename(columns={'tweet_text': 'testi', 'cyberbullying_type': 'discriminazione'})

# eliminiamo la categoria "other_cyberbullying" poichè troppo generica e simile alle altre
data = data[data['discriminazione'] != 'other_cyberbullying']

# controlliamo se ci sono valori nulli
print(data.isnull().sum())

# applichiamo il preprocessing ai testi
data['testi puliti'] = data['testi'].apply(preprocess_text)

# rimuoviamo i testi non puliti
del data["testi"]

# rimuoviamo degli eventuali testi nulli
data["testi puliti"].replace('', numpy.nan, inplace=True)
data.dropna(inplace=True)

# prendimao il path in cui salvare il file dopo il preprocessing
script_dir = os.path.dirname(__file__)[:-20]
rel_path = "ModuloML/dati.csv"
abs_file_path = os.path.join(script_dir, rel_path)
f = open(abs_file_path, "w", newline='')

# salviamo i dati nel file
data.to_csv(path_or_buf=f, index=False)
