from random import random

import numpy
import sklearn.datasets as DS
import pandas as pd
import time
import threading
import nltk
import multiprocessing
import re
import tqdm
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from tqdm.auto import tqdm

from nltk.corpus import stopwords
from nltk.corpus import wordnet


def preprocess_text(text: str, remove_stopwords=True) -> str:
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

    TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

    text = text.lower()

    text = re.sub(TEXT_CLEANING_RE, " ", text)

    listStopWord = stopwords.words("english")

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


data = pd.read_csv("cyberbullying_tweets.csv")

data = data.rename(columns={'tweet_text': 'testi', 'cyberbullying_type': 'discriminazione'})

pre = time.time()


data['testi puliti'] = data['testi'].apply(preprocess_text)

del data["testi"]

post = time.time()
print("tempo {0}".format(post - pre))
script_dir = os.path.dirname(__file__)[:-20]
rel_path = "ModuloML/dati.csv"
abs_file_path = os.path.join(script_dir, rel_path)
f = open(abs_file_path, "w", newline='')
data["testi puliti"].replace('', numpy.nan, inplace=True)
data.dropna(inplace=True)
data.to_csv(path_or_buf=f, index=False)
quit()
