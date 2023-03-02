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

    # rimuovi link
    # text = re.sub(r"rt\s", " ", text)
    #
    # text = re.sub(r"http\S+", " ", text)
    #
    # text = re.sub(r"#\S+", " ", text)
    #
    # text = re.sub(r"RT\s", " ", text)
    #
    # text = re.sub(r"@\S+", " ", text)
    # # rimuovi numeri e caratteri speciali
    # text = re.sub("[^A-Za-z]+", " ", text)
    # # rimuovere le stopword
    if remove_stopwords:
        # 1. crea token
        tokens = nltk.word_tokenize(text)
        # 2. controlla se è una stopword
        tokens = [w for w in tokens if not w.lower() in listStopWord]
        # 3. unisci tutti i token
        text = " ".join(tokens)
    # restituisci il testo pulito, senza spazi eccessivi, in minuscolo

    return text


# def process_file(text):
#     r = pd.DataFrame(data={}, columns=["clean", "Category"])
#     l = []
#     for index, rows in text.iterrows():
#         words = preprocess_text(rows["testi"], remove_stopwords=True)
#         l.append((words, rows["cyberbullying_type"]))
#
#     r = pd.concat([r, pd.DataFrame(data=l, columns=["clean", "Category"])])
#     # risultati.put(r)
#     return r


data = pd.read_csv("cyberbullying_tweets.csv")

data = data.rename(columns={'tweet_text': 'testi', 'cyberbullying_type': 'discriminazione'})

data["valore_discriminazione"] = data['discriminazione'].replace(
    {"religion": 1, "age": 2, "ethnicity": 3, "gender": 4, "other_cyberbullying": 5, "not_cyberbullying": 6})

# indexNames = df[df['cyberbullying_type'] == 'not_cyberbullying'].index
# # Delete these row indexes from dataFrame
# df.drop(indexNames, inplace=True)

# indexNames = df[df['cyberbullying_type'] == 'other_cyberbullying'].index
# # Delete these row indexes from dataFrame
# df.drop(indexNames, inplace=True)


# start = 0
# threads = []
# results = multiprocessing.Queue()
# n_threads = 1
# size = int((df.size / df.columns.size) / n_threads)
#
# wordnet.ensure_loaded()
# stopwords.ensure_loaded()
# clean = pd.DataFrame(data={}, columns=["clean", "Category"])
# lista = []
#
# for _ in range(n_threads):
#     p = df.iloc[start:(start + size)]
#     start = start + size
#     lista.append(p)
#
pre = time.time()
#
# with ThreadPoolExecutor(max_workers=n_threads) as ex:
#     risultato = ex.map(process_file, lista)
#     for ris in risultato:
#         clean = pd.concat([clean, ris])


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
