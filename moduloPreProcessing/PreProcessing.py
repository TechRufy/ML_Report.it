from random import random

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


def process_file(text : pd.DataFrame):
    r = pd.DataFrame(data={}, columns=["clean", "Categoria"])
    l = []
    for row in text.iterrows():
        words = preprocess_text(row[1]["Testo"], remove_stopwords=True)
        l.append((words,row[1]["Categoria"]))

    r = pd.concat([r, pd.DataFrame(data=l, columns=["clean", "Categoria"])])
    # risultati.put(r)
    return r


df = pd.read_csv("Progetto Fondamenti Intelligenza artificiale.csv", encoding="utf-8")
del df["Informazioni cronologiche"]
print(df)

start = 0
threads = []
results = multiprocessing.Queue()
n_threads = 5
size = int((df.size / df.columns.size) / n_threads)

wordnet.ensure_loaded()
stopwords.ensure_loaded()
clean = pd.DataFrame(data={}, columns=["clean","Categoria"])
lista = []

for _ in range(n_threads):
    p = df.iloc[start:(start + size)]
    start = start + size
    lista.append(p)

pre = time.time()

with ThreadPoolExecutor(max_workers=n_threads) as ex:
    risultato = ex.map(process_file, lista)
    for ris in risultato:
        clean = pd.concat([clean, ris])

post = time.time()
print("tempo {0}".format(post - pre))
print(clean)
script_dir = os.path.dirname(__file__)[:-20]
rel_path = "ModuloML/dati.csv"
abs_file_path = os.path.join(script_dir, rel_path)
f = open(abs_file_path, "w")
clean.to_csv(path_or_buf=f, encoding="ISO-8859-1", index=False)
quit()
