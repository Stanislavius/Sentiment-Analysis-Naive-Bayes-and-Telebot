import numpy as np
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from words_proc import tokenize


def code_messages(texts):
    total_words = len(words)
    X = np.zeros(shape = (len(texts), total_words), dtype = 'i4')
    texts = [tokenize(text) for text in texts]
    for i, text in enumerate(texts):
        for j in range(len(text)):
            if text[j] in words:
                X[i][list(words).index(text[j])] +=1
    return X


def classify_text(message_text):
    X = code_messages([message_text])
    return model.predict(X)


def classify_file(file_name):
    with open(file_name, 'r') as f:
        texts = [line for line in f.readlines()]
    X = code_messages(texts)
    predictions = model.predict(X)
    results_name = "results.txt"
    with open(results_name, 'w') as f:
        for p in predictions:
            f.write(str(p) + "\n")
    return results_name


with open('model.data', 'rb') as f:
    model = pickle.load(f)
with open('words.data', 'rb') as f:
    words = pickle.load(f)
