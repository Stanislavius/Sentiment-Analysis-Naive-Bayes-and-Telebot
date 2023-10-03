import numpy as np
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from words_proc import tokenize

def code_message(message_text):
    total_words = len(words)
    X = np.zeros(shape = (1, total_words), dtype = 'i4')
    message_text = tokenize(message_text)
    
    for i in range(len(message_text)):
        if message_text[i] in words:
            X[0][list(words).index(message_text[i])] +=1
    return X

def classify_text(message_text):
    X = code_message(message_text)
    return model.predict(X)

with open('model.data', 'rb') as f:
    model = pickle.load(f)
with open('words.data', 'rb') as f:
    words = pickle.load(f)
