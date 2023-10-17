import numpy as np
import sklearn
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

import time
import pickle
import copy

from words_proc import *
from time import time
from my_models import *
from one_vs_models import *


from collections import namedtuple
Sample = namedtuple('Sample', ['x', 'y'])

ENCODING = "latin-1"
PATH_TO_DATASETS = "data"
DATASETS = ["Sentiment"]
DIVISIONS = ["test", "train", "val"]


def construct_path(path_to_data: str, dataset_name: str, division_name: str, XorY: bool):
    result = path_to_data + "/" + dataset_name + "/" + division_name
    if XorY:
        return result + "_text.txt"
    else:
        return result + "_labels.txt"


class DataLoader:
    def __init__(self, dataset_name="sentiment"):
        read_samples = []
        for division in DIVISIONS:
            with (open(construct_path(PATH_TO_DATASETS, dataset_name, division, True),
                       mode="r", encoding=ENCODING) as fx,
                  open(construct_path(PATH_TO_DATASETS, dataset_name, division, False),
                       mode="r", encoding=ENCODING) as fy):
                while True:
                    x = fx.readline()
                    y = fy.readline()
                    if x == "":
                        break
                    else:
                        read_samples.append(Sample(x, y))

        self.read_samples = read_samples
        self.i = 0
        
    def __iter__(self):
        return iter(self.read_samples)

    def __len__(self):
        return len(self.read_samples)

    def __getitem__(self, key):
        if type(key) == int:
            return self.read_samples[key]
        elif type(key) == str:
            if key == "X":
                return [sample.x for sample in self.read_samples]
            if key == "y":
                return [sample.y for sample in self.read_samples]
    
    def __next__(self):
        self.i += 1
        return self.read_samples[self.i-1]


def load():
    texts = []
    y = []
    loader = DataLoader()
    for sample in loader:
        texts.append(tokenize(sample.x))
        y.append(int(sample.y))
    return texts, y


def my_vectorizer(texts, y):
    total_count, class_count = {}, {}
    class_num = len(set(y))
    for i in range(len(texts)):
        for word in texts[i]:
            if(word not in class_count):
                class_count[word] = [0 for c in range(class_num)]
                class_count[word][y[i]] = 1 #MARK THAT WORD APPEARS IN THIS CLASS
            else:
                class_count[word][y[i]] += 1
                
    count = {k : sum(class_count[k]) for k in class_count.keys()}
    total_count = copy.deepcopy(count)
    for word in class_count.keys():
        class_count[word] = [class_count[word][i] / total_count[word] for i in range(len(class_count[word]))]

    count = {word: c for word, c in count.items() if c >= 20
             if stop_word(word) == False
             if max(class_count[word]) >= 0.45
             if min(class_count[word]) <= 0.25
             } #deletion words under these criterias
    
    words = {word : i for i, word in enumerate(count.keys())}
    result = np.zeros(shape = (len(texts), len(words)), dtype = 'i4')
    threshold = 0.45
    for i in range(len(texts)):
        for word in texts[i]:
            if word in words:
                class_num = y[i]
                if class_count[word][class_num] < threshold:
                    result[i][words[word]] = 0
                else:
                    result[i][words[word]] += 1

    return result, words


def my_label_encoding(texts, y):
    unique = {}
    ec = {} #0 - padding
    temp = 1
    for line in texts:
        for word in line:
            if word not in unique:
                unique[word] = 1
                ec[word] = temp
                temp = temp + 1
            else:
                unique[word] +=1

    maxlen = max([len(texts[i]) for i in range(len(texts))])
    X = np.zeros(shape = (len(texts), maxlen))
    for i in range(len(texts)):
        for j in range(len(texts[i])):
            X[i][j] = ec[texts[i][j]]
            
    return X, ec


def time_count(func):
    def wrapper(*args, **kwargs):
        t1 = time()
        func(*args,  **kwargs)
        t2 = time()
        print("Executed in %i" % (t2-t1))
    return wrapper


@time_count
def model_training():
    model.fit(X_train, y_train)
    print("Accuracy is %f" % accuracy_score(model.predict(X_test), y_test))
    #SAVING RESULTS
    with open('model.data', 'wb') as f:
        pickle.dump(model, f)
    with open('words.data', 'wb') as f:
        pickle.dump(words, f)


texts, y = load()
label_encoding = False
if label_encoding:
    X, words = my_label_encoding(texts, y)
else:
    X, words = my_vectorizer(texts, y)

models = {0: GaussianNB(), 1: NaiveBayes(), 2: OneVSOne(NaiveBayes), 3: OneVSRest(NaiveBayes)}

num = 1
model = models[num]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                    random_state  = 76)
model_training()
