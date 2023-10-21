import numpy as np
import sklearn
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

import time
import pickle
import sys
import argparse


from words_proc import *
from time import time
from my_models import *
from one_vs_models import *

from collections import namedtuple

Sample = namedtuple('Sample', ['x', 'y'])

encoding = None
path_to_datasets = None

ENCODING = "latin-1"
PATH_TO_DATASETS = "data"
DATASETS = ["sentiment"]
DIVISIONS = ["test", "train", "val"]  # for now support only for this structure of data
DEFAULT_MODEL = "NB"


def construct_path(path_to_data: str, dataset_name: str, division_name: str, XorY: bool):
    result = path_to_data + "/" + dataset_name + "/" + division_name
    if XorY:
        return result + "_text.txt"
    else:
        return result + "_labels.txt"


def construct_path_mapping(path_to_data: str, dataset_name: str):
    return path_to_data + "/" + dataset_name + "/" + "mapping.txt"


def get_samples(dataset_name="sentiment"):
    read_samples = []
    for division in DIVISIONS:
        with (open(construct_path(path_to_datasets, dataset_name, division, True),
                   mode="r", encoding=encoding) as fx,
              open(construct_path(path_to_datasets, dataset_name, division, False),
                   mode="r", encoding=encoding) as fy):
            while True:
                x = fx.readline()
                y = fy.readline()
                if x == "":
                    break
                else:
                    read_samples.append(Sample(x, y))

    map_classes = {}
    with open(construct_path_mapping(path_to_datasets, dataset_name), mode="r", encoding=encoding) as f:
        for line in f.readlines():
            map_classes[int(line[:line.index("\t")])] = line[line.index("\t") + 1:-1]
    return iter(read_samples), map_classes


def load():
    texts, y = [], []
    samples, mapping = get_samples()
    for sample in samples:
        texts.append(lemmatize(tokenize(sample.x)))
        y.append(int(sample.y))
    return texts, y, mapping


def deletion_rule(count, class_count):  # to delete words if they are not passing some criteria
    new_count = {word: c for word, c in count.items()
                 if c >= 5
                 if stop_word(word) is False
                 if max(class_count[word]) >= 0.45
                 if min(class_count[word]) <= 0.25
                 }  # deletion words under these criteria
    return new_count


def bag_of_words(texts, y, rule=deletion_rule, threshold=0.45):
    """
    Transforms texts into bag-of-words form (array of shape (N_samples, N_words))
    texts: list of texts
    y: list of actual classes
    rule: rule under which words would be deleted from dataset
    threshold: if word appears in class less than threshold times - it is deleted.
    return: bag of words representation and dictionary with keys-words and values-their code
    """
    total_count, class_count = {}, {}
    class_num = len(set(y))
    for i in range(len(texts)):
        for word in texts[i]:
            if word not in class_count:
                class_count[word] = [0 for c in range(class_num)]
                class_count[word][y[i]] = 1  # MARK THAT WORD APPEARS IN THIS CLASS
            else:
                class_count[word][y[i]] += 1

    count = {k: sum(class_count[k]) for k in class_count.keys()}
    for word in class_count.keys():
        class_count[word] = [class_count[word][i] / count[word] for i in range(len(class_count[word]))]

    count = rule(count, class_count)

    words = {word: i for i, word in enumerate(count.keys())}
    result = np.zeros(shape=(len(texts), len(words)), dtype='i4')
    for i in range(len(texts)):
        for word in texts[i]:
            if word in words:
                class_num = y[i]
                if class_count[word][class_num] < threshold:
                    result[i][words[word]] = 0
                else:
                    result[i][words[word]] += 1

    return result, words


def label_encoding(texts, y):
    unique = {}
    ec = {}  # 0 - padding
    temp = 1
    for line in texts:
        for word in line:
            if word not in unique:
                unique[word] = 1
                ec[word] = temp
                temp = temp + 1
            else:
                unique[word] += 1

    maxlen = max([len(texts[i]) for i in range(len(texts))])
    X = np.zeros(shape=(len(texts), maxlen))
    for i in range(len(texts)):
        for j in range(len(texts[i])):
            X[i][j] = ec[texts[i][j]]

    return X, ec


def time_count(func):
    def wrapper(*args, **kwargs):
        t1 = time()
        func(*args, **kwargs)
        t2 = time()
        print("Executed in %i" % (t2 - t1))

    return wrapper


@time_count
def model_training(model, X_train, y_train, X_test, y_test, words, mapping):
    model.fit(X_train, y_train)
    print("Accuracy is %f" % accuracy_score(model.predict(X_test), y_test))
    # SAVING RESULTS
    with open('model.data', 'wb') as f:
        pickle.dump(model, f)
    with open('words.data', 'wb') as f:
        pickle.dump(words, f)
    with open('mapping.data', 'wb') as f:
        pickle.dump(mapping, f)


def main():
    global path_to_datasets
    global encoding
    models = {"GNB": GaussianNaiveBayes(), "NB": NaiveBayes(), "OVO(NB)": OneVSOne(NaiveBayes),
              "OVS(NB)": OneVSRest(NaiveBayes)}

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model",
                        help="""model selection:
                                    GNB: Gaussian Naive Bayes;
                                    NB: simple Naive Bayes;
                                    OVO(NB) - OneVSOne(NaiveBayes);
                                    OVS(NB) - OneVSRest(NaiveBayes).""",
                        default=DEFAULT_MODEL)
    parser.add_argument("-e", "--encoding", help = "choosing of encoding of data files", default=ENCODING)
    parser.add_argument("-p","--path_to_data", help="choosing folder with data", default=PATH_TO_DATASETS)
    args = parser.parse_args()
    model = models[args.model]
    encoding = args.encoding
    path_to_datasets = args.path_to_data

    texts, y, mapping = load()
    encode_as_label = False
    if encode_as_label:
        X, words = label_encoding(texts, y)
    else:
        X, words = bag_of_words(texts, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        random_state=76)
    model_training(model, X_train, y_train, X_test, y_test, words, mapping)


if __name__ == "__main__":
    main()
