import itertools

import numpy as np
import collections

class NaiveBayes:
    def __init__(self):
        self.apriori = None
        self.freq = None

    def fit(self, X: np.array, y: list):
        if self.apriori is None and self.freq is None:
            self.apriori = np.array(list(dict(sorted(collections.Counter(y).items())).values()))
            self.freq = np.zeros(shape = (len(self.apriori), len(X[0])))
        
        for sample, class_num in zip(X, y):
            self.freq[class_num] += sample
        #TODO: add recount appriori for new data or maybe not

        self.freq = np.transpose(np.transpose(self.freq) / self.apriori)
        self.apriori = self.apriori / self.apriori.sum()
        
    def predict(self, X: np.array):
        result = np.zeros(shape = (len(X)))
        for i in range(len(X)):
            result[i] = ((self.freq * X[i]).sum(axis = 1) * self.apriori).argmax()
        return result


    def __repr__(self):
        return "NaiveBayes()"

class OneVSRest:
    def __init__(self, model):
        self.model = model

    def fit(self, X : np.array, y: list):
        class_num = len(set(y))
        models = [self.model() for i in range(class_num)]
        for i in range(class_num):
            y_num = [0 for i in range(len(y))]
            for j in range(len(y_num)):
                if y[j] != i:
                    y_num[j] = 1
            models[i].fit(X, y_num)
        self.models = models

    def predict(self, X:np.array):
        predictions = np.array([self.models[i].predict(X) for i in range(len(self.models))])
        predictions = predictions.transpose()
        return predictions.argmin(axis = 1)

class OneVsOne:
    def __init__(self, model):
        self.model = model

    def fit(self, X : np.array, y: list):
        class_num = len(set(y))
        models = [self.model() for i in range(sum(range(class_num)))] #or (class_num * (class_num - 1))/2
        classes = list(set(y))
        variants = list(itertools.product(classes, classes))
        new_variants = []
        for v in variants:
            i, j = v
            if i != j and (j, i) not in new_variants:
                new_variants.append(v)
        variants = new_variants
        tmp = 0
        for i, j in variants:
            y_num = []
            X_num = []
            for k in range(len(y)):
                if y[k] in (i, j):
                    X_num.append(X[k])
                    y_num.append(y[k])
            X_num = np.array(X_num)
            y_num = [0 if v == i else 1 for v in y_num]
            models[tmp].fit(X_num, y_num)
            tmp += 1
        self.models = models
        self.variants = variants
        self.class_num = class_num

    def predict(self, X:np.array):
        predictions = [self.models[i].predict(X) for i in range(len(self.models))]
        predictions = [[self.variants[i][predictions[i][j].astype("int64")] for i in range(len(self.models))] for j in range(len(X))]
        print(self.variants)
        print(predictions[0], predictions[1], predictions[2])
        result = np.zeros(shape = (self.class_num, len(X)))
        for i in range(len(predictions)):
            for j in range(len(predictions[i])):
                result[predictions[i][j]] += 1
        return result.argmax(axis = 0)
