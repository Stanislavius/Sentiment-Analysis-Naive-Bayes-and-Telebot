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
