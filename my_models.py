import numpy as np
import collections 
class Naive_Bayes:
    def __init__(self):
        self.apriori = None
        self.frequency = None

    def fit(self, X: np.array, y: list):
        apriori = collections.Counter(y)

        self.frequency = np.zeros(shape = (len(X[0]), len(apriori)))
        for sample, class_num in zip(X, y):
            for word, count in enumerate(sample):
                self.frequency[word][class_num] += count

        for i in range(len(self.frequency)):
            for j in range(len(self.frequency[i])):
                self.frequency[i][j] /= apriori[j]
                
        apriori = {k:v/sum(apriori.values()) for k, v in zip(apriori.keys(), apriori.values())}
        self.apriori = apriori
        
    def predict(self, X):
        result = np.zeros(shape = (len(X)))
        for i in range(len(X)):
            probs = np.zeros(shape = (len(self.apriori)))
            for cl in self.apriori.keys():
                ap_prob = self.apriori[cl]
                temp_res = 0.0
                for index, count in enumerate(X[i]):
                    temp_res += self.frequency[index][cl]*count
                probs[cl] = temp_res * ap_prob
            result[i] = probs.argmax()
        return result
