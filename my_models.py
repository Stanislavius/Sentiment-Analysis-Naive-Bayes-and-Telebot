import numpy as np
import collections


class ClassicalNaiveBayes:
    """ Just simple Naive Bayes classifier, which uses simple Bayes' theorem to classify and nothing more beyond it.
        Methods:
            fit(X :np.array, y: list) - to train on data.
            predict(X: np.array) - to predict to which classes corresponds X.
    """

    ZERO_REPLACER = 0.00000000001

    def __init__(self):
        self.apriori = None
        self.freq = None

    def fit(self, X: np.array, y: list):
        if self.apriori is None and self.freq is None:
            self.apriori = np.array(list(dict(sorted(collections.Counter(y).items())).values()))
            self.freq = np.zeros(shape=(len(self.apriori), len(X[0])))

        for sample, class_num in zip(X, y):
            self.freq[class_num] += sample
        # TODO: add recount appriori for new data or maybe not

        self.freq = np.transpose(np.transpose(self.freq) / self.apriori)
        self.apriori = self.apriori / self.apriori.sum()

    def predict(self, X: np.array):
        result = np.zeros(shape=(len(X)))
        for i in range(len(X)):
            freq = self.freq[:, X[i] != 0.0]
            freq[freq == 0.0] = 0.000000000001
            result[i] = ((freq * X[i][X[i] != 0.0]).prod(axis=1) * self.apriori).argmax()
        return result

    def __repr__(self):
        return "ClassicalNaiveBayes()"


class NaiveBayes(ClassicalNaiveBayes):
    """ Use summation instead of multiplication of p(x|y).
        Methods:
            fit(X :np.array, y: list) - to train on data.
            predict(X: np.array) - to predict to which classes corresponds X.
    """

    def predict(self, X: np.array):
        result = np.zeros(shape = (len(X)))
        for i in range(len(X)):
            result[i] = ((self.freq * X[i]).sum(axis = 1) * self.apriori).argmax()
        return result

    def __repr__(self):
        return "NaiveBayes()"


class GaussianNaiveBayes(ClassicalNaiveBayes):
    """ Gaussian Naive Bayes classifier, which uses simple Bayes' theorem, but assumes x has normal distribution.
        Methods:
            fit(X :np.array, y: list) - to train on data.
            predict(X: np.array) - to predict to which classes corresponds X.
    """
    def __init__(self):
        super().__init__()
        self.means = None
        self.stds = None

    def fit(self, X: np.array, y: list):
        if self.apriori is None:
            self.apriori = np.array(list(dict(sorted(collections.Counter(y).items())).values()))
        y_a = np.array(y)
        self.means = np.array([X[np.where(y_a == i)].mean(axis=0) for i in range(len(self.apriori))])
        self.stds = np.array([X[np.where(y_a == i)].std(axis=0) for i in range(len(self.apriori))])
        self.means[self.means == 0.0] = GaussianNaiveBayes.ZERO_REPLACER
        self.stds[self.stds == 0.0] = GaussianNaiveBayes.ZERO_REPLACER
        self.apriori = self.apriori / self.apriori.sum()

    def predict(self, X: np.array):
        result = np.zeros(shape=(len(X)))
        for i in range(len(X)):
            sample = np.array([X[i] for _ in range(len(self.apriori))])
            freq = 1 / np.sqrt(2 * np.pi * (self.stds**2)) * np.exp(-0.5 * ((sample - self.means)**2) / (self.stds**2))
            freq = freq * sample
            result[i] = (freq.sum(axis=1) * self.apriori).argmax()
        return result

    def __repr__(self):
        return "GaussianNaiveBayes()"
