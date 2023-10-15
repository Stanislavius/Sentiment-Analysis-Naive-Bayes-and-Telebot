import numpy as np
import itertools

class OneVSRest:
    def __init__(self, model):
        self.model = model
        self.models = None

    def fit(self, X: np.array, y: list):
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

    def __repr__(self):
        return "OneVSRest()"


class OneVsOne:
    def __init__(self, model):
        self.model = model
        self.models = None
        self.variants = None #in order to predifine dictionary of attributes

    def fit(self, X: np.array, y: list):
        class_num = len(set(y))
        models = [self.model() for i in range(sum(range(class_num)))] #or (class_num * (class_num - 1))/2
        classes = list(set(y))
        variants = list(itertools.product(classes, classes))
        new_variants = []
        for i, j in variants:
            if i != j and (j, i) not in new_variants:
                new_variants.append((i, j))
        variants = new_variants
        tmp = 0
        y = np.array(y)
        for i, j in variants:
            indexes_of_ij_classes = [index for index in range(len(y)) if y[index] in (i, j)]
            X_num = X[indexes_of_ij_classes]
            y_num = np.array(y)[indexes_of_ij_classes]
            y_num = np.array([0 if v == i else 1 for v in y_num])
            models[tmp].fit(X_num, y_num)
            tmp += 1
        self.models = models
        self.variants = variants
        self.class_num = class_num

    def predict(self, X:np.array):
        predictions = [self.models[i].predict(X) for i in range(len(self.models))]
        predictions = [[self.variants[i][predictions[i][j].astype("int64")] for i in range(len(self.models))] for j in range(len(X))]
        result = np.zeros(shape = (len(X), self.class_num))
        for i in range(len(predictions)):
            for j in range(len(predictions[i])):
                result[i][predictions[i][j]] += 1
        return result.argmax(axis = 1)
    
    def __repr__(self):
        return "OneVSOne()"
