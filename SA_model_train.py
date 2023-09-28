import numpy as np
import time
import sklearn
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import pickle
from words_proc import *

def load():
    encoding = "latin-1"
    path = "data"
    files = ["test", "train", "val"]
    texts = []
    y = []
    for i in range(len(files)):
        path = "data" + "/"+files[i] + "_text.txt"
        file = open(path, mode = "r", encoding = encoding)
        for line in file.readlines():
            texts.append(tokenize(line))
                
        path = "data" + "/"+files[i] + "_labels.txt"
        file = open(path, mode = "r", encoding = encoding)
        for label in file.readlines():
            y.append(int(label))
    return texts, y

def my_vectorizer(texts, y):
    words = {}
    count = {}
    total_count = {}
    class_count = {}
    temp = 0
    for i in range(len(texts)):
        for word in texts[i]:
            if(word not in class_count):
                class_count[word] = [0, 0, 0]
                class_count[word][y[i]] = 1
            else:
                class_count[word][y[i]] += 1
            if word not in words:
                words[word] = temp
                count[word] = 1
                total_count[word] = 1
            else:
                count[word] +=1
                total_count[word] += 1

    for word in class_count.keys():
        S = sum(class_count[word])
        for i in range(len(class_count[word])):
            class_count[word][i] /= S
    lens = {}
    for word in count.keys():
        if(count[word] in lens):
            lens[count[word]] += 1
        else:
            lens[count[word]] = 1

    for word in count.keys():
        """This values were found arbitrary by experementing""" 
        if (count[word] < 20):
            del(words[word])
        elif(stop_word(word) == True):
            del(words[word])
        elif(total_count[word] < 20):
            pass
        elif(max(class_count[word]) < 0.45):
            del(words[word])
        elif(min(class_count[word]) > 0.25):
            del(words[word])
 
    for word in words.keys():
        words[word] = temp
        temp = temp + 1

    total_words = len(words)
    print("Total_words = %i" % total_words)
    result = np.zeros(shape = (len(texts), total_words), dtype = 'i4')
    for i in range(len(texts)):
        for word in texts[i]:
            if(word in words):
                class_num = y[i]
                if(class_count[word][class_num] < 0.45):
                    result[i][words[word]] = 0
                else:
                    result[i][words[word]] = 1
    return result, words

texts, y = load()
X, words = my_vectorizer(texts, y)
classes = {}
for i in range(len(y)):
    if(y[i] in classes):
        classes[y[i]] += 1
    else:
        classes[y[i]] = 1
model = GaussianNB()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state  = 76)
model.fit(X_train, y_train)
print("Accuracy is %f" % accuracy_score(model.predict(X_test), y_test))
with open('model.data', 'wb') as f:
    pickle.dump(model, f)
with open('words.data', 'wb') as f:
    pickle.dump(words, f)
