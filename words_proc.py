from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def tokenize(message_text):
    line = message_text.lower()
    res = line.replace('.', '').replace('?','').replace('!', '').split()
    stem = WordNetLemmatizer()
    res1 = res
    res = []
    for word in res1:
        res.append(stem.lemmatize(word))
    return res

def stop_word(word):
    return word in stopwords.words("english")
