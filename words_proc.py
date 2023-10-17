from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def tokenize(message_text):
    line = message_text.lower()
    res = line.replace('.', '').replace('?','').replace('!', '').split()
    res1 = res
    res = []
    for word in res1:
        res.append(word)
    return res


def lemmatize(message_text):
    for i, word in enumerate(message_text):
        message_text[i] = lemmatizer.lemmatize(word)
    return message_text


def stop_word(word):
    return word in stopwords.words("english")


lemmatizer = WordNetLemmatizer()
