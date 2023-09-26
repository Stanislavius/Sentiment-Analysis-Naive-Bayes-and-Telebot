import telebot
from telebot import types
import numpy as np
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def tokenize(message_text):
    line = message_text.lower()
    res = line.replace('.', '').replace('?','').replace('!', '').split()
    stem = WordNetLemmatizer()
    res1 = res
    res = []
    for word in res1:
        res.append(stem.lemmatize(word))
    return res

def code_message(message_text):
    total_words = len(words)
    X = np.zeros(shape = (1, total_words), dtype = 'i4')
    message_text = tokenize(message_text)
    
    for i in range(len(message_text)):
        if message_text[i] in words:
            X[list(words).index(message_text[i])] +=1
    return X

def classify_text(message_text):
    X = code_message(message_text)
    return model.predict(X)

 
api = ''
bot = telebot.TeleBot(api)

with open('model.data', 'rb') as f:
    model = pickle.load(f)
with open('words.data', 'rb') as f:
    words = pickle.load(f)

@bot.message_handler(commands=['classify'])
def classify(message):
    print(message)
    if message.chat.type == "private":
        bot.reply_to(message, classify_text(message.text[len("classify")+1:]))

bot.polling(non_stop=True, interval=0)
