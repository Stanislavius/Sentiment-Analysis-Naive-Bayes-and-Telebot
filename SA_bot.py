import telebot
from telebot import types
from SA_classification import *

def read_api():
    PATH_API = "bot_api.txt"
    with open(PATH_API, "r") as f:
        return f.read()
    
api = read_api()
bot = telebot.TeleBot(api)

@bot.message_handler(commands=['classify'])
def classify(message):
    if message.chat.type == "private":
        bot.reply_to(message, classify_text(message.text[len("classify")+1:]))
        
bot.polling(non_stop=True, interval=0)
