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


@bot.message_handler(content_types=['document'])
def document_handler(message):
    if message.chat.type == "private":
        bot.reply_to(message, "File is being downloaded.")
        file_name = message.document.file_name
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        file_path = "downloaded_files/" + file_name
        with open(file_path, 'wb') as new_file:
            new_file.write(downloaded_file)
        result_path = classify_file(file_path)
        bot.send_document(message.chat.id, open(result_path, 'rb'))


bot.polling(non_stop=True, interval=0)
