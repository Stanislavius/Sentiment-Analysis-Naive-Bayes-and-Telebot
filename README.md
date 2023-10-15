## Sentiment-Analysis-Naive-Bayes-and-telebot
# Contents
* [What it does?](#what-it-does)
* [Why?](#why)
* [How?](#how)
* [How to run?](#how-to-run)
* [How to use?](#how-to-use)
* [Structure of project](#structure-of-project)

# What it does?
Analyses [sentiment](https://en.wikipedia.org/wiki/Sentiment_analysis) or emotional tone of a sentense (e.g. 0 - negative, 1 - positive, or it can be like 0 - joy, 1 - anger, 2 - sadness, 3 - amusement, 4 - disgust and so on and so forth).
In this case there is three type of texts for now: 0 - negative, 1 - neutral, 2 - positive. Given text of arbitrary length, it determines if text is negative, neutral or positive.
# Why?
Sentiment analysis can be used:
1. to automatically determine by text of user review if user is satisfied with a product;
2. to detect spam messages;
3. to detect hate speech of message that violates rules of forum or community or chat group etc.
   
# How?
The analysis of the tonality of the text can be considered as a classification task. 
Uses [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) to classify sentense, it can be done by enother methods which may be implemented later.
For training was used dataset from [tweeteval](https://github.com/cardiffnlp/tweeteval/tree/main/datasets/sentiment). Telegram bot is used to communicate with user.

# How to run?
1. Install Python 3.9.
2. Install Python libraries: sklearn, NLTK, pickle, telebot, NumPy. If you are using pip, open command prompt and type "pip3 install <library name">
3. Run nltk_download.py to download dependencies
4. Run SA_model_train.py to train model on included train data.
5. Write to https://t.me/BotFather, create bot and get token for him.
6. Create file bot_api.txt in root folder of project. Write api in it. E.g.:

    ![image](https://github.com/Stanislavius/Sentiment-Analysis-Naive-Bayes-and-Telebot/assets/56927835/62ff7318-39ca-4cce-96d4-c3d701df6342)
8. Run SA_bot.py


# How to use?
 Open chat with your bot. type /classity and write your message to classify it's sentiment.
1. Type /classity and write your message to classify it's sentiment:

![image](https://github.com/Stanislavius/Sentiment-Analysis-Naive-Bayes-and-telebot-/assets/56927835/62b91d4b-937c-4e65-a0cb-57874df85d91)

2. Or send a text document in txt format where each line is a message.

![image](https://github.com/Stanislavius/Sentiment-Analysis-Naive-Bayes-and-Telebot/assets/56927835/93a6c649-b9de-4aec-a5cc-d157f2de9518) 

Test input document:

![image](https://github.com/Stanislavius/Sentiment-Analysis-Naive-Bayes-and-Telebot/assets/56927835/cc182637-6688-4002-b00c-8f7a41afef75)

Test output document:

![image](https://github.com/Stanislavius/Sentiment-Analysis-Naive-Bayes-and-Telebot/assets/56927835/398500b6-401e-4b7e-92cf-f1f4609e3137)


# Structure of project
Files and their main function:
1. Data folder - data for training.
2. SA_bot.py - telegram bot able to recieve command, pass text to classifier and send back the result.
3. SA_classification.py - usage of trained model to classificate input.
4. SA_model_train.py - training of model.
5. model.data - saved trained model.
6. nltk_download.py - to download dependencies inside NLTK.
7. words.data - words which were used to train and their positions.
8. words_proc.py - some functions to process text data.
9. my_models.py - contains models written by me, Naive Bayes only for now.
