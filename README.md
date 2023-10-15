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
Before passing data to model in order to train it, there are some steps that need to be done:
1. Delete any symbols from text which don't have any meaning, e.g. dots, commas, question marks, exclamation marks, semicolons etc.
2. Lower-case all symbols in order to not work with "Word" and "word" as different words.
3. Tokenization - division of sentences into separate words.
4. Remove stop-words (words that have no meaning) e.g. this, that, is, am, will, be, would, how, why, etc
5. [Stemming](https://en.wikipedia.org/wiki/Stemming) or [lemmatization](https://en.wikipedia.org/wiki/Lemmatization).
6. Vectorization, convert text into digital form. For example, [bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model) is used in this project. Words are treated as independent from each other.
7. Optionally, using n-grams. If two or more words are frequently in one sentense thay can be treated as one and it can make sense in some cases. For example: "badly want" express some strong desire for something. If treated as one word, model can be more precise when encounter it in texts. If bad and want are treated as different words, some of the meaning can be lost.

After that, when we have our text in appropriate form: 
1. Make train and test split. Usually 70% of data is for train. 10% for validation, 20% for test.
2. Train model.
3. Predict on test data and get accuracy score.

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
10. one_vs_models.py - containts OneVsRest and OneVsOne classes, which implement strategies for multiclass classificators based on binary classificators.
