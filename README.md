# Sentiment-Analysis-Naive-Bayes-and-telebot
Analyses [sentiment](https://en.wikipedia.org/wiki/Sentiment_analysis) of a sentense(0 - negative, 1 - neutral, 2 - positive in this case, for now). 
Uses [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) to classify sentense. 
Natural language is the language used for human communication, in contrast to formal languages, it was not created specifically.
The analysis of the tonality of the text can be considered as a classification task. 
I have used naive Bayesian classifier to solve the classification problem, but it can be done by another methods.
For training was used dataset from [tweeteval](https://github.com/cardiffnlp/tweeteval/tree/main/datasets/sentiment). Telegram bot is used to communicate with user.


1. Install Python 3.9.
2. Install Python libraries: sklearn, NLTK, pickle, telebot, NumPy. If you are using pip, open command prompt and type "pip3 install <library name">
3. Run nltk_download.py to download dependencies
4. Run SA_model_train.py to train model on included train data.
5. Write to https://t.me/BotFather, create bot and get token for him.
6. Create file boa_api.txt. Write api in it. E.g.:
   ![image](https://github.com/Stanislavius/Sentiment-Analysis-Naive-Bayes-and-Telebot/assets/56927835/62ff7318-39ca-4cce-96d4-c3d701df6342
7. Run SA_bot.py
8. Open chat with your bot, type /classity and write your message to classify it's sentiment or send a text document where each line is a message.

/classify example:

![image](https://github.com/Stanislavius/Sentiment-Analysis-Naive-Bayes-and-telebot-/assets/56927835/62b91d4b-937c-4e65-a0cb-57874df85d91)

Send document example: 

![image](https://github.com/Stanislavius/Sentiment-Analysis-Naive-Bayes-and-Telebot/assets/56927835/93a6c649-b9de-4aec-a5cc-d157f2de9518) 

Test input document:

![image](https://github.com/Stanislavius/Sentiment-Analysis-Naive-Bayes-and-Telebot/assets/56927835/cc182637-6688-4002-b00c-8f7a41afef75)

Test output document:

![image](https://github.com/Stanislavius/Sentiment-Analysis-Naive-Bayes-and-Telebot/assets/56927835/398500b6-401e-4b7e-92cf-f1f4609e3137)



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
