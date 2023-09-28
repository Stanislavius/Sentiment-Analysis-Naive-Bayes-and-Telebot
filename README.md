# Sentiment-Analysis-Naive-Bayes-and-telebot
Analyses [sentiment](https://en.wikipedia.org/wiki/Sentiment_analysis) of a sentense(0 - negative, 1 - neutral, 2 - positive in this case, for now). 
Uses [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) to classify sentense. 
Natural language is the language used for human communication, in contrast to formal languages, it was not created specifically.
The analysis of the tonality of the text can be considered as a classification task. 
I have used naive Bayesian classifier to solve the classification problem, but it can be done by another methods.
For training was used dataset from [here](https://github.com/cardiffnlp/tweeteval/tree/main/datasets/sentiment). Telegram bot is used to communicate with user.


1. Install Python 3.9.
2. Install Python libraries: sklearn, NLTK, pickle, telebot, NumPy. If you are using pip, open command prompt and type "pip3 install <library name">
3. Run nltk_download.py to download dependencies
4. Run SA_model_train.py to train model on included train data.
5. Write to https://t.me/BotFather, create bot and get token for him.
6. Change api in SA_bot.py to be your api recieved from BotFather. For example, api = "dgfdgregerergerge"
7. Run SA_bot.py
8. Open chat with your bot, type /classity and write your message to classify it's sentiment.

![image](https://github.com/Stanislavius/Sentiment-Analysis-Naive-Bayes-and-telebot-/assets/56927835/62b91d4b-937c-4e65-a0cb-57874df85d91)

Files and their main function:
1. Data folder - data for training.
2. SA_bot.py - telegram bot able to recieve command, pass text to classifier and send back the result.
3. SA_classification.py - usage of trained model to classificate input.
4. SA_model_train.py - training of model.
5. model.data - saved trained model.
6. nltk_download.py - to download dependencies inside NLTK.
7. words.data - words which were used to train and their positions.
8. words_proc.py - some functions to process text data.
