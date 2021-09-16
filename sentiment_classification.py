import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv('dataset.csv')
cv = CountVectorizer()
words = cv.fit_transform(data.tweet.values.astype(str))
print("Total no. of tweets and words in the dataset "+str(words.shape))
training_corpus = []
for i in range(0, 1500):
  revised = re.sub('[^a-zA-Z]',' ',str(data['tweet'][i]))
  revised = revised.lower()
  revised = revised.split()
  revised = word_tokenize(str(revised))
  ps = PorterStemmer()
  revised = [ps.stem(word) for word in revised if not word in set(stopwords.words('english'))]
  revised = ' '.join(revised)
  training_corpus.append(revised)
cv = CountVectorizer()
x = cv.fit_transform(training_corpus).toarray()
y = data['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 0)
clf = MultinomialNB()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("Training Accuracy :", clf.score(x_train, y_train))
print("Validation Accuracy :", clf.score(x_test, y_test))
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
print("Training Accuracy :", clf.score(x_train, y_train))
print("Validation Accuracy :", clf.score(x_test, y_test))
forest = RandomForestClassifier()
forest.fit(x_train, y_train)
y_pred = forest.predict(x_test)
print("Training Accuracy :", forest.score(x_train, y_train))
print("Validation Accuracy :", forest.score(x_test, y_test))

tweet = '#racist #hate in america for whites africa for africans '
tweet = re.sub('[^a-zA-Z]',' ',tweet)
tweet = tweet.lower()
tweet = tweet.split()
tweet = word_tokenize(str(tweet))
ps = PorterStemmer()
tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
tweet = ' '.join(tweet)
tweet = [tweet]
tweet = cv.transform(tweet).toarray()
y_pred = forest.predict(tweet)
print(y_pred)

