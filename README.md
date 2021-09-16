# Sentiment-Analysis-of-Tweets-using-Naive-Bayes-Random-Forest-Classifiers---Python
Project on sentiment analysis of Twitter data to help detect tweets with hate speech by applying ML models like Gaussian Naive Bayes, Multinomial Naive Bayes and Random Forests with Python.

Objective

The project aims to detect hate speech in Twitter data. The task is to classify racist/sexist tweets from a given set of tweets. In the training dataset, tweets with hate speech are labeled '1' and all other tweets as '0'. There are 1500 hand-classified tweets in the training dataset which will be fed to the ML model for training. 

Preprocessing & Cleaning

Preprocessing the text data is an important step in Natural Language Processing as it makes the raw text ready for data extraction and mining. In the Python script, we first clean the data by removing all punctuation, special characters, numbers, and whitespaces using Regular Expression module. We then remove all the stopwords from the tweets for creating better feature vectors. Doing all this is essential to remove the noise and inconsistency from the data. This can help in achieving a better model accuracy .
 
Classification & Analysis
 
We build the vocabulary for the training corpus using CountVectorizer and extract the labels from the dataset. Our text is now ready for classification in the form of feature vectors. Using different classifiers like Gaussian Naive Bayes, Multinomial Naive Bayes and Random Forests, accuracy is tested and analysed. It is observed that with smaller test dataset sizes and large training dataset, Multinomial and Gaussian Naive Bayes appear to perform better than Random Forest but an increase in size of test dataset leads to better accuracy of Random Forest Classifier. Finally, a random tweet containing hate speech is tested with the model. The classifier correctly returns the label '1' showing the model is correctly classifying the tweets. 
 
The Python script is written and tested in Spyder IDE which can be downloaded from the official Anaconda website.
