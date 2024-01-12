import re

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from textblob import Word

import numpy as np

import matplotlib as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

import pickle 

import seaborn 


# importing wordnet
import nltk
nltk.download('wordnet')

# clean text method
def clean_str(string):
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"\'t", "", string) 
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'lls", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", "", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"'", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[0-9]\w+|[0-9]", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower() 
    #lower() is used to convert everything to lowercase as it is requirement for our algorithm(tf-idf)
    
#Our dataset to make our model learn classification
data = pd.read_csv("class.csv")
print(data.head())

#Knowing density of different classes in the dataset
print(data.groupby('type').count())

#Extracting colums of dataset as list for further operation
x = data['news'].tolist()
y = data['type'].tolist()
print("Processing data...")
for index, value in enumerate(x):
    x[index] = ' '.join([Word(word).lemmatize() for word in clean_str(value).split()])

#Used tfidf  for vectrization as it gives importance to context besides frequency of words in document
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
#Feel free to tune parameters, 
features = tfidf.fit_transform(data.news).toarray()
labels = data.type
print("features shape\n",features.shape)



#Performance measures of different machine learning models

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_val_score


models = [
    RandomForestClassifier(n_estimators=500, max_depth=3, random_state=0),
    MultinomialNB(),
    LogisticRegression(random_state=21),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

performance=cv_df.groupby('model_name').accuracy.mean()
print(performance)


#Since we have got maximum accuracy by Logistic Regression we would proceed with it.
from sklearn.model_selection import train_test_split

model = LogisticRegression(random_state=0)

X_train, X_test, y_train, y_test, val_train, val_test = train_test_split(features, labels, data.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)

model.fit(features, labels)

#pickling our model
import pickle
with open('trained_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
    
with open('tfidfd.pkl', 'wb') as tfidf_file:
    pickle.dump(tfidf, tfidf_file)



























