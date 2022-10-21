import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

nltk.download("stopwords")
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

#print(stop_words)

#<-------------training model--------------->

df = pd.read_csv('dataset/train.txt',sep = ';',header = None)
df.columns = ['statement','emotion']
X = np.array(df[['statement']]).reshape(-1)
y = np.array(df[['emotion']]).reshape(-1)

emotions = np.unique(y)

X_token = [word_tokenize(i) for i in X]

#print(X.shape)

for sentences in X_token:
    for word in list(sentences):
        if word in stop_words:
            sentences.remove(word)

#print(X_token)

stemmer = PorterStemmer()
X_pretrain = []
for sentences in X_token:
    X_pretrain.append([stemmer.stem(word) for word in sentences])

#print(X_train)

#X_train = np.array(X_train).reshape(-1,1)

#print(emotions)

X_train = []
for sentences in X_pretrain:
    X_train.append(nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize([(word) for word in sentences]))

#print(X_train.shape)

#for i in range(len(X)):
    #print(f'{X[i]} \n')
#    print(f'{X_train[i]} -> {y[i]}')
#    print('\n')

pipe_lr = Pipeline(steps = [
    ('cv',CountVectorizer()),
    ('lr',LogisticRegression(solver = 'saga',max_iter = 1000))
])

#print(y.shape)

pipe_lr.fit(X_train,y)

#<-----------------testing model----------------->

df = pd.read_csv('dataset/test.txt',sep = ';',header = None)
df.columns = ['statement','emotion']
X_pre = np.array(df[['statement']]).reshape(-1)
y_testpre = np.array(df[['emotion']]).reshape(-1)

#print(X[2])

emotions = np.unique(y)

X_test_token = [word_tokenize(i) for i in X_pre]

#print(X.shape)

for sentences in X_test_token:
    for word in list(sentences):
        if word in stop_words:
            sentences.remove(word)

#print(X_token)

stemmer = PorterStemmer()
X_pretest = []
for sentences in X_test_token:
    X_pretest.append([stemmer.stem(word) for word in sentences])

#print(X_train)

#X_train = np.array(X_train).reshape(-1,1)

#print(emotions)

X_test = []
for sentences in X_pretest:
    X_test.append(nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize([(word) for word in sentences]))

y_test = pipe_lr.predict(X_test)

print(accuracy_score(y_test,y_testpre))

#accuracy = 0.8575

#<----------------manual testing---------------->
"""
sample1 = 'what do you mean ghosts are scary'

#token > stop > stem

y_token = word_tokenize(sample1)
for words in y_token:
    if words in stop_words:
        y_token.remove(words)
stemmer1 = PorterStemmer()
y_pretest = [stemmer.stem(word) for word in y_token]
y_test = nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize([(word) for word in y_pretest])
#print(y_test)
print(sample1)
print(pipe_lr.predict([y_test]))
"""