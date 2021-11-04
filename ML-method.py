import clean
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# data pre-processing
'''
# This part only run on the first time
True_news = pd.read_csv('/Users/yuechenjiang/Desktop/project660/code-data/True.csv')
Fake_news = pd.read_csv('/Users/yuechenjiang/Desktop/project660/code-data/Fake.csv')
True_news['category'] = 1
Fake_news['category'] = 0
df = pd.concat([True_news,Fake_news])
df['text'] = df['text'] + " " + df['title']
del df['title']
del df['subject']
del df['date']
clean_text = []
for i in df['text']:
    data_cleaning = clean.clean(text = i)
    clean_text.append(data_cleaning.denoise_text(i))
del df['text']
df = pd.DataFrame({'text':clean_text,'category':df['category']})
df.to_csv("combined.csv",index=False)
'''

def get_corpus(text):
    words = []
    for i in text:
        for j in i.split():
            words.append(j.strip())
    return words

def get_top_text_ngrams(corpus, n, g):
    vec = CountVectorizer(ngram_range=(g, g)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

df = pd.read_csv('combined.csv')
X = df['text']
Y = df['category']
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y , train_size=0.8, shuffle = True, test_size=0.2, random_state=1)
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=15000, use_idf=True, stop_words='english')
x_train = tfidf_vectorizer.fit_transform(x_train)
x_test = tfidf_vectorizer.fit_transform(x_test)

print('train:',x_train.shape,'\n','test:',x_test.shape)

# model setup
models = [LogisticRegression(solver='lbfgs'),        # Logistic regression
          RandomForestClassifier(n_estimators=100),  # Random forest
          DecisionTreeClassifier(),                  # Decision tree
          MLPClassifier(max_iter=100),               # Multilayer perceptron
          AdaBoostClassifier(),                      # Adaptive gradient boost
          BaggingClassifier(),                       # Bagging algorithm
          GradientBoostingClassifier(),              # Gradient Boosting Algorithm
          SVC(kernel = 'linear')]
          # GaussianNB()]            

model_name = ['LogisticRegression',
              'RandomForestClassifier',
              'DecisionTreeClassifier',
              'MLPClassifier',
              'AdaBoostClassifier',
              'BaggingClassifier',
              'GradientBoostingClassifier',
              'SVMClassifier',
              'NaiveBayesClassifier']

acc = []
cms = []
for model in models:
    model.fit(x_train,y_train)
    # model_acc = model.score(x_test, y_test)*100
    acc.append(model.score(x_test, y_test))
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test,y_p
                          red)
    cms.append(cm)
    print(model,'\n',cm)
    
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train.toarray(),y_train)
acc.append(model.score(x_test.toarray(), y_test))
y_pred = model.predict(x_test.toarray())
cm = confusion_matrix(y_test,y_pred)
cms.append(cm)
print('GaussianNB()\n',cm)

fig,ax=plt.subplots(3,3,figsize=(20,18))
for i in range(len(cms)):
    plt.subplot(3, 3, i+1)
    sns.heatmap(np.array(cms[i]),cmap= "Blues", linecolor = 'black' , xticklabels = ['Fake','Real'] , yticklabels = ['Fake','Real'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(model_name[i])

a = pd.DataFrame({"name": model_name, "acc": acc})
a

#####################################################################################
#                                                                                   #
#                                    Test Part                                      #
#                                                                                   #
#####################################################################################

df.to_csv("text2.csv",index=False)

x_test2 = df['text']
y_test2 = df['label']

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=15000, use_idf=True, stop_words='english')
x_test2 = tfidf_vectorizer.fit_transform(x_test2)

acc = []
cms = []
for model in models:
    model.fit(x_train,y_train)
    # model_acc = model.score(x_test, y_test)*100
    acc.append(model.score(x_test2, y_test2))
    y_pred = model.predict(x_test2)
    cm = confusion_matrix(y_test2,y_pred)
    cms.append(cm)
    print(model,'\n',cm)
    
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train.toarray(),y_train)
acc.append(model.score(x_test2.toarray(), y_test2))
y_pred = model.predict(x_test2.toarray())
cm = confusion_matrix(y_test2,y_pred)
cms.append(cm)
print('GaussianNB()\n',cm)

fig,ax=plt.subplots(3,3,figsize=(20,18))
for i in range(len(cms)):
    plt.subplot(3, 3, i+1)
    sns.heatmap(np.array(cms[i]),cmap= "Blues", linecolor = 'black' , xticklabels = ['Fake','Real'] , yticklabels = ['Fake','Real'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(model_name[i])

a = pd.DataFrame({"name": model_name, "acc": acc})
a