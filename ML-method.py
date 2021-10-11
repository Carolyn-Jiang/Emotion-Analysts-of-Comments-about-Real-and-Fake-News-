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
True_news = pd.read_csv('/Users/yuechenjiang/Desktop/project660/True.csv')
Fake_news = pd.read_csv('/Users/yuechenjiang/Desktop/project660/Fake.csv')
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

# model setup
models = [LogisticRegression(solver='lbfgs'),        # Logistic regression
          RandomForestClassifier(n_estimators=100),  # Random forest
          DecisionTreeClassifier(),                  # Decision tree
          MLPClassifier(max_iter=100),               # Multilayer perceptron
          AdaBoostClassifier(),                      # Adaptive gradient boost
          BaggingClassifier(),                       # Bagging algorithm
          GradientBoostingClassifier(),              # Gradient Boosting Algorithm
          SVC(kernel = 'linear')]            

model_name = ['LogisticRegression',
              'RandomForestClassifier',
              'DecisionTreeClassifier',
              'MLPClassifier',
              'AdaBoostClassifier',
              'BaggingClassifier',
              'GradientBoostingClassifier',
              'SVMClassifier']

tfidf_vectorizer = TfidfVectorizer(use_idf=True, stop_words='english')
X = tfidf_vectorizer.fit_transform(df['text'])
Y = df['category']
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y , train_size=0.8,test_size=0.2, random_state=1)

acc = []
cms = []
for model in models:
    model.fit(x_train,y_train)
    # model_acc = model.score(x_test, y_test)*100
    acc.append(model.score(x_test, y_test))
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test,y_pred)
    cms.append(cm)
    print(model,'\n',cm)

fig,ax=plt.subplots(2,4,figsize=(20,8))
for i in range(len(cms)):
    plt.subplot(2, 4, i+1)
    sns.heatmap(np.array(cms[i]),cmap= "Blues", linecolor = 'black' , xticklabels = ['Fake','Real'] , yticklabels = ['Fake','Real'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(model_name[i])

a = pd.DataFrame({"name": model_name, "acc": acc})
a