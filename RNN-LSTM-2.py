# This model has some issue

import os
import clean
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
import re,string,unicodedata
from keras.preprocessing import text, sequence
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Dropout
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf

# data pre-processing(only run at the first time)
'''
True_news = pd.read_csv('/Users/yuechenjiang/Desktop/project660/True.csv') # change path
Fake_news = pd.read_csv('/Users/yuechenjiang/Desktop/project660/Fake.csv') # change path
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

class RNN(keras.Model):
    def __init__(self, units, num_classes, num_layers):
        super(RNN, self).__init__()
        
        self.rnn = keras.layers.LSTM(units,return_sequences = True)
        self.rnn2 = keras.layers.LSTM(units)
        
        # have 1000 words totally, every word will be embedding into 100 length vector
        # the max sentence lenght is 80 words
        self.embedding = keras.layers.Embedding(top_words, 100, input_length=max_review_length)
        self.fc = keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        
        x = self.embedding(inputs)

        x = self.rnn(x) 
        x = self.rnn2(x) 

        x = self.fc(x)
        print(x.shape)

        return x

from tensorflow import keras
from tensorflow.keras import layers
if __name__ == '__main__':
    units = 64
    num_classes = 2
    batch_size = 32
    epochs = 10
    df = pd.read_csv('/Users/yuechenjiang/Desktop/project660/code-data/combined.csv')
    # change path

    # load the dataset but only keep the top n words, zero the rest
    top_words = 10000
    # truncate and pad input sequences
    max_review_length = 80
    x_train,x_test,y_train,y_test = train_test_split(df.text,df.category,random_state = 0)

    tokenizer = text.Tokenizer(num_words=top_words)
    tokenizer.fit_on_texts(x_train)
    tokenized_train = tokenizer.texts_to_sequences(x_train)
    x_train = sequence.pad_sequences(tokenized_train, max_review_length)

    tokenized_test = tokenizer.texts_to_sequences(x_test)
    x_test = sequence.pad_sequences(tokenized_test, max_review_length)

    # x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_length)
    # x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_length)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    model = RNN(units, num_classes, num_layers=2)


    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # print(model.summary())
    
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)
    # train
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test), verbose=1, callbacks = [learning_rate_reduction])

    # evaluate on test set
    scores = model.evaluate(x_test, y_test, batch_size, verbose=1)
    print("Final test loss and accuracy :", scores)

model.summary()

# Model Analysis
print("Accuracy of the model on Training Data is - " , model.evaluate(x_train,y_train)[1]*100 , "%")
print("Accuracy of the model on Testing Data is - " , model.evaluate(x_test,y_test)[1]*100 , "%")

# visualiz training and testing accuracy and loss
epochs = [i for i in range(10)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

fig.set_size_inches(20,10)

ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc , 'ro-' , label = 'Testing Accuracy')
ax[0].set_title('Training & Testing Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'go-' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'ro-' , label = 'Testing Loss')
ax[1].set_title('Training & Testing Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
plt.show()

pred = model.predict(x_test)
pred[:5]

pred = list(pred)
test_result = []
for i in range(len(pred)):
    test_result.append(pred[i][0])

confusion = pd.DataFrame({'Pred':test_result, 'Truth':list(y_test)})
confusion['binary_pred'] = (confusion['Pred'] > 0.5).astype(int)

cm_DL = confusion_matrix(confusion['Truth'],confusion['binary_pred'])

plt.figure(figsize = (10,8))
sns.heatmap(cm_DL,cmap= "Blues", linecolor = 'black' , xticklabels = ['Fake','Real'] , yticklabels = ['Fake','Real'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
