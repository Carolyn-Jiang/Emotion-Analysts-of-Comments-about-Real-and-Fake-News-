import nltk
import clean
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re,string,unicodedata
from collections import Counter
from wordcloud import WordCloud,STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer

# data preprocessing
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

if __name__ == "__main__":
    # load the real news data and preview
    True_news = pd.read_csv('/Users/yuechenjiang/Desktop/project660/True.csv')
    print('===================== Real Data Preview =====================')
    print(True_news.head())
    print('===================== Real Data Describe ====================')
    True_news.describe()
    print('Real Dataset Shape:','\n',True_news.shape)
    print('Real Columns name','\n',True_news.columns)
    print('Real Subject count','\n',True_news['subject'].value_counts())
    
    # load the fake news data and preview
    Fake_news = pd.read_csv('/Users/yuechenjiang/Desktop/project660/Fake.csv')
    print('===================== Fake Data Preview =====================')
    print(Fake_news.head())
    print('===================== Fake Data Describe ====================')
    Fake_news.describe()
    print('Fake Dataset Shape:','\n',Fake_news.shape)
    print('Fake Columns name','\n',Fake_news.columns)
    print('Fake Subject count','\n',Fake_news['subject'].value_counts())
    
    # combine the datasets and data visualization
    True_news['category'] = 1
    Fake_news['category'] = 0
    df = pd.concat([True_news,Fake_news])
    print('========== Comparie the Number of Real and Fake News =========')
    sns.set_style("darkgrid")
    sns.countplot(df.category)
    print('Check whether the data set has null values','\n',df.isna().sum())
    print('Check news subjects','\n',df.subject.value_counts())
    
    plt.figure(figsize = (12,8))
    plt.title('Real and Fake News subjects')
    sns.set(style = "whitegrid",font_scale = 1.2)
    chart = sns.countplot(x = "subject", hue = "category" , data = df)
    chart.set_xticklabels(chart.get_xticklabels(),rotation=90)

    df['text'] = df['text'] + " " + df['title']
    del df['title']
    del df['subject']
    del df['date']
    
    print('============= Real News World Cloud Before Cleaning =============')
    plt.figure(figsize = (20,20)) # Text that is not Fake
    wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(df[df.category == 1].text))
    plt.imshow(wc , interpolation = 'bilinear')
    print('============= Fake News World Cloud Before Cleaning =============')
    plt.figure(figsize = (20,20)) # Text that is Fake
    wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(df[df.category == 0].text))
    plt.imshow(wc , interpolation = 'bilinear')
    
    clean_text = []
    for i in df['text']:
        data_cleaning = clean.clean(text = i)
        clean_text.append(data_cleaning.denoise_text(i))
    del df['text']
    df = pd.DataFrame({'text':clean_text,'category':df['category']})
    # df.head()
    # data_cleaning = clean()
    # df['text'] = df['text'].apply(data_cleaning.denoise_text)
    # WHEN I USE CLASS I'M UNABLE TO USE 'apply' FUNCTION, STILL NEED TO FIND OUT WHY, USING LOOPS ARE SLOW
    print('============== Real News World Cloud After Cleaning =============')
    plt.figure(figsize = (20,20)) # Text that is not Fake
    wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(df[df.category == 1].text))
    plt.imshow(wc , interpolation = 'bilinear')
    print('============== Fake News World Cloud After Cleaning =============')
    plt.figure(figsize = (20,20)) # Text that is Fake
    wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(df[df.category == 0].text))
    plt.imshow(wc , interpolation = 'bilinear')
    # Number of characters in texts
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,8))
    text_len=df[df['category']==1]['text'].str.len()
    ax1.hist(text_len,color='red')
    ax1.set_title('Real text')
    text_len=df[df['category']==0]['text'].str.len()
    ax2.hist(text_len,color='green')
    ax2.set_title('Fake text')
    fig.suptitle('Characters in texts')
    plt.show()
    print('The distribution of both seems to be a bit different.','\n', 
          '2500 characters in text is the most common in original text category', '\n', 
          'while around 5000 characters in text are most common in fake text category.')
    
    # Number of words in each text
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,8))
    text_len=df[df['category']==1]['text'].str.split().map(lambda x: len(x))
    ax1.hist(text_len,color='red')
    ax1.set_title('Real text')
    text_len=df[df['category']==0]['text'].str.split().map(lambda x: len(x))
    ax2.hist(text_len,color='green')
    ax2.set_title('Fake text')
    fig.suptitle('Words in texts')
    plt.show()
    
    # Average word length in a text
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(20,10))
    word=df[df['category']==1]['text'].str.split().apply(lambda x : [len(i) for i in x])
    sns.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='red')
    ax1.set_title('Real text')
    word=df[df['category']==0]['text'].str.split().apply(lambda x : [len(i) for i in x])
    sns.distplot(word.map(lambda x: np.mean(x)),ax=ax2,color='green')
    ax2.set_title('Fake text')
    fig.suptitle('Average word length in each text')
    
    corpus = get_corpus(df.text)
    print('Top 5 Words','\n',corpus[:5])
    
    counter = Counter(corpus)
    most_common = counter.most_common(10)
    most_common = dict(most_common)
    print('Numbers of most common words','\n',most_common)
    
    # Unigram Analysis
    plt.figure(figsize = (16,9))
    most_common_uni = get_top_text_ngrams(df.text,10,1)
    most_common_uni = dict(most_common_uni)
    sns.barplot(x=list(most_common_uni.values()),y=list(most_common_uni.keys()))
    
    # Bigram Analysis
    plt.figure(figsize = (16,9))
    most_common_bi = get_top_text_ngrams(df.text,10,2)
    most_common_bi = dict(most_common_bi)
    sns.barplot(x=list(most_common_bi.values()),y=list(most_common_bi.keys()))
    
    # Trigram Analysis
    plt.figure(figsize = (16,9))
    most_common_tri = get_top_text_ngrams(df.text,10,3)
    most_common_tri = dict(most_common_tri)
    sns.barplot(x=list(most_common_tri.values()),y=list(most_common_tri.keys()))