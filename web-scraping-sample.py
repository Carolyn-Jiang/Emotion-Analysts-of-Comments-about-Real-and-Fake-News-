import numpy as np
import pandas as pd
from selenium import webdriver
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
from nltk.tokenize import word_tokenize,sent_tokenize
from keras.preprocessing import text, sequence

'''
This is a list of fake news websites from wikipedia. Most of the urls are blocked
https://en.wikipedia.org/wiki/List_of_fake_news_websites
Following urls are still can be visit
https://www.globalresearch.ca/
12+6
https://nationalreport.net/
//*[@id="home-featured"]/h2/a
//*[@id="category-box1-1"]/div/ul/li[1]/h2/a
//*[@id="category-box1-1"]/div/ul/li[2]/h2/a
//*[@id="column2"]/li[1]/h2/a
//*[@id="column2"]/li[2]/h2/a
8*35
https://now8news.com/
//*[@id="main-content"]/article/h3/a
//*[@id="main-content"]/div[2]/article[1]/h3/a
//*[@id="main-content"]/div[2]/article[2]/h3/a
9*132
https://oneworld.press/
/html/body/div[2]/div[3]/div[1]/div/div[1]/div[1]/div/a
/html/body/div[2]/div[3]/div[1]/div/div[1]/div[2]/div/a
/html/body/div[2]/div[3]/div[1]/div/div[1]/div[3]/div/a
4
https://worldnewsdailyreport.com/
//*[@id="mh_newsdesk_posts_large-2"]/div/article/h3/a
//*[@id="mh_newsdesk_custom_posts-11"]/div/article/h3/a
//*[@id="mh_newsdesk_custom_posts-25"]/div/article/h3/a
//*[@id="internal_trc_39066"]/div[1]/a[1]
//*[@id="internal_trc_39066"]/div[2]/a[1]
'''

driver = webdriver.Chrome(executable_path='/Users/yuechenjiang/Desktop/project660/chromedriver')

news = []
page_url = 'https://www.globalresearch.ca/'
driver.get(page_url)
a = driver.find_elements_by_xpath('//*[@id="content"]/div[2]/div[1]/div[1]/div/strong/a')
for i in a:
    print(i)
    i.click()
    paragraph = []
    b = driver.find_elements_by_xpath('//*[@id="post-5758334"]/div[2]/p')
    for j in b:
        paragraph.append(j.text)
    str = ''
    c = str.join(paragraph)
# print('++++++++')
#n print(c)
news.append(c)
driver.get(page_url)
a = driver.find_elements_by_xpath('//*[@id="content"]/div[2]/div[2]/div[1]/div/strong/a')
for i in a:
    print(i)
    i.click()
    paragraph = []
    b = driver.find_elements_by_xpath('//*[@id="post-5758117"]/div[2]/p')
    for j in b:
        paragraph.append(j.text)
    str = ''
    c = str.join(paragraph)
# print('++++++++')
# print(c)
news.append(c)
driver.get(page_url)
a = driver.find_elements_by_xpath('//*[@id="content"]/div[2]/div[3]/div[1]/div/strong/a')
for i in a:
    print(i)
    i.click()
    paragraph = []
    b = driver.find_elements_by_xpath('//*[@id="post-5758110"]/div[2]/p')
    for j in b:
        paragraph.append(j.text)
    str = ''
    c = str.join(paragraph)
# print('++++++++')
# print(c)
news.append(c)
driver.get(page_url)
a = driver.find_elements_by_xpath('//*[@id="content"]/div[2]/div[4]/div[1]/div/strong/a')
for i in a:
    print(i)
    i.click()
    paragraph = []
    b = driver.find_elements_by_xpath('//*[@id="post-5757337"]/div[2]/p')
    for j in b:
        paragraph.append(j.text)
    str = ''
    c = str.join(paragraph)
# print('++++++++')
# print(c)
news.append(c)

driver.quit()

len(news)

Truth = [0,0,0,0]
test_fake = pd.DataFrame({'text':news,'category':Truth})
test_fake
# wordcloud before data cleaning
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(test_fake[test_fake.category == 0].text))
plt.imshow(wc , interpolation = 'bilinear')

clean_text = []
for i in test_fake['text']:
    data_cleaning = clean.clean(text = i)
    clean_text.append(data_cleaning.denoise_text(i))
del test_fake['text']
df = pd.DataFrame({'text':clean_text,'category':test_fake['category']})
df
# wordcloud after data cleaning
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(df[df.category == 0].text))
plt.imshow(wc , interpolation = 'bilinear')

# Machine Learning
tfidf_vectorizer = TfidfVectorizer(use_idf=True, stop_words='english')
X = tfidf_vectorizer.fit_transform(df['text'])
Y = df['category']

cms = []
for model in models:
	real_pred_ML = model.predict(X)
	confusion = pd.DataFrame({'Pred':real_pred_ML, 'Truth':list(Y)})
	confusion['binary_pred'] = (confusion['Pred'] > 0.5).astype(int)
	cm = confusion_matrix(confusion['Truth'],confusion['binary_pred'])
	cms.append(cm)
	print(model,'\n',cm)
fig,ax=plt.subplots(2,4,figsize=(25,12))
for i in range(len(cms)):
    plt.subplot(2, 4, i+1)
    sns.heatmap(np.array(cms[i]),cmap= "Blues", linecolor = 'black' , xticklabels = ['Fake','Real'] , yticklabels = ['Fake','Real'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(model_name[i])

# LSTM-1

df_test = pd.read_csv('fake_or_real_news.csv')

df_test['text'] = df_test['text'] + " " + df_test['title']

label = []
for i in df_test['label']:
    if i == 'FAKE':
        label.append(0)
    elif i == 'REAL':
        label.append(1)
    else:
        label.append2
        
df = pd.DataFrame({'text':df_test['text'],'label':label})

x_test2 = df['text']
y_test2 = df['label']

max_features = 10000
maxlen = 101

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(x_test2)
tokenized_test2 = tokenizer.texts_to_sequences(x_test2)
x_test2 = sequence.pad_sequences(tokenized_test2, maxlen=maxlen)

pred = model.predict(x_test2)
pred[:5]

pred = list(pred)
test_result = []
for i in range(len(pred)):
    test_result.append(pred[i][0])

confusion = pd.DataFrame({'Pred':test_result, 'Truth':list(y_test2)})
confusion['binary_pred'] = (confusion['Pred'] > 0.5).astype(int)

cm_DL = confusion_matrix(confusion['Truth'],confusion['binary_pred'])
cm_DL
'''
array([[ 277, 2887],
       [ 332, 2839]])
'''
print("Accuracy of the model on Other Testing Data is - " , model.evaluate(x_test2,y_test2)[1]*100 , "%")
'''
198/198 [==============================] - 4s 18ms/step - loss: 2.6734 - accuracy: 0.4919
Accuracy of the model on Other Testing Data is -  49.187055230140686 %
'''
plt.figure(figsize = (10,8))
sns.heatmap(cm_DL,cmap= "Blues", linecolor = 'black' , xticklabels = ['Fake','Real'] , yticklabels = ['Fake','Real'])
plt.xlabel("Predicted")
plt.ylabel("Actual")