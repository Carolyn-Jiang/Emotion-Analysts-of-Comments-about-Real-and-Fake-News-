Emotion Analysts of Comments about Real and Fake News.

1. Introduction

Fake news refers to misinformation, disinformation, or malformation, which is spread through word of mouth and traditional media and more recentlythrough digital forms of communication such as edited videos, memes, unverified advertisements, and social media propagated rumors. Fake news spread through social media has become a serious social problem. In this paper, we want to study the emotional impact of fake news on people through comments under true and fake news. Identify the true and fake news by establishing a model.


2. Objectives and expected contributions

We hereby posit one research question, formalized in the following, and by leveraging the Fake and real news dataset and real news comments from internet to answer these questions: \
• Research Questions: How much influence can news have on people’s thoughts? And how much negative social impact can fake news cause? \
We expect this work can make the following contributions: \
• We provide a machine learning approach to identify the news and help people get the correct information from the internet. \
• Analyze the emotional impact of fake news on people, and formulate psychological interventions of misleading information to prevent it with the potential of it resulting in mob violence, suicides, etc., due to misinformation circulating on social media. 


3. Data Source: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset


4. Method: Compared the performance of eight machine learning models (Logistic Regression, Random Forest, Decision Tree, Multilayer Perceptron, Adaptive Gradient Boost, SVM) in Kaggle dataset named “Fake and real news”. Used K-fold cross-validation to choose the best models with accuracy over 99.5%.  Built an LSTM model using GloVe embedding with accuracy around 95% (After 30 Epoch). Scraped news from AP-Fact-Check. Tested the model efficiency using data outside the training and made corresponding adjustments to the model. Scraped news from news websites and using existing models to determine whether the news is real or fake. Get the comments corresponding to each news item from the Twitter account of the news website and label the comments. Use labeled comments dataset for emotion analysts to find out if people feel different emotions towards the real news compared to fake news. 


5. File description 

clean.py: this file contain a class "clean" used in further data pre-processing. \
data_pre-processing.py: data process visualization and Unigran, Bigram, Trigram analysis. \
ML-method.py: build 8 machine learning method and resulit visualization. \
LSTM.py: build LSTM model and result visualization. \
RNN.py: another model(rnn.LSTM) with accuracy 99.4% \
Fake-Real-News-Detective-Model.ipynb: Partial result display.



To be continued \
Update October 10, 2021
