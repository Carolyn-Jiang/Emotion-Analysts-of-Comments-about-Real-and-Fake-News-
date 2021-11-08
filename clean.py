import re,string,unicodedata
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

class clean:

    def __init__(self,text):
        self.text = text
        
    def strip_html(self,text):
        soup = BeautifulSoup(self.text, "html.parser")
        return soup.get_text()
    
    # Removing the square brackets
    def remove_betweenn_square_brackets(self, text):
        return re.sub('\[[^]]*\]', '', self.text)
    
    # Removing URL's
    def remove_between_square_brackets(self, text):
        return re.sub(r'http\S+', '', self.text)
    
    # Removing the stopwords from text
    def remove_stopwords(self, text):
        final_text = []
        text = self.text
        text = text.lower()
        stop = set(stopwords.words('english'))
        # punctuation = list(string.punctuation)
        # stop.update(punctuation)
        stop.update(['make','toward','’s'])
        for i in text.split():
            if i.strip().lower() not in stop:
                final_text.append(i.strip())        
        return " ".join(final_text)
    
    def lemma_pinct(self, text):
        text = self.text
        punctuation = list(string.punctuation)
        punctuation.extend(['“','”','’','...'])
        text = word_tokenize(text)
        nlp = spacy.load('en_core_web_sm')
        texts = []
        for token in text:
            doc = nlp(" ".join(token))
            texts.append([token.lemma_ for token in doc])
        result = []
        for i in text:
            result.append("".join(i))
        final = []
        for word in result:
            if word not in punctuation:
                final.append(word)
    
        return ' '.join(final)
        #return " ".join(texts)
    
    # Removing the noisy text
    def denoise_text(self, text):
        text = self.strip_html(self.text)
        text = self.remove_between_square_brackets(text)
        text = self.lemma_pinct(text)
        text = self.remove_stopwords(text)
        return text
