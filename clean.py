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
        stop = set(stopwords.words('english'))
        punctuation = list(string.punctuation)
        stop.update(punctuation)
        for i in text.split():
            if i.strip().lower() not in stop:
                final_text.append(i.strip())        
        return " ".join(final_text)
    
    # Removing the noisy text
    def denoise_text(self, text):
        text = self.strip_html(self.text)
        text = self.remove_between_square_brackets(text)
        text = self.remove_stopwords(text)
        return text