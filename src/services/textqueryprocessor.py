import os
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

vectorizer = TfidfVectorizer()

class TextQueryProcessor:
    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def remove_whitespace(text):
        return " ".join(text.split())
    
    def remove_stopwords(text):
        words_to_remove = stopwords.words('english')
        cleaned_doc = []
        for word in text:
            if word not in words_to_remove:
                cleaned_doc.append(word)
        return cleaned_doc
    
    def get_tokenized_list(doc):
        return nltk.word_tokenize(doc)
    
    def word_stemmer(token_list):
        stemmer = nltk.stem.PorterStemmer()
        stemmed = []
        for words in token_list:
            stemmed.append(stemmer.stem(words))
        return stemmed

    def preprocessing_query(query):
        query = query.lower()
        query = TextQueryProcessor.remove_punctuation(query)
        query = TextQueryProcessor.remove_whitespace(query)
        query = TextQueryProcessor.get_tokenized_list(query)
        query = TextQueryProcessor.remove_stopwords(query)
        q = []
        for word in word_stemmer(query):
            q.append(word)
        q = ' '.join(q)
        vector_query = vectorizer.fit_transform([q])
        return vector_query
    
    