import os
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import string
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

# Download necessary NLTK data (if not already downloaded)
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('punkt_tab')

vectorizer = TfidfVectorizer()

class TextQueryProcessor:
    """
    A class used to process text queries by performing various preprocessing steps such as removing punctuation,
    removing whitespace, removing stopwords, tokenizing, and stemming.
    Methods
    -------
    remove_punctuation(text)
        Removes punctuation from the given text.
    remove_whitespace(text)
        Removes extra whitespace from the given text.
    remove_stopwords(text)
        Removes stopwords from the given text.
    get_tokenized_list(doc)
        Tokenizes the given document into a list of words.
    word_stemmer(token_list)
        Applies stemming to the given list of tokens.
    preprocessing_query(query)
        Performs a series of preprocessing steps on the given query and returns a TF-IDF vector.
    """
    def remove_punctuation(text):
        try:
            return text.translate(str.maketrans('', '', string.punctuation))
        except Exception as e:
            print(f"Error in remove_punctuation: {e}")
            return text
    
    def remove_whitespace(text):
        try:
            return " ".join(text.split())
        except Exception as e:
            print(f"Error in remove_whitespace: {e}")
            return text
    
    def remove_stopwords(text):
        try:
            words_to_remove = stopwords.words('english')
            cleaned_doc = []
            for word in text:
                if word not in words_to_remove:
                    cleaned_doc.append(word)
            return cleaned_doc
        except Exception as e:
            print(f"Error in remove_stopwords: {e}")
            return text
    
    def get_tokenized_list(doc):
        try:
            return nltk.word_tokenize(doc)
        except Exception as e:
            print(f"Error in get_tokenized_list: {e}")
            return []
    
    def word_stemmer(token_list):
        try:
            stemmer = nltk.stem.PorterStemmer()
            stemmed = []
            for words in token_list:
                stemmed.append(stemmer.stem(words))
            return stemmed
        except Exception as e:
            print(f"Error in word_stemmer: {e}")
            return token_list

    def preprocessing_query(query):
        try:
            query = query.lower()
            query = TextQueryProcessor.remove_punctuation(query)
            query = TextQueryProcessor.remove_whitespace(query)
            query = TextQueryProcessor.get_tokenized_list(query)
            query = TextQueryProcessor.remove_stopwords(query)
            query = TextQueryProcessor.word_stemmer(query)
            # q = []
            # for word in query:
            #     q.append(word)
            # q = ' '.join(q)
            # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            # text_vector = model.encode([q])
            # vectorizer = TfidfVectorizer()
            # text = vectorizer.fit_transform(query)
            return query
        except Exception as e:
            print(f"Error in preprocessing_query: {e}")
            return None