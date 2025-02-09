from services.imageprocessor import ImageProcessor as image_processor
from services.textqueryprocessor import TextQueryProcessor as text_processor
from pymongo import MongoClient
import gridfs
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

client = MongoClient('mongodb://localhost:27017/')
db = client['image_features_db']
fs = gridfs.GridFS(db)

query = "A dog and a cat."
query = query.lower()

query = text_processor.remove_punctuation(query)
print("Query without punctuation:", query)

query = text_processor.remove_whitespace(query)
print("Query without extra whitespace:", query)

query = text_processor.get_tokenized_list(query)
print("Tokenized query:", query)

query = text_processor.remove_stopwords(query)
print("Query without stopwords:", query)

query = text_processor.word_stemmer(query)
print("Stemmed query:", query)

q = ' '.join(query)
vector_query = vectorizer.fit_transform([q])
print("Vectorized query:", vector_query.toarray())