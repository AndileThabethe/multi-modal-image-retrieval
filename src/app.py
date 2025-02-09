from services.imageprocessor import ImageProcessor as image_processor
from services.textqueryprocessor import TextQueryProcessor as text_processor
from pymongo import MongoClient
import gridfs
from sklearn.feature_extraction.text import TfidfVectorizer

client = MongoClient('mongodb://localhost:27017/')
db = client['image_features_db']

