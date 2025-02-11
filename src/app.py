from services.image_feature_extractor import ImageProcessor as image_processor
from services.text_feature_extractor import TextQueryProcessor as text_processor
from pymongo import MongoClient
import gridfs
from sklearn.feature_extraction.text import TfidfVectorizer

client = MongoClient('mongodb://localhost:27017/')
db = client['image_features_db']

