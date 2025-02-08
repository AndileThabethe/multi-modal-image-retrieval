from services.imageprocessor import *
from pymongo import MongoClient
import gridfs

client = MongoClient('mongodb://localhost:27017/')
db = client['image_features_db']
fs = gridfs.GridFS(db)

