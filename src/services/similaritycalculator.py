from imageprocessor import ImageProcessor as image_processor
from textqueryprocessor import TextQueryProcessor as text_processor
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

vectorizer = TfidfVectorizer()

query = "A family of elephants walking in the forest"
query_features = text_processor.preprocessing_query(query)

client = MongoClient("mongodb://localhost:27017/")
db = client["image_features_db"]
collection = db["features"]

image_collection = db["features"].find()
image_features = []
image_ids = []
for image in image_collection:
    image_features.append(image['feature'])
    image_ids.append(image['_id'])
    
image_features = np.array(image_features)
if query_features.ndim == 1:
    query_features = query_features.reshape(1, -1)

similarities = cosine_similarity(query_features, image_features)

K = 5
nearest_neighbors_indices = np.argsort(similarities[0])[-K:][::-1]
nearest_neighbors = [(image_ids[idx], similarities[0][idx]) for idx in nearest_neighbors_indices]

for neighbor in nearest_neighbors:
    print(f"Image ID: {neighbor[0]}, Similarity: {neighbor[1]}")





# image_features, image_ids = get_image_features_and_ids(db, 'features')

# if image_features is not None:
#     if isinstance(query_features, list):
#         query_features = np.array(query_features)

#     if query_features.ndim == 1:
#         query_features = query_features.reshape(1,-1)

#     similarities = cosine_similarity(query_features, image_features)
# else:
#     print("No image features retrieved.")